import drl
import numpy as np
import pathlib
import gym.wrappers
import argparse
import time

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim

from lib import environ, data, models, common

SAVE_DIR = pathlib.Path("saves")
STOCKS = "data/YNDX_160101_161231.csv"
VAL_STOCKS = "data/YNDX_150101_151231.csv"

BATCH_SIZE = 32
BARS_COUNT = 10

EPS_START = 1.0
EPS_END = 0.1
EPS_STEPS = 1000000

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUTATE = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="Enable cuda", default=False, action="store_true")
    parser.add_argument("--data", default=STOCKS, help=f"Stock file or dir, default={STOCKS}")
    parser.add_argument("--year", type=int, help="Year to train on, overides --data")
    parser.add_argument("--val", default=VAL_STOCKS, help="Validation data, default="+VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = SAVE_DIR / f"simple-{args.run}"
    saves_path.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(args.data)
    val_path = pathlib.Path(args.val)

    if args.year is not None or data_path.is_file():
        if args.year is not None:
            stock_data = data.load_year_data(args.year)
        else:
            stock_data = {"YNDX": data.load_relative(data_path)}
        env = environ.StockEnv(stock_data, bars_count=BARS_COUNT)
        env_tst = environ.StockEnv(stock_data, bars_count=BARS_COUNT)
    elif data_path.is_dir():
        env = environ.StockEnv.from_dir(data_path, bars_count=BARS_COUNT)
        env_tst = environ.StockEnv.from_dir(data_path, bars_count=BARS_COUNT)
    else:
        raise RuntimeError("No data to train on")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    val_data = {"YNDX": data.load_relative(val_path)}

    net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = drl.agent.TargetNet(net)

    selector = drl.actions.EpsilonGreedySelector(epsilon=EPS_START)
    eps_tracker = drl.tracker.EpsilonTracker(selector, EPS_START, EPS_END, EPS_STEPS)

    agent = drl.agent.DQNAgent(net, selector, device)
    rp_buffer = drl.experience.ReplayBuffer(REPLAY_SIZE)
    exp_source = drl.experience.ExperienceSource(env, agent, rp_buffer, REWARD_STEPS, GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss = drl.net.loss.DQNLoss(net, tgt_net.target_model, GAMMA, device)

    writer = SummaryWriter(comment='-'+args.run)
    total_reward = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        eps_tracker.update(frame_idx)
        exp_source.play_steps()
        
        reward, step = exp_source.reward_step()
        
        if reward is not None:
            total_reward.append(reward)
            speed = step / (time.time() - ts)
            ts = time.time()
            m_reward = np.mean(total_reward[-100:])
            print("%d: done %d games, reward %.3f, eps %.2f, speed %.2f f/s"%(frame_idx, len(total_reward), m_reward, selector.epsilon, speed))
            writer.add_scalar("epsilon", selector.epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.run + "-best_%.0f.dat"%m_reward)
                if best_m_reward is not None:
                    print("Best reward update %.3f -> %.3f"%(best_m_reward, m_reward))
                best_m_reward = m_reward

        if len(rp_buffer) < REPLAY_INITIAL:
            continue

        if frame_idx % STATES_TO_EVALUTATE == 0:
            tgt_net.sync()

        optimizer.zero_grad()
        batch = rp_buffer.sample(BATCH_SIZE)
        #loss_t = calc_loss(batch, agent.model, tgt_net.target_model, params.gamma, device)
        loss_t = loss(batch)
        loss_t.backward()
        optimizer.step()

    writer.close()

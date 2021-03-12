from lib import common
from lib import dqn_model
from lib import dqn_extra

import argparse
import time
import numpy as np
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import drl.agent as dag
import drl.actions as dac
import drl.experience as dexp
import drl

import gym

# DEFAULT_ENV_NAME = "CartPole-v0"
# MEAN_REWARD_BOUND = 180.

# GAMMA = 0.99
# BATCH_SIZE = 16
# REPLAY_SIZE = 1000
# REPLAY_START_SIZE = 1000
# LEARNING_RATE = 1e-2
# SYNC_TARGET_FRAMES = 10

# EPSILON_DECAY_LAST_FRAME = 10**5
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.01

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state']
)

def calc_loss (batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_states_v[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    return loss

if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = drl.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    selector = dac.EpsilonGreedySelector()
    eps_tracker = dac.EpsilonTracker(selector, params.epsilon_start, params.epsilon_final, params.epsilon_frames)

    net = dqn_extra.DuelingDQN(input_shape, n_actions).to(device)
    agent = dag.DQNAgent(net, selector, device)
    tgt_net = dag.TargetNet(net)

    buffer = dexp.ReplayBuffer(params.replay_size)
    exp_source = dexp.ExperienceSource(env, agent, buffer, 1, params.gamma)

    writer = SummaryWriter(comment="-" + params.env_name)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    total_reward = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        eps_tracker.decay_eps(frame_idx)
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
                torch.save(net.state_dict(), params.env_name + "-dueling-best_%.0f.dat"%m_reward)
                if best_m_reward is not None:
                    print("Best reward update %.3f -> %.3f"%(best_m_reward, m_reward))
                best_m_reward = m_reward

            if m_reward > params.stop_reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < params.replay_initial:
            continue

        if frame_idx % params.target_net_sync == 0:
            tgt_net.sync()

        optimizer.zero_grad()
        batch = buffer.sample(params.batch_size)
        loss_t = calc_loss(batch, agent.dqn_model, tgt_net.target_model, params.gamma, device)
        loss_t.backward()
        optimizer.step()

    writer.close()
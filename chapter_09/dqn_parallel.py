from lib import common
from lib import dqn_model

import argparse
import time
import numpy as np
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
import drl.agent as dag
import drl.actions as dac
import drl.experience as dexp
import drl

import gym

BATCH_MUL = 4

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state']
)

EpisodeEnded = collections.namedtuple("EpisodeEnded", field_names=['reward', 'step', 'epsilon'])

def play_func (params, net, cuda, exp_queue):
    env = drl.common.atari_wrappers.make_atari(params.env_name, skip_noop=True, skip_maxskip=True)
    env = drl.common.atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)
    env.seed(common.SEED)
    device = torch.device("cuda" if cuda else "cpu")

    selector = drl.actions.EpsilonGreedySelector()
    eps_tracker = drl.tracker.EpsilonTracker(selector, params.epsilon_start, params.epsilon_final, params.epsilon_frames)

    agent = drl.agent.DQNAgent(net, selector, device)

    rp_buffer = drl.experience.ReplayBuffer(params.replay_size)
    exp_source = drl.experience.ExperienceSource(env, agent, rp_buffer, 1, params.gamma)

    idx = 0
    while True:
        idx += 1
        eps_tracker.update(idx/BATCH_MUL)
        exp = exp_source.play_steps()
        exp_queue.put(exp)

        reward, step = exp_source.reward_step()
        if reward is not None:
            exp_queue.put(drl.experience.EpisodeEnded(reward, step))

if __name__ == "__main__":

    mp.set_start_method("spawn")

    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = drl.common.atari_wrappers.make_atari(params.env_name, skip_noop=True, skip_maxskip=True)
    env = drl.common.atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    net = dqn_model.DQN(input_shape, n_actions).to(device)
    tgt_net = dag.TargetNet(net)

    rp_buffer = dexp.ReplayBuffer(params.replay_size)

    writer = SummaryWriter(comment="-" + params.env_name)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    loss = drl.net.loss.DQNLoss(net, tgt_net.target_model, params.gamma, device)

    exp_queue = mp.Queue(maxsize=BATCH_MUL * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue))

    total_reward = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    play_proc.start()

    trainer = drl.net.trainer.ParallelTrainer()
    trainer.add_net(net)
    trainer.add_target_agent(tgt_net)
    trainer.add_buffer(rp_buffer, params.replay_initial)
    trainer.add_tensorboard_writer(writer)

    trainer.run_parallel(exp_queue, optimizer, loss, params.batch_size * BATCH_MUL, params.stop_reward, 100, params.target_net_sync)
    #while True:
    #    reward, step, epsilon = None, None, None
    #    while exp_queue.qsize() > 0:
    #        exp = exp_queue.get()
    #        if isinstance(exp, EpisodeEnded):
    #            reward, step, epsilon = exp.reward, exp.step, exp.epsilon
    #        else:
    #            rp_buffer.append(exp)

    #    if reward is not None:
    #        total_reward.append(reward)
    #        speed = step / (time.time() - ts)
    #        ts = time.time()
    #        m_reward = np.mean(total_reward[-100:])
    #        print("%d: done %d games, reward %.3f, eps %.2f, speed %.2f f/s"%(frame_idx, len(total_reward), m_reward, epsilon, speed))
    #        writer.add_scalar("epsilon", epsilon, frame_idx)
    #        writer.add_scalar("speed", speed, frame_idx)
    #        writer.add_scalar("reward_100", m_reward, frame_idx)
    #        writer.add_scalar("reward", reward, frame_idx)

    #        if best_m_reward is None or best_m_reward < m_reward:
    #            torch.save(net.state_dict(), params.env_name + "-best_%.0f.dat"%m_reward)
    #            if best_m_reward is not None:
    #                print("Best reward update %.3f -> %.3f"%(best_m_reward, m_reward))
    #            best_m_reward = m_reward

    #        if m_reward > params.stop_reward:
    #            print("Solved in %d frames!" % frame_idx)
    #            break

    #    if len(rp_buffer) < params.replay_initial:
    #        continue
    #    if frame_idx % params.target_net_sync == 0:
    #        tgt_net.sync()

    #    optimizer.zero_grad()
    #    batch = rp_buffer.sample(params.batch_size * BATCH_MUL)
    #    #loss_t = calc_loss(batch, agent.model, tgt_net.target_model, params.gamma, device)
    #    loss_t = loss(batch)
    #    loss_t.backward()
    #    optimizer.step()

    play_proc.kill()
    play_proc.join()
    #writer.close()

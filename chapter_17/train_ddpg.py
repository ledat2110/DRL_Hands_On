import os
import time
import math
import gym
import pybullet_envs
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import drl

from tensorboardX import SummaryWriter
from lib import model

ENV_ID = 'MinitaurBulletEnv-v0'
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

TEST_ITERS = 1000

def test_net (net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = drl.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break

    return rewards / count, steps / count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = drl.agent.TargetNet(act_net)
    tgt_crt_net = drl.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment='-ddpg_'+args.name)
    agent = model.DDPGAgent(act_net, device=device)

    exp_source = drl.experience.ExperienceSourceFirstLast(env, agent, GAMMA)
    rp_buffer = drl.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    best_reward = None

    with drl.tracker.RewardTracker(writer) as tracker:
        with drl.tracker.TBMeanTracker (writer, 10) as tb_tracker:
            while True:
                step_idx += 1
                rp_buffer.populate(1)

                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if len(rp_buffer) < REPLAY_INITIAL:
                    continue

                batch = rp_buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask_v, last_states_v = drl.experience.unpack_batch_dqn(batch, device)

                # train critic
                crt_opt.zero_grad()

                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask_v] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA

                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()

                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, step_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), step_idx)

                # train actor
                act_opt.zero_grad()

                cur_actions_v = act_net(states_v)
                actor_loss_v = - crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()

                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, step_idx)

                tgt_act_net.alpha_sync(1 - 1e-3)
                tgt_crt_net.alpha_sync(1 - 1e-3)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time() - ts, reward, step))
                    writer.add_scalar("test_reward", reward, step_idx)
                    writer.add_scalar("test_step", reward, step_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                            name = "best_%+.3f_%d.dat"%(reward, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = reward

    pass

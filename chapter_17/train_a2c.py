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
REWARD_STEP = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

TEST_ITERS = 1000

def test_net (net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = drl.agent.float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break

    return rewards / count, steps / count

def cal_logprob (mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))

    return p1 + p2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    net = model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)

    writer = SummaryWriter(comment='-a2c_'+args.name)
    agent = model.A2CAgent(net, device)

    exp_source = drl.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEP)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None

    with drl.tracker.RewardTracker(writer) as tracker:
        with drl.tracker.TBMeanTracker (writer, 10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):

                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = test_net(net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time() - ts, reward, step))
                    writer.add_scalar("test_reward", reward, step_idx)
                    writer.add_scalar("test_step", step, step_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                            name = "best_%+.3f_%d.dat"%(reward, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = reward

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = drl.experience.unpack_batch_a2c(batch, lambda x:net(x)[2], device=device, last_val_gamma=GAMMA**REWARD_STEP)
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_val_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * cal_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()

                ent_v = - (torch.log(2 * math.pi * var_v) + 1) / 2
                loss_entropy_v = ENTROPY_BETA * ent_v.mean()

                loss_v = loss_policy_v + loss_val_v + loss_entropy_v
                loss_v.backward()

                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", loss_entropy_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_policy_v, step_idx)
                tb_tracker.track("loss", loss_v, step_idx)


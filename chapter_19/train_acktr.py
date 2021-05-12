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
from lib import model, common, kfac

ENV_ID = "HalfCheetahBulletEnv-v0"
GAMMA = 0.99
REWARD_STEP = 5
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3

TEST_ITERS = 100000

ENVS_COUNT = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default="+ENV_ID)
    parser.add_argument("-am", "--act_model", help="The pretrained actor model")
    parser.add_argument("-cm", "--crt_model", help="the pretrained critic model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    envs = [gym.make(args.env) for _ in range(ENVS_COUNT)]
    test_env = gym.make(ENV_ID)

    act_net = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    crt_net = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    if args.act_model:
        act_net.load_state_dict(torch.load(args.act_model))
    if args.crt_model:
        crt_net.load_state_dict(torch.load(args.crt_model))

    writer = SummaryWriter(comment='-a2c_'+args.name)
    agent = model.AgentA2C(act_net, device)

    exp_source = drl.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEP)

    act_optimizer = kfac.KFACOptimizer(act_net.parameters(), lr=LEARNING_RATE_ACTOR)
    crt_optimizer = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE_CRITIC)

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
                    reward, step = common.test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time() - ts, reward, step))
                    writer.add_scalar("test_reward", reward, step_idx)
                    writer.add_scalar("test_step", step, step_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                            name = "best_%+.3f_%d.dat"%(reward, step_idx)
                            val_name = "val"+name
                            fname = os.path.join(save_path, name)
                            val_fname = os.path.join(save_path, val_name)
                            torch.save(act_net.state_dict(), fname)
                            torch.save(crt_net.state_dict(), val_fname)
                        best_reward = reward

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = drl.experience.unpack_batch_a2c(batch, crt_net, device=device, last_val_gamma=GAMMA**REWARD_STEP)
                batch.clear()

                crt_optimizer.zero_grad()
                value_v = crt_net(states_v)
                loss_val_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_val_v.backward()
                crt_optimizer.step()

                act_optimizer.zero_grad()
                mu_v = act_net(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, actions_v)
                loss_policy_v = -log_prob_v.mean()
                loss_entropy_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * torch.exp(act_net.logstd)))).mean()

                loss_v = loss_policy_v + loss_entropy_v
                loss_v.backward()
                act_optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", loss_entropy_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_policy_v, step_idx)
                tb_tracker.track("loss", loss_v, step_idx)


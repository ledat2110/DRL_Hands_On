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
from lib import model, common

ENV_ID = "HalfCheetahBulletEnv-v0"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 100000

def calc_adv_ref (trajectory, net_crt, states_v, device="cpu"):
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()

    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp, ) in zip (reversed(values[:-1]),
            reversed(values[1:]),
            reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae

        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))

    return adv_v.to(device), ref_v.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default="+ENV_ID)
    parser.add_argument("-am", "--act_model", help="the pretrained actor model")
    parser.add_argument("-cm", "--crt_model", help="the pretrained critic model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join("saves", "ppo-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    #envs = [gym.make(args.env) for _ in range(ENVS_COUNT)]
    env = gym.make(args.env)
    test_env = gym.make(args.env)

    act_net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    if (args.act_model):
        act_net.load_state_dict(torch.load(args.act_model))
    if (args.crt_model):
        crt_net.load_state_dict(torch.load(args.crt_model))

    writer = SummaryWriter(comment='-a2c_'+args.name)
    agent = model.AgentA2C(act_net, device)

    exp_source = drl.experience.ExperienceSource(env, agent, steps_count=1)

    act_optimizer = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_ACTOR)
    crt_optimizer = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = []
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
                            val_name = "val_"+name
                            fname = os.path.join(save_path, name)
                            val_fname = os.path.join(save_path, val_name)
                            torch.save(act_net.state_dict(), fname)
                            torch.save(crt_net.state_dict(), val_fname)
                        best_reward = reward

                trajectory.append(exp)
                if len(trajectory) < TRAJECTORY_SIZE:
                    continue

                traj_states = [t[0].state for t in trajectory]
                traj_actions = [t[0].action for t in trajectory]
                traj_states_v = torch.FloatTensor(traj_states).to(device)
                traj_actions_v = torch.FloatTensor(traj_actions).to(device)
                traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, crt_net, traj_states_v, device)

                mu_v = act_net(traj_states_v)
                old_logprob_v = drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, traj_actions_v)
                traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
                traj_adv_v /= torch.std(traj_adv_v)

                trajectory = trajectory[:-1]
                old_logprob_v = old_logprob_v[:-1].detach()

                sum_loss_value = 0.0
                sum_loss_policy = 0.0
                count_steps = 0

                for epoch in range(PPO_EPOCHES):
                    for batch_ofs in range(0, len(trajectory), PPO_BATCH_SIZE):
                        batch_l = batch_ofs + PPO_BATCH_SIZE
                        states_v = traj_states_v[batch_ofs:batch_l]
                        actions_v = traj_actions_v[batch_ofs:batch_l]
                        batch_adv_v = traj_adv_v[batch_ofs:batch_l].unsqueeze(-1)
                        batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                        batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                        crt_optimizer.zero_grad()
                        value_v = crt_net(states_v)
                        loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                        loss_value_v.backward()
                        crt_optimizer.step()

                        act_optimizer.zero_grad()
                        mu_v = act_net(states_v)
                        logprob_pi_v = drl.common.utils.cal_cont_logprob(mu_v, act_net.logstd, actions_v)
                        ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                        surr_obj_v = batch_adv_v * ratio_v
                        c_ratio_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                        clipped_surr_v = batch_adv_v * c_ratio_v
                        loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                        loss_policy_v.backward()
                        act_optimizer.step()

                        sum_loss_value += loss_value_v.item()
                        sum_loss_policy += loss_policy_v.item()
                        count_steps += 1

                trajectory.clear()
                tb_tracker.track("advantage", traj_adv_v.mean(), step_idx)
                tb_tracker.track("values", traj_ref_v.mean(), step_idx)
                tb_tracker.track("loss_policy", sum_loss_policy / count_steps, step_idx)
                tb_tracker.track("loss_value", sum_loss_value / count_steps, step_idx)
                tb_tracker.track("loss", (sum_loss_value + sum_loss_policy) / count_steps, step_idx)


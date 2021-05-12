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
BATCH_SIZE = 64
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
SAC_ENTROPY_ALPHA = 0.1

TEST_ITERS = 10000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default="+ENV_ID)
    parser.add_argument("-am", "--act_model", help="The pretrained actor model")
    parser.add_argument("-cm", "--crt_model", help="the pretrained critic model")
    parser.add_argument("-tw", "--twinq", help="The pretrained twin q model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join("saves", "sac-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    act_net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.ModelCritic(env.observation_space.shape[0]).to(device)
    twinq_net = model.ModelSACTwinQ(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    print(twinq_net)
    if args.act_model:
        act_net.load_state_dict(torch.load(args.act_model))
    if args.crt_model:
        crt_net.load_state_dict(torch.load(args.crt_model))
    if args.twinq:
        twinq_net.load_state_dict(torch.load(args.twinq))

    tgt_crt_net = drl.agent.TargetNet(crt_net)
    writer = SummaryWriter(comment='-sac_'+args.name)
    agent = model.AgentDDPG(act_net, device=device)

    exp_source = drl.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    rp_buffer = drl.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

    act_optimizer = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_ACTOR)
    crt_optimizer = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE_CRITIC)
    twinq_optimizer = optim.Adam(twinq_net.parameters(), lr=LEARNING_RATE_CRITIC)


    frame_idx = 0
    best_reward = None
    with drl.tracker.RewardTracker(writer) as tracker:
        with drl.tracker.TBMeanTracker (writer, 10) as tb_tracker:
            while True:
                frame_idx += 1
                rp_buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(rp_buffer) < REPLAY_INITIAL:
                    continue

                batch = rp_buffer.sample(BATCH_SIZE)
                states_v, actions_v, ref_vals_v, ref_q_v = drl.experience.unpack_batch_sac(batch, crt_net, twinq_net, act_net, GAMMA, SAC_ENTROPY_ALPHA, device)
                tb_tracker.track("ref_v", ref_vals_v.mean(), frame_idx)
                tb_tracker.track("ref_q", ref_q_v.mean(), frame_idx)

                # Train twinq
                twinq_optimizer.zero_grad()
                q1_v, q2_v = twinq_net(states_v, actions_v)
                q1_loss_v = F.mse_loss(q1_v.squeeze(), ref_q_v.detach())
                q2_loss_v = F.mse_loss(q2_v.squeeze(), ref_q_v.detach())
                q_loss_v = q1_loss_v + q2_loss_v
                q_loss_v.backward()
                twinq_optimizer.step()
                tb_tracker.track("loss_q1", q1_loss_v, frame_idx)
                tb_tracker.track("loss_q2", q2_loss_v, frame_idx)

                # Train critic
                crt_optimizer.zero_grad()
                val_v = crt_net(states_v)
                v_loss_v = F.mse_loss(val_v.squeeze(), ref_vals_v.detach())
                v_loss_v.backward()
                crt_optimizer.step()
                tb_tracker.track("loss_v", v_loss_v, frame_idx)

                # Train Actor
                act_optimizer.zero_grad()
                acts_v = act_net(states_v)
                q_out_v, _ = twinq_net(states_v, acts_v)
                act_loss = -q_out_v.mean()
                act_loss.backward()
                act_optimizer.step()
                tb_tracker.track("loss_act", act_loss, frame_idx)

                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    reward, step = common.test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d"%(time.time() - ts, reward, step))
                    writer.add_scalar("test_reward", reward, frame_idx)
                    writer.add_scalar("test_step", step, frame_idx)

                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f"%(best_reward, reward))
                            name = "best_%+.3f_%d.dat"%(reward, frame_idx)
                            val_name = "val-"+name
                            twin_name = "twin-"+name
                            fname = os.path.join(save_path, name)
                            val_fname = os.path.join(save_path, val_name)
                            twin_fname = os.path.join(save_path, twin_name)
                            torch.save(act_net.state_dict(), fname)
                            torch.save(crt_net.state_dict(), val_fname)
                            torch.save(twinq_net.state_dict(), twin_fname)
                        best_reward = reward

    pass

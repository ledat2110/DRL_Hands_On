import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nu

import numpy as np
import drl_lib
import gym
import time
import argparse

from tensorboardX import SummaryWriter
from typing import Optional

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
BASELINE_STEPS = 1000000
GRAD_L2_CLIP = 0.1

ENV_COUNT = 50

def make_env ():
    #env = drl_lib.wrapper.make_atari("PongNoFrameskip-v4", skip_noop=True, skip_maxskip=True)
    #env = drl_lib.wrapper.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)
    env = drl_lib.wrapper.wrap_dqn(gym.make("PongNoFrameskip-v4"))

    return env

class AtariA2C (nn.Module):
    def __init__ (self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
                )

        conv_out_size = drl_lib.utils.get_conv_out_size(self.conv, input_shape)
        self.policy = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
                )

        self.value = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
                )

    def forward (self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(x.size()[0], -1)
        policy = self.policy(conv_out)
        value = self.value(conv_out)
        return policy, value

def a2c_unpack_data (batch, net, device='cpu'):
    states, actions, rewards, dones, last_states = drl_lib.experience.unpack_data(batch)

    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    last_states_v = torch.FloatTensor(last_states).to(device)

    last_vals_v = net(last_states_v)[1]
    last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
    last_vals_np *= GAMMA ** REWARD_STEPS
    rewards[dones==False] += last_vals_np[dones==False]

    ref_vals_v = torch.FloatTensor(rewards).to(device)
    return states_v, actions_t, ref_vals_v

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment="-pong-a2c-"+args.name)

    input_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    net = AtariA2C(input_shape, n_actions).to(device)
    print(net)

    agent = drl_lib.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = drl_lib.experience.MultiExpSource(envs, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []
    with drl_lib.tracker.RewardTracker(writer, stop_reward=18) as tracker:
        with drl_lib.tracker.TBMeanTracker(writer, 10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                reward, step = exp_source.reward_step()
                if reward is not None:
                    end = tracker.update(reward, step_idx)
                    if end:
                        break

                if len(batch) < BATCH_SIZE:
                    continue
                states_v, actions_t, vals_ref_v = a2c_unpack_data(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
  
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()
  
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
  
                  # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                          for p in net.parameters()
                                          if p.grad is not None])
  
                  # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nu.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
                optimizer.step()
                  # get full loss
                loss_v += loss_policy_v
  
                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
                #states_v, actions_t, vals_ref_v = a2c_unpack_data(batch, net, device)
                #batch.clear()

                #optimizer.zero_grad()
                #logits_v, value_v = net(states_v)

                #loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                #log_prob_v = F.log_softmax(logits_v, dim=1)
                #adv_v = vals_ref_v - value_v.detach()
                #log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                #loss_policy_v = -log_prob_actions_v.mean()

                #prob_v = F.softmax(logits_v, dim=1)
                #entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
                #entropy_loss_v = -ENTROPY_BETA * entropy_v

                #loss_policy_v.backward(retain_graph=True)
                #grads = np.concatenate([
                #    p.grad.data.cpu().numpy().flatten()
                #    for p in net.parameters() if p.grad is not None
                #    ])
                #loss_v = loss_value_v + entropy_loss_v

                #loss_v.backward()
                #nu.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
                #optimizer.step()
                #loss_v += loss_policy_v

                ## Kullback_Leibler divergence between the new policy and old policy
                ##new_logits_v = net(states_v)
                ##new_prob_v = F.softmax(new_logits_v, dim=1)
                ##kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
                ##writer.add_scalar("kl", kl_div_v.item(), step_idx)

                #g_l2 = np.sqrt(np.mean(np.square(grads)))
                #g_max = np.max(np.abs(grads))
                #g_var = np.var(grads)

                #tb_tracker.track("advantage", adv_v, step_idx)
                #tb_tracker.track("values", value_v, step_idx)
                #tb_tracker.track("batch_reward", vals_ref_v, step_idx)
                #tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                #tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                #tb_tracker.track("loss_value", loss_value_v, step_idx)
                #tb_tracker.track("loss_total", loss_v, step_idx)
                #tb_tracker.track("grad_l2", g_l2, step_idx)
                #tb_tracker.track("grad_max", g_max, step_idx)
                #tb_tracker.track("grad_var", g_var, step_idx)

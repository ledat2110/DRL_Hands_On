import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nu

import numpy as np
import drl
import gym
import time

from tensorboardX import SummaryWriter
from typing import Optional

GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 10
BASELINE_STEPS = 1000000
GRAD_L2_CLIP = 0.1

ENV_COUNT = 32

def make_env ():
    env = drl.common.atari_wrappers.make_atari("PongNoFrameskip-v4", skip_noop=True, skip_maxskip=True)
    env = drl.common.atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)

    return env

class AtariPGN (nn.Module):
    def __init__ (self, input_shape, n_actions):
        super(AtariPGN, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
                )

        conv_out_size = drl.net.utils.get_conv_out_size(self.conv, input_shape)
        self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
                )

    def forward (self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(x.size()[0], -1)
        fc_out = self.fc(conv_out)
        return fc_out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment="-pong-pg")

    input_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    net = AtariPGN(input_shape, n_actions).to(device)
    print(net)

    agent = drl.agent.PolicyAgent(net, apply_softmax=True, device=device)
    exp_source = drl.experience.MultiExpSource(envs, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0
    reward_sum = 0.0
    baseline_buf = drl.common.utils.MeanBuffer(BASELINE_STEPS)

    m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []
    batch_states, batch_actions, batch_scales = [], [], []
    m_grad_max, m_grad_mean = [], []

    ts = time.time()
    eps_ts = time.time()

    with drl.tracker.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            baseline_buf.add(exp.reward)
            baseline = baseline_buf.mean()

            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            batch_scales.append(exp.reward - baseline)

            rewards, steps = exp_source.reward_step()
            for i, reward in enumerate(rewards):
                if reward is not None:
                    end = tracker.update(reward, step_idx)
                    if end:
                        break

            if len(batch_states) < BATCH_SIZE:
                continue

            train_step_idx += 1
            states_v = torch.FloatTensor(batch_states).to(device)
            batch_actions_t = torch.LongTensor(batch_actions).to(device)

            scale_std = np.std(batch_states)
            batch_scales_v = torch.FloatTensor(batch_scales).to(device)

            optimizer.zero_grad()
            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_scales_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v

            loss_v.backward()
            nu.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
            optimizer.step()

            # Kullback_Leibler divergence between the new policy and old policy
            new_logits_v = net(states_v)
            new_prob_v = F.softmax(new_logits_v, dim=1)
            kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
            writer.add_scalar("kl", kl_div_v.item(), step_idx)

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad**2).mean().sqrt().item()
                grad_count += 1

            writer.add_scalar("baseline", baseline, step_idx)
            writer.add_scalar("entropy", entropy_v.item(), step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
            writer.add_scalar("loss_total", loss_v.item(), step_idx)
            writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)
            writer.add_scalar("policy_loss", loss_policy_v.item(), step_idx, time.time())

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    writer.close()

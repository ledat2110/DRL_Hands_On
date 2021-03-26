import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import drl
import gym
import time

from tensorboardX import SummaryWriter
from typing import Optional

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
REWARD_STEPS = 10

class PGN (nn.Module):
    def __init__ (self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
                )

    def forward (self, x):
        fx = x.float()
        return self.net(fx)

def smooth (old: Optional[float], val: float, alpha: float=0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1 - alpha) * val

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = drl.agent.PolicyAgent(net, apply_softmax=True)
    exp_source = drl.experience.ExperienceSource(env, agent, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    ts = time.time()
    eps_ts = time.time()

    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)

        batch_states.append(exp.state)
        batch_actions.append(exp.action)
        batch_scales.append(exp.reward - baseline)

        reward, step = exp_source.reward_step()
        if reward is not None:
            done_episodes += 1
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])

            now = time.time()
            speed = step / (now - eps_ts)
            eps_ts = time.time()
            elapsed = now - ts
            print("Episodes: %d, Reward: %6.3f, Speed: %6.3f, Elapsed: %6.3f" % (done_episodes, m_reward, speed, elapsed))

            writer.add_scalar("reward", m_reward, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            writer.add_scalar("speed", speed, step_idx)

            if m_reward > 195:
                print("Solved in %d steps and %d episodes" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scales_v = torch.FloatTensor(batch_scales)

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

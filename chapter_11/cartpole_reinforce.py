import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import drl
import numpy as np

from tensorboardX import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

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


def calc_qvals (rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)

    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment='-cartpole-reinforce')

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = drl.agent.PolicyAgent(net, apply_softmax=True)
    exp_source = drl.experience.ExperienceSource(env, agent, steps_count=1, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_reward = []
    done_episodes = 0

    batch_episodes = 0
    cur_rewards = []
    batch_states, batch_actions, batch_qvals = [], [], []

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.done == True:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        reward, step = exp_source.reward_step()
        if reward is not None:
            done_episodes += 1
            total_reward.append(reward)
            m_reward = np.mean(total_reward[-100:])
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d"%(step_idx, reward, m_reward, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", m_reward, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if m_reward > 195:
                print("Solved in %d steps and %d episodes" %(step_idx, done_episodes))
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import drl
import gym

from tensorboardX import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 8

EPSILON_START = 1.0
EPSILON_STOP = 0.02
EPSILON_STEPS = 5000

REPLAY_BUFFER = 50000

class DQN (nn.Module):
    def __init__ (self, input_size, n_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
                )

    def forward (self, x):
        fx = x.float()
        return self.net(fx)

if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-dqn")

    net = DQN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    selector = drl.actions.EpsilonGreedySelector()
    eps_tracker = drl.tracker.EpsilonTracker(selector, EPSILON_START, EPSILON_STOP, EPSILON_STEPS)

    agent = drl.agent.DQNAgent(net, selector)
    tgt_agent = drl.agent.TargetNet(net)

    rp_buffer = drl.experience.ReplayBuffer(REPLAY_BUFFER)
    exp_source = drl.experience.ExperienceSource(env, agent, rp_buffer, 1, GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss = drl.net.loss.DQNLoss(net, tgt_agent.target_model, GAMMA)

    trainer = drl.net.trainer.Trainer()
    trainer.add_net(net)
    trainer.add_tracker(eps_tracker)
    trainer.add_exp_source(exp_source, BATCH_SIZE)
    trainer.add_target_agent(tgt_agent)
    trainer.add_tensorboard_writer(writer)

    trainer.run(optimizer, loss, BATCH_SIZE, 195, 100, 100)
    

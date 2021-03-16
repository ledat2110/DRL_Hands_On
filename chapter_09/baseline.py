from lib import common
from lib import dqn_model

import argparse
import time
import numpy as np
import collections
import random
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import drl

import gym

# DEFAULT_ENV_NAME = "CartPole-v0"
# MEAN_REWARD_BOUND = 180.

# GAMMA = 0.99
# BATCH_SIZE = 16
# REPLAY_SIZE = 1000
# REPLAY_START_SIZE = 1000
# LEARNING_RATE = 1e-2
# SYNC_TARGET_FRAMES = 10

# EPSILON_DECAY_LAST_FRAME = 10**5
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.01

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state']
)

def calc_loss (batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_states_v[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    return loss

class Trainer:
    def __init__ (self):
        self.rewards = []
        self.steps = []
        self.looses = []
        self.epslions = []
        self.fpss = []
        self.trackers = []
        self.iteration = 0
        self.ts_eps = time.time()
        self.ts = time.time()
        self.best_m_reward = None
        self.stop = False

    def add_tracker (self, tracker: drl.tracker.Tracker):
        assert isinstance(tracker, drl.tracker.Tracker)
        self.trackers.append(tracker)

    def add_exp_source (self, exp_source: drl.experience.ExperienceSource, initial_size: int):
        assert isinstance(exp_source, drl.experience.ExperienceSource)
        self.exp_source = exp_source
        self.initial_size = initial_size

    def add_agent (self, agent: drl.agent.BaseAgent):
        assert isinstance(agent, drl.experience.BaseAgent)
        self.agent = agent

    def add_target_agent (self, tgt_agent: drl.agent.TargetNet):
        assert isinstance(tgt_agent, drl.agent.TargetNet)
        self.target_agent = tgt_agent

    def add_tensorboard_writer (self, writer: SummaryWriter):
        assert isinstance(writer, SummaryWriter)
        self.writer = writer

    def _update_tracker (self):
        for tracker in self.trackers:
            tracker.update(self.iteration)

    def _end_of_iteration (self):
        reward, step = self.exp_source.reward_step()

        if reward is not None:
            self.rewards.append(reward)
            self.steps.append(step)
            self._fps(step)
            self._cal_metric()

            self._print_out()

            self._update_best_m_reward()
            self._check_stop_reward()

    def _fps (self, step):
        speed = step / (time.time() - self.ts_eps)
        self.ts_eps = time.time()
        self.fpss.append(speed)

    def _cal_metric (self):
        self.m_reward = np.mean(self.rewards[-100:])
        self.m_fps = np.mean(self.fpss[-100:])
        self.m_step = np.mean(self.steps[-100:])

    def _print_out (self):
        time_elapsed = time.time() - self.ts
        episode = len(self.rewards)

        print("Episoded %d: reward=%.3f, steps=%.3f, speed=%.3f fps, elapsed: %.3f"%(episode, self.m_reward, self.m_step, self.m_fps, time_elapsed))

    def _update_tensorboard (self):
        m_loss = np.mean(self.looses[-100:])
        self.writer.add_scalar("m_reward", self.m_reward, self.iteration)
        self.writer.add_scalar("loss", m_loss, self.iteration)
        self.writer.add_scalar("avg_fps", self.m_fps, self.iteration)

    def _update_best_m_reward (self):
        if self.best_m_reward is None or self.best_m_reward < self.m_reward:
            torch.save(self.agent.model, "best_%.0f.dat"%self.m_reward)
            if self.best_m_reward is not None:
                print("Best mean reward update %.3f -> %.3f"%(self.best_m_reward, self.m_reward))
            self.best_m_reward = self.m_reward

    def _check_stop_reward (self):
        m_reward = np.mean(self.rewards[-100:])
        if m_reward > self.stop_reward:
            self.stop = True

    def run (self, optimizer, loss, batch_size: int, stop_reward: float, tb_iteration: int, sync_iteration: int):
        self.stop_reward = stop_reward
        while True:
            self.iteration += 1
            self._update_tracker()
            self.exp_source.play_steps()

            reward, step = self.exp_source.reward_step()

            if reward is not None:
                self.rewards.append(reward)
                self.steps.append(step)
                self._fps(step)
                self._cal_metric()
                self._print_out()

                self._update_best_m_reward()

                if self.m_reward > stop_reward:
                    print("solved in %d iter"%self.iteration)
                    break

            #self._end_of_iteration()
            #if self.stop:
            #    break

            if len(self.exp_source.buffer) < self.initial_size:
                continue

            if self.iteration % sync_iteration == 0:
                self.target_agent.sync()

            optimizer.zero_grad()
            batch = self.exp_source.buffer.sample(batch_size)
            loss_t = loss(batch)
            #self.looses.append(loss_t.item())
            loss_t.backward()
            optimizer.step()

            #if self.iteration % tb_iteration == 0:
            #    self._update_tensorboard()

        self.writer.close()

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = drl.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    selector = drl.actions.EpsilonGreedySelector()
    eps_tracker = drl.tracker.EpsilonTracker(selector, params.epsilon_start, params.epsilon_final, params.epsilon_frames)

    net = dqn_model.DQN(input_shape, n_actions).to(device)
    agent = drl.agent.DQNAgent(net, selector, device)
    tgt_net = drl.agent.TargetNet(net)

    buffer = drl.experience.ReplayBuffer(params.replay_size)
    exp_source = drl.experience.ExperienceSource(env, agent, buffer, 1, params.gamma)

    writer = SummaryWriter(comment="-" + params.env_name)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    loss = drl.loss.DQNLoss(net, tgt_net.target_model, params.gamma, device)

    trainer = Trainer()
    trainer.add_agent(agent)
    trainer.add_target_agent(tgt_net)
    trainer.add_exp_source(exp_source, params.replay_initial)
    trainer.add_tracker(eps_tracker)
    trainer.add_tensorboard_writer(writer)

    trainer.run(optimizer, loss, params.batch_size, params.stop_reward, 100, params.target_net_sync)

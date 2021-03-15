import gym
import gym.envs.toy_text.frozen_lake
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9

class DiscreteOneHotWrapper (gym.ObservationWrapper):
    def __init__ (self, env: gym.Env):
        super (DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype=np.float32)

    def observation (self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

class Net (nn.Module):
    def __init__ (self, obs_size: int, hidden_size: int, n_actions: int):
        super (Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
                )

    def forward (self, x: torch.Tensor):
        return self.net(x)

Episode = namedtuple("Episode", ['reward', 'steps'])
EpisodeStep = namedtuple("EpisodeStep", ['observation', 'action'])

def iterate_batches (env: gym.Env, net: nn.Module, batch_size: int):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, done, _ = env.step(action)

        episode_reward += reward
        step = EpisodeStep(obs, action)
        episode_steps.append(step)

        if done:
            e = Episode(episode_reward, episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs

def filter_batch (batch: Episode, percentile: float):
    filter_fun = lambda s: s.reward * (GAMMA ** len(s.steps))
    disc_rewards = list(map(filter_fun, batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound

if __name__ == "__main__":
    # env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
    env.spec = gym.spec("FrozenLake-v0")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = DiscreteOneHotWrapper(env)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(comment='-frozenlake-nonslippery')

    solved_episode = 0
    full_batch = []

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_m = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_b = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue

        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]
        
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.3f, rw_bound=%.3f, batch=%d"%(iter_no, loss_v.item(), reward_m, reward_b, len(full_batch)))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        # if reward_m >= 0.8:
        #     solved_episode += 1
        if iter_no > 20000:
            print("out of iteration")
            break
        if reward_m > 0.8:
            print("solved")
            break

    writer.close()
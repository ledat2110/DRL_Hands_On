import gym
import time
import numpy as np
import collections
import pybullet_envs

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim

from tensorboardX import SummaryWriter

NOISE_STD = 0.05
LEARNING_RATE = 0.01
ITERS_PER_UPDATE = 10
MAX_ITERS = 100000
PROCESSES_COUNT = 6

RewardsItem = collections.namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])

def make_env ():
    return gym.make("HalfCheetahBulletEnv-v0")

class Net (nn.Module):
    def __init__ (self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(obs_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, act_size),
                nn.Tanh()
                )

    def forward (self, x):
        return self.net(x)

def evaluate (env, net, device="cpu"):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        action_v = net(obs_v)
        action = action_v.data.cpu().numpy()[0]
        obs, r, done, _ = env.step(action)
        reward += r
        steps += 1
        if done:
            break

    return reward, steps

def sample_noise (net, device='cpu'):
    pos = []
    neg = []
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size()).astype(np.float32)
        noise_t = torch.FloatTensor(noise).to(device)
        pos.append(noise_t)
        neg.append(-noise_t)

    return pos, neg

def eval_with_noise (env, net, noise, noise_std, device='cpu'):
    for p, p_n in zip(net.parameters(), noise):
        p.data += noise_std * p_n
    r,s = evaluate(env, net, device)
    for p, p_n in zip(net.parameters(), noise):
        p.data -= noise_std * p_n
    return r, s

def compute_ranks (x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks (x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def train_step (optimizer, net, batch_noise, batch_reward, writer, step_idx, noise_std):
    weighted_noise = None
    norm_reward = compute_centered_ranks(np.array(batch_reward))

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n

    m_updates = []
    optimizer.zero_grad()
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.grad = -update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)
    optimizer.step()

def worker_func (worker_id, params_queue, rewards_queue, device, noise_std):
    env = make_env()
    net = Net (env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net.eval()

    while True:
        params = params_queue.get()
        if params is None:
            break
        net.load_state_dict(params)
        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = sample_noise(net, device=device)
            pos_reward, pos_steps = eval_with_noise(env, net, noise, noise_std, device)
            neg_reward, neg_steps = eval_with_noise(env, net, neg_noise, noise_std, device)
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward, neg_reward=neg_reward, steps=pos_steps+neg_steps))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(comment='-carpole-es')
    env = make_env()

    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)

    params_queues = [
            mp.Queue(maxsize=1)
            for _ in range(PROCESSES_COUNT)
            ]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    workers = []

    for idx, params_queue in enumerate(params_queues):
        p_args = (idx, params_queue, rewards_queue, device, NOISE_STD)
        proc = mp.Process(target=worker_func, args=p_args)
        proc.start()
        workers.append(proc)

    print("all started!")
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for step_idx in range(MAX_ITERS):
        # broadcasting network params
        params = net.state_dict()
        for q in params_queues:
            q.put(params)
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        results = 0
        batch_steps = 0
        batch_steps_data = []
        while True:
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed)
                noise, neg_noise = sample_noise(net)
                batch_noise.append(noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps
                batch_steps_data.append(reward.steps)

            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break
            time.sleep(0.01)

        m_reward = np.mean(batch_reward)
        train_step(optimizer, net, batch_noise, batch_reward, writer, step_idx, NOISE_STD)
        speed = batch_steps / (time.time() - t_start)
        print("%d: reward=%.2f, speed=%.2f f/s" %(step_idx, m_reward, speed))

    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()



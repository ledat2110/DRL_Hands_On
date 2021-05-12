import os
import gym
import drl_lib
import numpy as np
import argparse
import collections

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 8

GRAD_BATCH = 64
TRAIN_BATCH = 2

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18

def make_env ():
    return drl_lib.wrapper.wrap_dqn(gym.make(ENV_NAME))

TotalReward = collections.namedtuple('TotalReward', field_names='reward')

def grads_func (proc_name, net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = drl_lib.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = drl_lib.experience.MultiExpSource(envs, agent, steps_count=REWARD_STEPS, gamma=GAMMA)

    batch = drl_lib.experience.BatchData(GRAD_BATCH)
    frame_idx = 0
    writer = SummaryWriter(comment=proc_name)

    with drl_lib.tracker.RewardTracker(writer, REWARD_BOUND) as tracker:
        with drl_lib.tracker.TBMeanTracker(writer, 100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                reward, step = exp_source.reward_step()
                if reward:
                    if tracker.update(reward, frame_idx):
                        break

                batch.add(exp)
                if len(batch) < GRAD_BATCH:
                    continue

                data = batch.pg_unpack(lambda x:net(x)[1], GAMMA**REWARD_STEPS, device=device)
                states_v, actions_t, vals_ref_v = data

                batch.clear()

                net.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(GRAD_BATCH), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy = (prob_v * log_prob_v).sum(dim=1).mean()
                loss_entropy_v = ENTROPY_BETA * entropy

                loss_v = loss_policy_v + loss_entropy_v + loss_value_v
                loss_v.backward()

                tb_tracker.track("advantage", adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_reward", vals_ref_v, frame_idx)
                tb_tracker.track("loss_entropy", loss_entropy_v, frame_idx)
                tb_tracker.track("loss_value", loss_value_v, frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v, frame_idx)
                tb_tracker.track("loss", loss_v, frame_idx)

                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                grads = [
                        param.grad.data.cpu().numpy() if param.grad is not None else None
                        for param in net.parameters()
                        ]
                train_queue.put(grads)

    train_queue.put(None)

class AtariA2C (nn.Module):
    def __init__ (self, input_shape, n_actions):
        super (AtariA2C, self).__init__()

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
        conv_out = self.conv(fx).view(fx.size()[0], -1)

        policy = self.policy(conv_out)
        value = self.value(conv_out)

        return policy, value


if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = SummaryWriter(comment=f"-a3c-data_pong_{args.name}")

    env = make_env()
    net = AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []

    for proc_idx in range(PROCESSES_COUNT):
        proc_name = f"-a3c-grad_pong_{args.name}#{proc_idx}"
        p_args = (proc_name, net, device, train_queue)
        data_proc = mp.Process(target=grads_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    frame_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break
            frame_idx += 1
            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad

            if frame_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

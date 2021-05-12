import torch
import torch.nn as nn
import torch.nn.functional as F

import drl
import numpy as np

HID_SIZE = 128

class ModelA2C (nn.Module):
    def __init__ (self, ob_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
                nn.Linear(ob_size, HID_SIZE),
                nn.ReLU(),
                )
        self.mu = nn.Sequential(
                nn.Linear(HID_SIZE, act_size),
                nn.Tanh(),
                )
        self.var = nn.Sequential(
                nn.Linear(HID_SIZE, act_size), 
                nn.Softplus(),
                )
        self.val = nn.Linear(HID_SIZE, 1)

    def forward (self, x):
        base_out = self.base(x)

        mu_out = self.mu(base_out)
        var_out = self.var(base_out)
        val_out = self.val(base_out)

        return mu_out, var_out, val_out

class A2CAgent (drl.agent.BaseAgent):
    def __init__ (self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)

        return actions, agent_states

class DDPGActor (nn.Module):
    def __init__ (self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(obs_size, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, act_size),
                nn.Tanh()
                )

    def forward (self, x):
        return self.net(x)

class DDPGCritic (nn.Module):
    def __init__ (self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
                nn.Linear(obs_size, 400),
                nn.ReLU(),
                )

        self.out_net = nn.Sequential(
                nn.Linear(400 + act_size, 300),
                nn.ReLU(),
                nn.Linear(300, 1)
                )

    def forward (self, x, a):
        obs = self.obs_net(x)
        obs_a = torch.cat([obs, a], dim=1)
        return self.out_net(obs_a)


class DDPGAgent (drl.agent.BaseAgent):
    def __init__ (self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_theta=0.15, ou_sigma=0.2, ou_eps=1.0):
        super(DDPGAgent, self).__init__()
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_eps = ou_eps
        self.agent_state = None

    def initial_state (self):
        return None

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_eps > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_theta * (self.ou_mu - a_state)
                a_state += self.ou_sigma  * np.random.normal(size=action.shape)

                action += self.ou_eps * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)

        return actions, new_a_states

N_ATOMS = 51

class D4PGCritic (nn.Module):
    def __init__ (self, obs_size, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
                nn.Linear(obs_size, 400),
                nn.ReLU(),
                )

        self.out_net = nn.Sequential(
                nn.Linear(400 + act_size, 300),
                nn.ReLU(),
                nn.Linear(300, n_atoms)
                )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer ("supports", torch.arange(v_min, v_max + delta, delta))

    def forward (self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q (self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

class D4PGAgent (drl.agent.BaseAgent):
    def __init__ (self, net, device="cpu", epsilon=0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__ (self, states, agent_states):
        states_v = drl.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)

        actions = mu_v.data.cpu().numpy()
        actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)

        return actions, agent_states

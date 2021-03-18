import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import drl

class SimpleFFDQN (nn.Module):
    def __init__ (self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
                nn.Linear(obs_len, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
                )

        self.fc_adv = nn.Sequential(
                nn.Linear(obs_len, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, actions_n)
                )

    def forward (self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)

        return val + (adv - adv.mean(dim=1, keepdim=True))


class Conv1DDQN (nn.Module):
    def __init__ (self, shape, actions_n):
        super(Conv1DDQN, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv1d(shape[0], 128, 5),
                nn.ReLU(),
                nn.Conv1d(128, 128, 5),
                nn.ReLU(),
                )

        out_size = drl.net.utils.get_conv_out_size(self.conv, shape)
        self.fc_val = nn.Sequential(
                nn.Linear(out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
                )

        self.fc_adv = nn.Sequential(
                nn.Linear(out_size, 512),
                nn.ReLU(),
                nn.Linear(512, actions_n)
                )

    def forward (self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)

        return val + (adv - adv.mean(dim=1, keepdim=True))

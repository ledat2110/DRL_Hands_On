import ptan
import torch.nn as nn
import copy

class DQNNet (nn.Module):
    def __init__ (self):
        super(DQNNet, self).__init__()
        self.ff = nn.Linear(5, 3)

    def forward (self, x):
        return self.ff(x)


if __name__ == "__main__":
    net = DQNNet()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    print("main net", net.ff.weight)
    print("target net", tgt_net.target_model.ff.weight)
    net.ff.weight.data += 1.0
    print("after update")
    print("main net", net.ff.weight)
    print("target net", tgt_net.target_model.ff.weight)
    tgt_net.sync()
    print("after sync")
    print("main net", net.ff.weight)
    print("target net", tgt_net.target_model.ff.weight)
    
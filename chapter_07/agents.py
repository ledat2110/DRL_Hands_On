import torch
import torch.nn as nn
import ptan
import numpy as np

class DQNNet (nn.Module):
    def __init__ (self, actions: int):
        super(DQNNet, self).__init__()
        self.actions = actions
    
    def forward (self, x):
        return torch.eye(x.size()[0], self.actions)

class PolicyNet (nn.Module):
    def __init__ (self, actions: int):
        super(PolicyNet, self).__init__()
        self.actions = actions

    def forward (self, x):
        shape = (x.size()[0], self.actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1
        res[:, 1] = 1
        return res

if __name__ == "__main__":
    state = torch.zeros(2, 10)

    net = DQNNet(actions=3)
    net_out = net(state)
    print("dqn net out", net_out)

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
    ag_out = agent(state)
    print("dqn agent out", ag_out)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0)
    agent.action_selector = selector
    ag_out = agent(state)
    print("epsilon 0", ag_out)

    selector.epsilon = 0.5
    ag_out = agent(state)
    print("epsilon 0.5", ag_out)

    selector.epsilon = 1
    ag_out = agent(state)
    print("epsilon 1", ag_out)

    net = PolicyNet(actions=5)
    net_out = net(state)
    print("policy net out", net_out)

    prob_softmax = torch.nn.functional.softmax(net_out, dim=1)
    print("prob softmax", prob_softmax)

    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
    ag_out = agent(state)
    print("policy agent out", ag_out)



    
    
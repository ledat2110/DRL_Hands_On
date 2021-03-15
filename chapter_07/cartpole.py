import gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


HIDDEN_SIZE = 128
BATCH_SIZE = 16
TGT_NET_SYNC = 10
GAMMA = 0.9
REPLAY_SIZE = 1000
LR = 1e-3
EPS_DECAY=0.99

class Net (nn.Module):
    def __init__ (self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward (self, x):
        return self.net(x.float())

@torch.no_grad()
def unpack_batch (batch, net, gamma):
    states = []
    actions = []
    rewards = []
    done_masks = []
    lats_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            lats_states.append(exp.state)
        else:
            lats_states.append(exp.last_state)

    states_v = torch.tensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    lats_states_v = torch.tensor(lats_states)
    lats_states_q_v = net(lats_states_v)
    best_last_q_v = torch.max(lats_states_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v

if __name__ == "__main__":
        env = gym.make("CartPole-v0")
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        net = Net(obs_size, HIDDEN_SIZE, n_actions)
        tgt_net = ptan.agent.TargetNet(net)

        selector = ptan.actions.ArgmaxActionSelector()
        selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1, selector=selector)
        agent = ptan.agent.DQNAgent(net, selector)

        exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA)
        buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
        optimizer = optim.Adam(net.parameters(), LR)

        step = 0
        episode = 0
        solved = False
        while True:
            step += 1

            # take the action to the env 
            buffer.populate(1)

            # if the env done return the reward
            for reward, steps in exp_source.pop_rewards_steps():
                episode += 1
                print("%d: episode %d done, %d steps, reward=%.3f, epsilon=%.2f" % (step, episode, steps, reward, selector.epsilon))
                solved = reward > 150

            if solved:
                print("Congrats!")
                break
            
            # get the bach size of experience
            if len(buffer) < 2*BATCH_SIZE:
                continue
            batch = buffer.sample(BATCH_SIZE)

            # train the model
            states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA)
            optimizer.zero_grad()
            q_v = net(states_v)
            q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            loss_v = F.mse_loss(q_v, tgt_q_v)
            loss_v.backward()
            optimizer.step()
            selector.epsilon *= EPS_DECAY

            if step % TGT_NET_SYNC == 0:
                tgt_net.sync()
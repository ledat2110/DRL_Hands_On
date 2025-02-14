import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent():
    def __init__ (self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env (self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = self.env.reset() if done else new_state

        return old_state, action, reward, new_state

    def best_value_and_action (self, state: int):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        
        return best_value, best_action

    def value_update (self, s: int, a: int, r: int, next_s: int):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = (1 - ALPHA) * old_v + ALPHA * new_v

    def play_episode (self, env: gym.Env):
        total_reward = 0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            if done:
                break
            state = new_state

        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f"%(best_reward, reward))
            best_reward = reward
        
        if reward > 0.8:
            print("Solved in %d iterations!"%iter_no)
            break

    writer.close()
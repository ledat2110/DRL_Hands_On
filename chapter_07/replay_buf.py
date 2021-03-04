import numpy as np
import gym
import ptan

from typing import List, Optional, Any, Tuple

class ToyEnv (gym.Env):
    def __init__ (self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(3)
        self.step_index = 0

    def reset (self):
        self.step_index = 0
        return self.step_index

    def step (self, action):
        done = self.step_index == 10
        if done:
            return self.step_index % self.observation_space.n, 0, done, {}
        self.step_index += 1
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, {}

class DullAgent (ptan.agent.BaseAgent):
    def __init__ (self, action: int):
        self.action = action

    def __call__ (self, observations: List[Any], state: Optional[List]=None) -> Tuple[List[int], Optional[List]]:
        return [self.action for _ in observations], state

if __name__ == "__main__":
    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)
    print("len buffer", len(buffer))
    
    for step in range(6):
        buffer.populate(1)
        if len(buffer) < 5:
            continue
        batch = buffer.sample(4)
        print("%d train time, %d batch samples: "% (step, len(batch)))
        for s in batch:
            print(s)
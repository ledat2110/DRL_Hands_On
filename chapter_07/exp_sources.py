import numpy as np
import gym
import ptan
import drl_lib

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

class DullAgent1 (drl_lib.agent.BaseAgent):
    def __init__ (self, action: int):
        self.action = action

    def __call__ (self, observations: List[Any]) -> Tuple[List[int]]:
        return self.action


if __name__ == "__main__":
    env = ToyEnv()
    agent = DullAgent(action=1)
    # exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=2)
    # for idx, exp in enumerate(exp_source):
    #     if idx > 15:
    #         break
    #     print(idx, exp)

    # exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=4)
    # print(next(iter(exp_source)))

    print("multi env")
    exp_source = ptan.experience.ExperienceSource([ToyEnv(), ToyEnv()], agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 20:
            break
        print(idx, exp, type(exp))

    print("exp source first last")
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=2)
    for idx, exp in enumerate(exp_source):
        print(exp_source.pop_rewards_steps())
        if idx > 10:
            break 
        print(idx, exp)
    print("multi env")
    exp_source = ptan.experience.ExperienceSourceFirstLast(env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=2, gamma=1.0)
    for idx, exp in enumerate(exp_source):
        print(exp)
        print(exp_source.pop_rewards_steps())
        if idx > 50:
            break

    agent = DullAgent1(action=1)
    exp_source = drl_lib.experience.MultiExpSource([ToyEnv(), ToyEnv()], agent=agent, steps_count=2, gamma=1.0)
    for idx, exp in enumerate(exp_source):
        print(exp)
        reward, step = exp_source.reward_step()
        print(reward, step)
        if idx > 50:
            break
    print("single exp source")
    exp_source = drl_lib.experience.ExperienceSource(ToyEnv(), agent=agent, steps_count=2, gamma=1.0)
    for idx, exp in enumerate(exp_source):
        reward, step = exp_source.reward_step()
        print(reward, step)
        if idx > 10:
            break
        print(exp)


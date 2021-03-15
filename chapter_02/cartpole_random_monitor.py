import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recording", force=True)

    total_reward = 0
    total_step = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        total_reward += reward
        total_step += 1

        if done:
            break

    print("Episode done %d episodes, reward %.2f" % (total_step, total_reward))
    env.close()
    env.env.close()

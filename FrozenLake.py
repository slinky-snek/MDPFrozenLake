import gym
import numpy as np


# Q-Learning = doesn't work
# suggest decay policy for epsilon
def q_learning():
    Q = np.zeros((16, 4))  # Q table
    epsilon = 0.2  # exploration rate
    alpha = 0.1  # learning rate
    gamma = 0.99  # reward discount
    env = gym.make('FrozenLake-v0')
    for episode in range(10):
        observation_orig = env.reset()
        episode_reward = 0
        for time_step in range(100):
            env.render()
            print(observation_orig)
            if np.random.uniform(0, 1) < epsilon:
                # select random action
                action = env.action_space.sample()
            else:
                # select best action for state
                action = np.argmax(Q[observation_orig, :])
            observation_next, reward, done, info = env.step(action)
            Q[observation_orig, action] = Q[observation_orig, action] * \
                (1 - alpha) + alpha * (reward + gamma * np.max(Q[observation_next, :]))
            observation_orig = observation_next
            episode_reward += reward
            if done:
                print("Episode finished after {} time steps".format(time_step + 1))
                print("Cumulative reward is {}".format(episode_reward))
                break
    env.close()


# SARSA = doesn't work
def sarsa():
    env = gym.make('FrozenLake-v0')
    for episode in range(10):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


q_learning()

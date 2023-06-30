# Aaron Barnett

import gym
import numpy as np
import random


# Dyna-Q
def dyna_q():
    epsilon = 0.9  # exploration rate
    decay = 0.0000075
    min_epsilon = 0.01
    alpha = 0.4  # learning rate
    gamma = 0.99  # reward discount
    planning_steps = 5  # planning steps per episode
    max_time = 100  # max time steps per episode
    max_episodes = 200000
    env = gym.make('FrozenLake-v0')
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # Q table
    world_model = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    reward_model = np.zeros(env.observation_space.n)
    for episode in range(max_episodes):
        # Train
        observation_orig = env.reset()
        epsilon -= decay
        epsilon = max(epsilon, min_epsilon)
        history = []
        for time_step in range(max_time):
            # Q-Learning
            # env.render()
            if np.random.uniform(0, 1) < epsilon:
                # select random action
                action = env.action_space.sample()
            else:
                # select best action for state
                action = np.argmax(Q[observation_orig, :])
            observation_next, reward, done, info = env.step(action)
            if observation_next == 15:
                Q[observation_orig, action] += alpha * (reward - Q[observation_orig, action])
            else:
                Q[observation_orig, action] += alpha * (reward + gamma * np.max(Q[observation_next, :]) - Q[observation_orig, action])
                world_model[observation_orig, action, observation_next] += 1  # model transition function
                reward_model[observation_next] = reward  # model reward function
                history.append((observation_orig, action))  # store past states and actions
                observation_orig = observation_next
            # Q-Planning
            for steps in range(planning_steps):
                state, action = random.choice(history)
                # Find next probable state using model
                max_prob = 0
                next_state = state
                for temp_state in range(len(world_model[state, action])):
                    visited = world_model[state, action, temp_state]
                    prob = visited / sum(world_model[state, action])
                    if prob > max_prob:
                        next_state = temp_state
                        max_prob = prob
                reward = reward_model[next_state]
                Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            if done:
                break
        # Test
        if episode % 2000 == 0:
            cumulative_reward = 0
            for ep in range(10):
                observation_orig = env.reset()
                for step in range(100):
                    action = np.argmax(Q[observation_orig, :])
                    observation_next, reward, done, info = env.step(action)
                    observation_orig = observation_next
                    cumulative_reward += reward
            print(cumulative_reward / 10)
    env.close()


# Q-Learning
def q_learning():
    epsilon = 0.9  # exploration rate
    decay = 0.0000075
    min_epsilon = 0.01
    alpha = 0.4  # learning rate
    gamma = 0.99  # reward discount
    max_time = 100  # max time steps per episode
    max_episodes = 200000
    env = gym.make('FrozenLake-v0')
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # Q table
    for episode in range(max_episodes):
        # Train
        observation_orig = env.reset()
        epsilon -= decay
        epsilon = max(epsilon, min_epsilon)
        for time_step in range(max_time):
            # env.render()
            if np.random.uniform(0, 1) < epsilon:
                # select random action
                action = env.action_space.sample()
            else:
                # select best action for state
                action = np.argmax(Q[observation_orig, :])
            observation_next, reward, done, info = env.step(action)
            if observation_next == 15:
                Q[observation_orig, action] += alpha * (reward - Q[observation_orig, action])
            else:
                Q[observation_orig, action] += alpha * (reward + gamma * np.max(Q[observation_next, :]) - Q[observation_orig, action])
                observation_orig = observation_next
            if done:
                break
        # Test
        if episode % 2000 == 0:
            cumulative_reward = 0
            for ep in range(10):
                observation_orig = env.reset()
                for step in range(100):
                    action = np.argmax(Q[observation_orig, :])
                    observation_next, reward, done, info = env.step(action)
                    observation_orig = observation_next
                    cumulative_reward += reward
            print(cumulative_reward / 10)
    env.close()


# SARSA
def sarsa():
    epsilon = 0.9  # exploration rate
    decay = 0.0000075
    min_epsilon = 0.01
    alpha = 0.4  # learning rate
    gamma = 0.99  # reward discount
    max_time = 100  # max time steps per episode
    max_episodes = 200000
    env = gym.make('FrozenLake-v0')
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # Q table
    for episode in range(max_episodes):
        # Train
        observation_orig = env.reset()
        epsilon -= decay
        epsilon = max(epsilon, min_epsilon)
        for time_step in range(max_time):
            # env.render()
            if np.random.uniform(0, 1) < epsilon:
                # select random action
                action = env.action_space.sample()
            else:
                # select best action for state
                action = np.argmax(Q[observation_orig, :])
            observation_next, reward, done, info = env.step(action)
            if observation_next == 15:
                Q[observation_orig, action] += alpha * (reward - Q[observation_orig, action])
            else:
                if np.random.uniform(0, 1) < epsilon:
                    # select random action
                    action_next = env.action_space.sample()
                else:
                    # select best action for state
                    action_next = np.argmax(Q[observation_next, :])
                Q[observation_orig, action] += alpha * (reward + gamma * Q[observation_next, action_next] - Q[observation_orig, action])
                observation_orig = observation_next
            if done:
                break
        # Test
        if episode % 2000 == 0:
            cumulative_reward = 0
            for ep in range(10):
                observation_orig = env.reset()
                for step in range(100):
                    action = np.argmax(Q[observation_orig, :])
                    observation_next, reward, done, info = env.step(action)
                    observation_orig = observation_next
                    cumulative_reward += reward
            print(cumulative_reward / 10)
    env.close()


# dyna_q()
# q_learning()
sarsa()

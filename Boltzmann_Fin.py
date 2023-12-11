#!/usr/bin/env python
# coding: utf-8

# In[21]:


import gym
import numpy as np
import random
import time

# Set the seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

# Create the environment
env = gym.make('Taxi-v3', render_mode="rgb_array")

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate

# Epsilon-greedy action selection
def epsilon_greedy_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration: choose random action
    else:
        return np.argmax(q_table[state, :])  # Exploitation: choose best action

# Boltzmann action selection
def boltzmann_action(state):
    temperature = 1.0  # Higher temperature for more exploration
    probabilities = np.exp(q_table[state, :] / temperature) / np.sum(np.exp(q_table[state, :] / temperature))
    action = np.random.choice(num_actions, p=probabilities)
    return action

# Implement your training loop here using either epsilon-greedy or Boltzmann
# For example, a Q-learning implementation follows in the next step.


# In[24]:


# Q-learning algorithm
num_episodes = 10000
rewards_all_episodes = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = boltzmann_action(state)  # Using Boltzmann action selection
        
        next_state, reward, done, terminated, _ = env.step(action)
        
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        total_reward += reward
        state = next_state
    
    rewards_all_episodes.append(total_reward)


# Calculate average reward per episodes
rewards_per_episodes = []
episode_rewards = []
count = 0

for reward in rewards_all_episodes:
    count += 1
    episode_rewards.append(reward)
    
    if count % 1000 == 0:
        avg_reward = sum(episode_rewards) / 1000
        rewards_per_episodes.append(avg_reward)
        episode_rewards = []


# Print average rewards per episodes
print("********Episodes Average Reward********\n")
for idx, reward in enumerate(rewards_per_episodes):
    print((idx + 1) * 1000, ": ", reward)

        
# Print the Q-table
print("\n\n********Q_table********\n")
print(q_table)


# In[23]:



# Run episodes to check the goal
num_test_episodes = 10



for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    print("Episode", episode + 1)
    time.sleep(1)
    while not done:
        env.render()  # Draw the current state
        time.sleep(0.3)
        action = np.argmax(q_table[state, :])  # Choose action based on learned Q-values
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        
        env.render()  # Draw the new state after taking action
        action = np.argmax(q_table[state, :])  # Take the highest action at the current state
        next_state, reward, done, _, _ = env.step(action)
        
        if reward == 20:
            print("Success")
            time.sleep(3)
        else:
            print("Failed")
            time.sleep(3)
        break
        
        
    env.close()


# In[ ]:





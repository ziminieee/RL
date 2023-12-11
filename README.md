# Taxi 구현하기

강화학습 문제 해결이 익숙하지 않아 꽤나 오랜 시간 과제를 진행했음에도 불구하고 제대로 된 결과값이 나오지 않아 애를 많이 먹었습니다. 할 수 있는 부분까지 최대한 작성해보았으며, 결과값이 이상하여 Jupyter Notebook과 Pycharm, 구글 Colab 등에서 여러 번 구현해보았습니다. 하단의 내용은 Jupyter에서 작성한 코드입니다.

> 먼저 Open AI Gymnaisum을 설치 및 불러오기

    !pip install gymnasium
    import gymnasium as gym
    import numpy as np
    import time
    from Ipython.display import clear_output


> Gym 환경설정 & Q테이블 초기화

    env = gym.make('Taxi-v3', render_mode = "rgb_arry")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

> Hyperparameter 설정 (alpha : 학습률, gamma : 할인율)
> 
    alpha = 0.1 
    gamma = 0.9

> Epsilon-Greedy 탐험 정책 설정

    epsilon = 1.0

> Boltzmann Exploration 탐험 정책 설정

    temperature = 1.0 

> Epsilon-Greedy Action 선택 (탐색:무작위 선택, 탐험:최상값 선택)

    def epsilon_greedy_action(state):
        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample() 
    else:
        return np.argmax(q_table[state, :])  

> Boltzmann Action 선택

    def boltzmann_action(state):
        temperature = 1.0  
        probabilities = np.exp(q_table[state, :] / temperature) / np.sum(np.exp(q_table[state, :] / temperature))
        action = np.random.choice(num_actions, p=probabilities)
        return action

> 1) Epsilon-Greedy Action 기반 Q-Learning 실시

    num_episodes = 40000
    rewards_all_episodes = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = epsilon_greedy_action(state)  
        
        next_state, reward, done, terminated, _ = env.step(action)
        
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        total_reward += reward
        state = next_state
    
    rewards_all_episodes.append(total_reward)
    successes.append(total_reward == 20)

> 2000개 단위 에피소드별로 평균 보상 계산

    rewards_per_episodes = []
    episode_rewards = []
    count = 0

for reward in rewards_all_episodes:
    count += 1
    episode_rewards.append(reward)
    
    if count % 2000 == 0:
        avg_reward = sum(episode_rewards) / 2000
        rewards_per_episodes.append(avg_reward)
        episode_rewards = []


> 2000개 단위 에피소드별 평균 보상 출력

    print("********2000 Episodes Average Reward********\n")
    for idx, reward in enumerate(rewards_per_episodes):
    print((idx + 1) * 2000, ": ", reward)

> 업데이트된 Q-table 출력

    print("\n\n********Q_table********\n")
    print(q_table)

> 학습 종료 후 목적 달성 여부 확인을 위해 Episode 실행

    num_test_episodes = 10

    for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    print("Episode", episode + 1)
    time.sleep(1)
    while not done:
        env.render()  # Draw the current state
        time.sleep(0.3)
        action = np.argmax(q_table[state, :])  
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        
        env.render() 
        action = np.argmax(q_table[state, :])  
        next_state, reward, done, _, _ = env.step(action)
        
        if reward == 20:
            print("Success")
            time.sleep(3)
        else:
            print("Failed")
            time.sleep(3)
        break
        
        
    env.close()


> 2) Boltzmann Action 기반 Q-Learning 실시

    num_episodes = 40000
    rewards_all_episodes = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = boltzmann_action(state) 
        
        next_state, reward, done, terminated, _ = env.step(action)
        
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        total_reward += reward
        state = next_state
    
    rewards_all_episodes.append(total_reward)
    successes.append(total_reward == 20)

> 2000개 단위 에피소드별로 평균 보상 계산
    rewards_per_episodes = []
    episode_rewards = []
    count = 0

for reward in rewards_all_episodes:
    count += 1
    episode_rewards.append(reward)
    
    if count % 2000 == 0:
        avg_reward = sum(episode_rewards) / 2000
        rewards_per_episodes.append(avg_reward)
        episode_rewards = []

> 2000개 단위 에피소드별 평균 보상 출력

    print("********2000 Episodes Average Reward********\n")
    for idx, reward in enumerate(rewards_per_episodes):
    print((idx + 1) * 2000, ": ", reward)

> 업데이트된 Q-table 출력

    print("\n\n********Q_table********\n")
    print(q_table)

> 학습 종료 후 목적 달성 여부 확인을 위해 Episode 실행

    num_test_episodes = 10

    for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    print("Episode", episode + 1)
    time.sleep(1)
    while not done:
        env.render()  # Draw the current state
        time.sleep(0.3)
        action = np.argmax(q_table[state, :])  
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        
        env.render() 
        action = np.argmax(q_table[state, :])  
        next_state, reward, done, _, _ = env.step(action)
        
        if reward == 20:
            print("Success")
            time.sleep(3)
        else:
            print("Failed")
            time.sleep(3)
        break
        
        
    env.close()

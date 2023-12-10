# Taxi 구현하기

강화학습 문제 해결이 익숙하지 않아 꽤나 오랜 시간 과제를 진행했음에도 불구하고 제대로 된 결과값이 나오지 않아 애를 많이 먹었습니다. 할 수 있는 부분까지 최대한 작성해보았으며, 결과값이 이상하여 Jupyter Notebook과 Pycharm, 구글 Colab 등에서 여러 번 구현해보았습니다.

> 먼저 Open AI Gymnaisum을 설치 및 불러오기

    !pip install gymnasium
    import gymnasium as gym
    import numpy as np
    import random
    import time


> Gym 환경설정

    env = gym.make('Taxi-v3')
    state_space = env.observation_space.n
    action_space = env.action_space.n

> Q테이블 초기화

    q_table = np.zeros((state_space, action_space))

> Hyperparameter 설정 (a : 학습률, g : 할인율)

    a = 0.1 
    g = 0.999 

> Epsilon-Greedy 탐험 정책 설정

    epsilon = 0.1

> Boltzmann Exploration 탐험 정책 설정

    tau = 1.0 

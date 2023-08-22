import gym
import numpy as np
from matplotlib import pyplot as plt
from q_learning_agent import Agent


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.9999995, n_actions=4, n_states=16)

    scores = []
    win_percentage = []
    n_games = 500000

    for i in range(n_games):
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, state_)
            state = state_
            score += reward
        
        scores.append(score)

        if i % 100 == 0:
            win_percentage.append(np.mean(scores[-100:]))
            if i % 500:
                print(f'episode {i} \t win_pct {win_percentage[-1]} epsilon {agent.epsilon}')

    
    plt.plot(win_percentage)
    plt.show()
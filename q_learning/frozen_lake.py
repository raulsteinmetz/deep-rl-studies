import gym
import numpy as np
from matplotlib import pyplot as plt

def load_env():
    return gym.make('FrozenLake-v1', is_slippery=True)

def random_policy(env):
    n_games = 1000
    scores = []
    avg_scores = []

    # n_games loop
    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        # game loop
        while not done:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            score += reward
        scores.append(score)
        if i % 10 == 0:
            avg_scores.append(np.mean(scores[-10:]))

    print('Win percentage: ', np.mean(scores))

    plt.plot(avg_scores)
    plt.show()

def deterministic_policy(env):
    n_games = 1000
    scores = []
    avg_scores = []
    # 0 left, 1 down, 2 right, 3 up
    policy = {0: 2, 1: 2, 2: 1, 3: 0, 4:1, 6: 1, 8: 2, 9: 2, 10: 1, 13: 2, 14: 2}

    for i in range(n_games):
        state = env.reset()
        _state = state[0]
        done = False
        score = 0
        while not done:
            action = policy[_state]
            state, reward, done, _, _ = env.step(action)
            score += reward
        scores.append(score)
        if i % 10 == 0:
            avg_scores.append(np.mean(scores[-10:]))
    
    print('Win percentage: ', np.mean(scores))

    plt.plot(avg_scores)
    plt.show()




def main():
    env = load_env()
    deterministic_policy(env)



if __name__ == '__main__':
    main()
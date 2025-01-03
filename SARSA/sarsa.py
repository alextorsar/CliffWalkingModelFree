import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cliff_walking_wrapper import cliff_walking_wrapper
import random
import numpy as np

env = cliff_walking_wrapper()
q = [[0 for _ in range(env.action_space.n)] for _ in range(env.state_space.n)]

num_episodes = 500
alpha = 0.1
epsilon = 0.1
gamma = 0.9

def print_policy(policy):
    grid = [[' ' for _ in range(12)] for _ in range(4)]
    for state in range(env.state_space.n):
        row = state // 12
        col = state % 12
        if state == 36:
            grid[row][col] = 'G'
        elif state == 47:
            grid[row][col] = 'C'
        else:
            grid[row][col] = ['^', '>', 'v', '<'][policy[state]]
    for row in grid:
        print(row)

def epison_greedy_policy(q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(env.action_space.n))
    else:
        max_index = np.argwhere(q[state] == np.max(q[state])).flatten()
        return random.choice(max_index)

for episode in range(num_episodes):
    state = env.get_random_initial_state()
    action = epison_greedy_policy(q, state, epsilon)
    done = False
    while not done:
        next_state, reward, done = env.execute_action_from_state(state, action)
        next_action = epison_greedy_policy(q, next_state, epsilon)
        q[state][action] = (1-alpha)*q[state][action] + alpha*(reward + gamma*q[next_state][next_action])
        state = next_state
        action = next_action

policy = {}

for state in range(env.state_space.n):
    policy[state] = np.argmax(q[state])
    
print_policy(policy)

test_episodes = 5
env.execute_episodes_with_policy(num_episodes = test_episodes, policy=policy)      
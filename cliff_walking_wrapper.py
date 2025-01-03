import gymnasium as gym
import random
import math
import numpy as np

class cliff_walking_wrapper:

    def __init__(self):
        self.env = gym.make('CliffWalking-v0', is_slippery=True, render_mode='human').unwrapped
        self.action_space = self.env.action_space
        self.state_space = self.env.observation_space
        self.q = [[0 for _ in range(self.action_space.n)] for _ in range(self.state_space.n)]

    
    def get_random_initial_state(self):
        is_valid_initial_state = False
        while not is_valid_initial_state:
            state = random.choice(range(self.state_space.n))
            if state <= 36 or state == 47:
                is_valid_initial_state = True
        return state
    
    def execute_action_from_state(self, state, action):
        transitions = self.env.P[state][action]
        prob, next_state, reward, terminated  = random.choice(transitions)
        if next_state == 47:
            reward = -1
            terminated = True
        elif next_state == 36 and reward == -100:
            reward = -100
            terminated = True
        return next_state, reward, terminated
    
    def execute_episodes_with_policy(self,num_episodes, policy):
        for i in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = policy[state]
                state, reward, done, truncated, info = self.env.step(action)
"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DoubleDQN
import numpy as np


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

natural_DQN = DoubleDQN(
    n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
    e_greedy_increment=0.001, double_q=False)


double_DQN = DoubleDQN(
    n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
    e_greedy_increment=0.001, double_q=True)


def train(RL):
    observation = env.reset()
    while True:
        env.render()

        action = RL.predict_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        observation = observation_
    return RL.q

#natural_DQN.load_model("natural_DQN.h5")
#q_natural = train(natural_DQN)

double_DQN.load_model("double_DQN.h5")
q_double = train(double_DQN)


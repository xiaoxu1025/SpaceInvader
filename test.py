# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午4:22
@Auth ： leex-1312759081@qq.com
@File ：test.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""
import gym
from gym import wrappers
import numpy as np

env = gym.make('SpaceInvaders-v0')
env = wrappers.Monitor(env, 'videos', force=True)
num_actions = env.action_space.n


state = env.reset()

while True:
    action = np.random.randint(num_actions)
    observation, reward, done, info = env.step(action)
    if done:
        break

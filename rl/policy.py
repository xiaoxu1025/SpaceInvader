# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午4:25
@Auth ： leex-1312759081@qq.com
@File ：policy.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""
import numpy as np


class Policy(object):
    def __init__(self, action_nums):
        self.action_nums = action_nums

    def action(self, actions):
        raise NotImplementedError('not implemented')


class GreedyPolicy(Policy):
    def __init__(self, action_nums):
        super(GreedyPolicy, self).__init__(action_nums)

    def action(self, actions):
        return np.argmax(actions)


class RandomPolicy(Policy):
    def __init__(self, action_nums):
        super(RandomPolicy, self).__init__(action_nums)

    def action(self, actions):
        return np.random.randint(0, self.action_nums)


class GreedyEpsilonPolicy(Policy):
    def __init__(self, action_nums, epsilon=0.1):
        super(GreedyEpsilonPolicy, self).__init__(action_nums)
        self.epsilon = epsilon

    def action(self, actions):
        if np.random.random() >= self.epsilon:
            action = np.argmax(actions)
        else:
            action = np.random.randint(0, self.num_actions)
        return action


class LinearDecayGreedyEpsilonPolicy(Policy):
    def __init__(self, action_nums, start_value=0.999, end_value=0.1,
                 step_nums=1000000):
        super(LinearDecayGreedyEpsilonPolicy, self).__init__(action_nums)
        self.start_value = start_value
        self.end_value = end_value
        self.step_nums = step_nums

        self.steps = 0.0

    def action(self, actions):
        # epsilon = self.start_value + (self.steps / self.step_nums) * (self.end_value - self.start_value)
        epsilon = self.start_value + (self.steps / self.step_nums) * -self.start_value
        if self.steps < self.step_nums:
            self.steps += 1.0
        if np.random.random() >= epsilon:
            action = np.argmax(actions)
        else:
            action = np.random.randint(0, self.action_nums)
        return action

    def reset(self):
        self.steps = 0.0

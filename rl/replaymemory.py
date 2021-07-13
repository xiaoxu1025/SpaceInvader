# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午7:19
@Auth ： leex-1312759081@qq.com
@File ：replaymemory.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""

import numpy as np


# 经验回放 存放数据和进行采样
class ReplayMemory(object):
    def __init__(self, max_size=1000000, frame_nums=4):
        # buffer size
        self.max_size = max_size
        self.frame_nums = frame_nums
        # 抽取多少帧作为一个输入
        self.mem_size = (max_size + frame_nums - 1)
        # 存放数据data
        self.data = []

    def append(self, state, action, reward, is_end, next_state=None):
        memory = {
            'state': state,
            'action': action,
            'reward': reward,
            'is_end': is_end,
            'next_state': next_state
        }
        if self.size > 0:
            last_memory = self.data[-1]
        if self.size == 0 or last_memory['is_end']:
            # 每次游戏开始 存四张一样的帧当做一个state
            for i in range(self.frame_nums):
                self.data.append(memory)
        else:
            self.data.append(memory)
        while len(self.data) >= self.mem_size:
            first_memory = self.data[0]
            self.data.remove(first_memory)

    def sample(self, batch_size=32):
        size = self.size
        if size == 0:
            raise MemoryError('采样数据不足，请继续添加采样数据')
        if batch_size >= size + 3:
            batch_size = size + 3
        indexes = np.random.randint(0, size - 1, size=batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        is_ends = []
        for i in indexes:
            if self.data[i - 1]['is_end'] or self.data[i - 2]['is_end'] or self.data[i - 3]['is_end']:
                state = np.array([self.data[i]['state'],
                                  self.data[i]['state'],
                                  self.data[i]['state'],
                                  self.data[i]['state']])
                next_state = np.array([self.data[i]['state'],
                                       self.data[i]['state'],
                                       self.data[i]['state'],
                                       self.data[i + 1]['state']])
            elif self.data[i]['is_end']:
                state = np.array([self.data[i - 3]['state'],
                                  self.data[i - 2]['state'],
                                  self.data[i - 1]['state'],
                                  self.data[i]['state']])
                next_state = np.array([self.data[i - 2]['state'],
                                       self.data[i - 1]['state'],
                                       self.data[i]['state'],
                                       self.data[i]['state']])
            else:
                state = np.array([self.data[i - 3]['state'],
                                  self.data[i - 2]['state'],
                                  self.data[i - 1]['state'],
                                  self.data[i]['state']])
                next_state = np.array([self.data[i - 2]['state'],
                                       self.data[i - 1]['state'],
                                       self.data[i]['state'],
                                       self.data[i + 1]['state']])
            states.append(state)
            next_states.append(next_state)
            actions.append(self.data[i]['action'])
            rewards.append(self.data[i]['reward'])
            is_ends.append(self.data[i]['is_end'])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        is_ends = np.array(is_ends)

        states = np.transpose(states, (0, 2, 3, 1))
        next_states = np.transpose(next_states, (0, 2, 3, 1))

        return states, actions, rewards, next_states, is_ends

    @property
    def size(self):
        return len(self.data)

    def clear(self):
        if len(self.data) > 0:
            self.data.clear()

# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/13 上午4:51
@Auth ： leex-1312759081@qq.com
@File ：preprocessor.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""
import numpy as np
from PIL import Image


class Preprocessor(object):

    def get_state(self, state):
        return state

    def get_state_memory(self, state):
        return state

    def get_reward(self, reward):
        return reward

    def get_batch_data(self, data):
        return data

    def reset(self):
        raise NotImplementedError("Preprocessor reset not Implemented")


class MyPreprocessor(Preprocessor):

    def __init__(self, frame_nums, action_nums, new_size=(128, 128)):
        self.frame_nums = frame_nums
        self.action_nums = action_nums
        self.new_size = new_size
        self.frames = [None] * self.frame_nums

    def get_state(self, state):
        data = self.get_state_memory(state)
        data = data / 255.
        if self.frames[0] is None:
            for i in range(self.frame_nums):
                self.frames[i] = data
        else:
            # 更新最后一帧
            self.frames[0:self.frame_nums - 1] = self.frames[1:self.frame_nums]
            self.frames[self.frame_nums - 1] = data
        frames = np.expand_dims(np.asarray(self.frames), 0)
        # (batch_size, h, w, c)
        frames = np.transpose(frames, (0, 2, 3, 1))
        return frames

    def crop_image(self, image, new_height, new_width):
        width, height = image.size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        return image.crop((left, top, right, bottom))

    def get_state_memory(self, state):
        image = Image.fromarray(state, 'RGB')
        image = image.convert(mode='L')
        min_side = min(image.width, image.height)
        max_side = max(image.width, image.height)
        new_height = int(max_side * float(self.new_size[0]) / min_side)
        # (width, height)
        image = image.resize((self.new_size[0], new_height))
        image = self.crop_image(image, self.new_size[0], self.new_size[1])
        data = np.asarray(image)
        return data

    def get_reward(self, reward):
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -1.0
        else:
            return 0.0

    def get_batch_data(self, states, actions, next_states):
        states = states.astype('float32') / 255.
        next_states = next_states.astype('float32') / 255.
        new_actions = np.zeros((len(actions), self.action_nums), dtype='float32')
        new_actions[np.arange(len(actions), dtype='int'), actions] = 1.
        return states, new_actions, next_states

    def reset(self):
        self.frames = [None] * self.frame_nums

# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/13 上午9:14
@Auth ： leex-1312759081@qq.com
@File ：duelingdqn.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""

# 对网络结构进行改进 改成  𝑄 = V + A
from rldqn.naturedqn import NatureDQN
from rldqn.doubledqn import DoubleDQN
import numpy as np


class DuelingDQN(NatureDQN):
    def __init__(self, model, policies, q_values_func, memory, preprocessor, batch_size,
                 target_update_freq, gamma):
        super(DuelingDQN, self).__init__(model, policies, q_values_func, memory, preprocessor, batch_size,
                                         target_update_freq, gamma)

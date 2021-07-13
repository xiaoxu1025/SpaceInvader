# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/13 下午3:55
@Auth ： leex-1312759081@qq.com
@File ：duelinglayer.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""
from keras.layers import Layer
import keras.backend as K


class DuelingLayer(Layer):

    def __init__(self):
        super(DuelingLayer, self).__init__()

    def call(self, V, A):
        q_values = V + (A - K.mean(A, axis=1, keepdims=True))
        return q_values

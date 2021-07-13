# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午2:36
@Auth ： leex-1312759081@qq.com
@File ：model.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""

import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, dot, Flatten, Dropout
import keras.backend as K
from models.duelinglayer import DuelingLayer


def create_model(input_shape, action_nums):
    input_state = Input(shape=input_shape)
    input_action = Input(shape=(action_nums,))
    conv1 = Conv2D(16, kernel_size=(7, 7), strides=(4, 4), activation='relu')(input_state)
    conv2 = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv2)
    flattened = Flatten()(conv3)
    dense1 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(flattened)
    dense2 = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(dense1)
    q_values = Dense(action_nums, kernel_initializer='glorot_uniform', activation='tanh')(dense2)
    q_v = dot([q_values, input_action], axes=1)
    model = Model(inputs=[input_state, input_action], outputs=q_v)
    q_values_func = K.function([input_state], [q_values])
    return model, q_values_func


def create_duelingDQN_model(input_shape, action_nums):
    input_state = Input(shape=input_shape)
    input_action = Input(shape=(action_nums,))
    conv1 = Conv2D(16, kernel_size=(7, 7), strides=(4, 4), activation='relu')(input_state)
    conv2 = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv2)
    flattened = Flatten()(conv3)
    dense1 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(flattened)
    dense2 = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(dense1)

    V = Dense(1, kernel_initializer='glorot_uniform')(dense2)

    A = Dense(action_nums, kernel_initializer='glorot_uniform', activation='tanh')(dense2)

    q_values = DuelingLayer()(V, A)

    q_v = dot([q_values, input_action], axes=1)
    model = Model(inputs=[input_state, input_action], outputs=q_v)
    q_values_func = K.function([input_state], [q_values])
    return model, q_values_func

# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午4:56
@Auth ： leex-1312759081@qq.com
@File ：loss.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""

import tensorflow as tf


# 采用smooth_l1
# 1 说起来就是对于网络初期, 由于真实值和预测值之间误差较大, L2 损失的梯度也会过大可能导致梯度爆炸, 网络模型不稳定
# 2 而对于网络训练后期, 损失已经很小了, 在lr不变的情况下, 梯度绝对值1, 损失函数将在稳定值附近继续波动, 达不到更高的精度
def smooth_l1(y_true, y_pred, sigma=1.):
    diff = y_true - y_pred
    conditional = tf.less(tf.abs(diff), 1 / sigma ** 2)
    close = 0.5 * (sigma * diff) ** 2
    far = tf.abs(diff) - 0.5 / sigma ** 2
    loss = tf.where(conditional, close, far)
    return tf.reduce_mean(loss)

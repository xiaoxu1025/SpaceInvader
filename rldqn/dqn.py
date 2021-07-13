# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午9:50
@Auth ： leex-1312759081@qq.com
@File ：rldqn.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""


class DQN(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def select_action(self, state):
        raise NotImplementedError('select_action NotImplementedError')

    def update_model(self):
        raise NotImplementedError('update_model NotImplementedError')

# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/13 上午9:13
@Auth ： leex-1312759081@qq.com
@File ：doubledqn.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""

# double dqn implement
# double dqn  和 nature dqn 仅仅在update_model方法里有区别
# nature dqn 是从 𝑄′ 取 q 值最大的 action
# double dqn 1 而是先在当前Q网络中先找出最大Q值对应的动作 2 然后利用这个选择出来的动作在目标网络里面去计算目标Q值
from rldqn.naturedqn import NatureDQN
import numpy as np
from keras.utils import to_categorical


class DoubleDQN(NatureDQN):

    def __init__(self, model, policies, q_values_func, memory, preprocessor, batch_size,
                 target_update_freq, gamma):
        super(DoubleDQN, self).__init__(model, policies, q_values_func, memory, preprocessor, batch_size,
                                        target_update_freq, gamma)

    def update_model(self, update_nums):
        states, actions, rewards, next_states, is_ends = self.memory.sample(self.batch_size)
        states_normal, actions_normal, next_states_normal = self.preprocessor.get_batch_data(states, actions,
                                                                                             next_states)
        # 1 先在当前Q网络中先找出最大Q值对应的动作
        q_values = self.calc_q_values_func(next_states_normal)
        q_values_actions = np.argmax(q_values, axis=1)

        q_values_actions = to_categorical(q_values_actions, self.preprocessor.action_nums)
        # 2 然后利用这个选择出来的动作在目标网络里面去计算目标Q值
        target_q_values = self.target_model.predict_on_batch([next_states_normal, q_values_actions])

        new_rewards = rewards + self.gamma * target_q_values

        y = np.where(is_ends, rewards, new_rewards)
        y = np.expand_dims(y, axis=1)

        loss = self.model.train_on_batch([states, actions_normal], y)

        print('%s update model train loss: %s ' % (update_nums, loss))

        if self.step_nums != 0 and self.step_nums % self.target_update_freq == 0:
            print("update target model %s steps and %s times" % (
                self.step_nums, self.step_nums // self.target_update_freq))
            self.update_target_model_weights()

# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/12 下午10:00
@Auth ： leex-1312759081@qq.com
@File ：naturedqn.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""
# nature dqn 实现

from rldqn.dqn import DQN
from keras.models import clone_model
import keras.backend as K
import numpy as np
import sys


class NatureDQN(DQN):
    def __init__(self, model, policies, q_values_func, memory, preprocessor, batch_size,
                 target_update_freq, gamma):
        params = {
            'model': model,
            'policies': policies,
            'q_values_func': q_values_func,
            'memory': memory,
            'preprocessor': preprocessor
        }
        super(NatureDQN, self).__init__(**params)
        self.batch_size = batch_size
        self.step_nums = 0
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.target_model = clone_model(model)
        self.target_model.set_weights(model.get_weights())
        self.target_q_values_func = K.function([self.target_model.layers[0].input],
                                               [self.target_model.layers[-3].output])

        self.mode = 'init'

    def compile(self, optimizer, loss_func):
        self.model.compile(optimizer=optimizer, loss=loss_func)
        self.target_model.compile(optimizer=optimizer, loss=loss_func)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        self.update_target_model_weights()

    def update_target_model_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def calc_q_values_func(self, state):
        return self.q_values_func([state])[0]

    def calc_target_q_values_func(self, state):
        return self.target_q_values_func([state])[0]

    def select_action(self, state):
        state_normal = self.preprocessor.get_state(state)
        actions = self.calc_q_values_func(state_normal)
        return self.policies[self.mode].action(actions), state_normal

    def update_model(self, update_nums):
        states, actions, rewards, next_states, is_ends = self.memory.sample(self.batch_size)
        states_normal, actions_normal, next_states_normal = self.preprocessor.get_batch_data(states, actions,
                                                                                             next_states)
        q_values = self.calc_target_q_values_func(next_states_normal)
        max_q_values = np.max(q_values, axis=1)

        new_rewards = rewards + self.gamma * max_q_values

        y = np.where(is_ends, rewards, new_rewards)
        y = np.expand_dims(y, axis=1)
        loss = self.model.train_on_batch([states, actions_normal], y)

        print('%s update model train loss: %s ' % (update_nums, loss))

        if self.step_nums != 0 and self.step_nums % self.target_update_freq == 0:
            print("update target model %s steps and %s times" % (
            self.step_nums, self.step_nums // self.target_update_freq))
            self.update_target_model_weights()

    def fit(self, env, iterations_nums=5000000):
        print('init replay memory start')
        self.clear()

        update_nums = 0
        game_nums = 0
        while update_nums < iterations_nums:
            state = env.reset()
            game_nums += 1
            total_reward = 0
            times = 0
            while True:
                self.step_nums += 1
                times += 1
                action, _ = self.select_action(state)

                next_state, reward, done, info = env.step(action)

                reward = self.preprocessor.get_reward(reward)
                total_reward += reward

                state_normal = self.preprocessor.get_state_memory(state)
                self.memory.append(state_normal, action, reward, done)

                if self.step_nums > 10000:
                    if self.mode != 'train':
                        print("add data end! start training!")
                        self.mode = 'train'

                    if self.step_nums % 100 == 0:
                        self.update_model(update_nums)
                        update_nums += 1
                        if update_nums % 10000 == 0:
                            self.model.save_weights('weights/weights_%s.h5' % int(update_nums) // 10000,
                                                    save_format='h5')
                if done or times > 10000:
                    break
                state = next_state
            print(
                'one game over game_nums is (%s) update_nums is (%s) total reword is (%s) times is (%s) '
                'update_nums is (%s) step_nums is (%s)'
                % (game_nums, update_nums, total_reward, times, update_nums, self.step_nums))

    def evaluate(self, env, game_nums):
        self.mode = 'test'
        rewards = []
        average_game_length = 0.0
        for i in range(game_nums):
            state = env.reset()
            times = 0
            total_reward = 0.0
            while True:
                times += 1
                action, _ = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                average_game_length += 1
                if done or times > 10000:
                    break
                state = next_state

            rewards.append(total_reward)
        self.mode = 'train'
        return rewards, np.mean(rewards), np.std(rewards), average_game_length / game_nums

    def clear(self):
        sys.stdout.flush()
        self.memory.clear()
        self.preprocessor.reset()
        self.step_nums = 0
        self.mode = 'init'

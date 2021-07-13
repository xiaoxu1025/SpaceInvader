# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/13 上午9:14
@Auth ： leex-1312759081@qq.com
@File ：train.py
@IDE ：PyCharm
@Motto：梦中梦见梦中人

"""
import argparse
import os
import gym
from gym import wrappers
from models.model import create_model, create_duelingDQN_model
from rl.preprocessor import MyPreprocessor
from rl.replaymemory import ReplayMemory
from rl.policy import LinearDecayGreedyEpsilonPolicy, GreedyPolicy, RandomPolicy
from rldqn.naturedqn import NatureDQN
from rldqn.doubledqn import DoubleDQN
from rldqn.duelingdqn import DuelingDQN


def test(args):
    if not os.path.exists(args.weights_path):
        raise ValueError('model weights should not be null')
    env = gym.make(args.env)
    action_nums = env.action_space.n

    model, q_values_func = create_model((args.cropped_size, args.cropped_size, args.frame_nums), action_nums)

    rewards = []
    game_lens = []
    nums = 0
    while True:
        env = wrappers.Monitor(env, args.videos_path, force=True)

        preprocessor = MyPreprocessor(frame_nums=args.frame_nums, action_nums=action_nums,
                                      new_size=(args.cropped_size, args.cropped_size))
        memory = ReplayMemory(max_size=args.memsize, frame_nums=args.frame_nums)
        policies = {
            'init': RandomPolicy(action_nums),
            'train': LinearDecayGreedyEpsilonPolicy(action_nums),
            'test': GreedyPolicy(action_nums),
        }

        dqn = None
        if args.dqn == 1:
            dqn = NatureDQN(model, policies, q_values_func, memory, preprocessor, args.batch_size,
                            args.target_update_freq,
                            args.gamma)
        elif args.dqn == 2:
            dqn = DoubleDQN(model, policies, q_values_func, memory, preprocessor, args.batch_size,
                            args.target_update_freq,
                            args.gamma)
        elif args.dqn == 3:
            model, q_values_func = create_duelingDQN_model((args.cropped_size, args.cropped_size, args.frame_nums),
                                                           action_nums)
            dqn = DuelingDQN(model, policies, q_values_func, memory, preprocessor, args.batch_size,
                             args.target_update_freq,
                             args.gamma)
        else:
            raise NotImplementedError("dqn not implemented")

        dqn.load_weights(weights_path=args.weights_path)

        _, mean_reward, std_reward, game_length = dqn.evaluate(env, 1)
        nums += 1

        rewards.append(mean_reward)
        game_lens.append(game_length)

        print('mean reward = %s, std reward = %s, average_game_length = %d' % (mean_reward, std_reward, game_length))
        if nums > 100 or mean_reward > 300:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dqn params desc')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='gym env name')
    parser.add_argument('--dqn', default=2, type=int,
                        help='1: nature dqn, 2: double dqn, 3 dueling dqn 4 not implemented')
    parser.add_argument('--weights_path', default='weights/weights.h5', type=str, help='Directory to save data to')
    parser.add_argument('--output_path', default='weights', type=str, help='Directory to save data to')
    parser.add_argument('--videos_path', default='videos', type=str, help='Directory to save data to')
    parser.add_argument('--memsize', default=100000, type=int, help='replay memory max size')
    parser.add_argument('--frame_nums', default=4, type=int, help='the number of frames')
    parser.add_argument('--cropped_size', default=84, type=int, help='the size of the cropped image')
    parser.add_argument('--gamma', default=1., type=float, help='gamma')
    parser.add_argument('--target_update_freq', default=10000, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--iteration_nums', default=1000000, type=int)

    args = parser.parse_args()

    test(args)

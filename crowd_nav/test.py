import logging
import argparse
import configparser
import os
import sys
import time
import torch
import gym
import numpy as np

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.hesrl import HESRL
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *

state_dim = 13
action_dim = 2
hidden_dim = 128
hidden_size = 256
self_state_dim = 6
critic_lr = 5e-4
actor_lr = 5e-4
alpha_lr = 5e-3


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--model_dir', type=str, default='data/hesrl')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=True, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
    else:
        env_config_file = args.env_config

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    policy = HESRL(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, critic_lr, actor_lr, alpha_lr, device)
    policy.actor.load_state_dict(torch.load('../crowd_nav/data/hesrl/rl_model.pth', map_location='cpu'))
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    robot.print_info()
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    phase = 'test'
    episode = None
    total_episodes = 500
    info = ''
    for i in range(0, total_episodes):
        ob = env.reset(phase)
        done = False
        rewards = []
        while not done:
            joint_state = JointState(robot.get_full_state(), ob)
            state = policy.transform(joint_state)
            action = policy.predict(state, phase)
            action_use = ActionXY(np.float32(action[0][0]), np.float32(action[0][1]))
            ob, reward, done, info = env.step(action_use)
            rewards.append(reward)
            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)
        # env.render(mode="video")
        # env.render(mode="traj")
        discounted_reward = sum([pow(0.99, t) * reward for t, reward in enumerate(rewards)])
        cumulative_rewards.append(discounted_reward)
        print('episode:{}, reward:{:.2f}, time:{}, info:{}'.format(i, discounted_reward, env.global_time, info))
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.global_time)

    success_rate = success / total_episodes
    collision_rate = collision / total_episodes
    assert success + collision + timeout == total_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit
    time.sleep(2)
    extra_info = '' if episode is None else 'in episode {} '.format(episode)
    logging.info('{:<5} {}has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.2f}, average reward: {:.2f}'.
                 format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                        average(cumulative_rewards)))
    if phase in ['val', 'test']:
        num_step = sum(success_times + collision_times + timeout_times) / 0.25
        logging.info('Frequency of being in danger: %.3f and average min separate distance in danger: %.2f',
                     too_close / num_step, average(min_dist))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0


if __name__ == '__main__':
    main()

import logging
import numpy as np
import torch
import sys
import argparse
import configparser
import os
import shutil
import gym

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.hesrl import HESRL
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state import JointState1
from crowd_sim.envs.utils.info import *
from crowd_nav.trajectory_prediction.predict_trajectory import Predict
from crowd_nav.mpc_control.control import Control

state_dim = 13
action_dim = 2
hidden_dim = 128
hidden_size = 256
self_state_dim = 6
critic_lr = 5e-4
actor_lr = 5e-4
alpha_lr = 5e-3


def main():
    # 创建一个解析器
    parser = argparse.ArgumentParser('Parse configuration file')
    # default后是默认值
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    # 使用选项时，值被设置为true
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = 'y'
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            # 让args.env_config指向data/output下的同名文件
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
        if make_new_dir:
            os.makedirs(args.output_dir)
            shutil.copy(args.env_config, args.output_dir)
            shutil.copy(args.policy_config, args.output_dir)
            shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    torch.cuda.set_device(3)
    device = torch.device("cuda:3" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    policy = HESRL(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, critic_lr, actor_lr, alpha_lr, device)

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(policy)
    robot.print_info()
    timeout = 0
    phase = 'train'
    for episode in range(0, 50000):
        ob = env.reset(phase)
        done = False
        rewards = []
        success_rate = 0.0
        collision_rate = 0.0
        info = ''
        mpc_control = Control()
        while not done:
            joint_state = JointState(robot.get_full_state(), ob)
            state = policy.transform(joint_state)
            action = policy.predict(state, phase)
            action = action.cpu()
            action_use = ActionXY(np.float32(action[0][0]), np.float32(action[0][1]))

            human_x, human_y, human_vx, human_vy = policy.get_human_array(joint_state)
            robot_x, robot_y = policy.get_robot_state(joint_state)

            human_num = len(human_x)
            pred_list = []
            for i in range(human_num):
                predictor = Predict(human_x[i], human_y[i], human_vx[i], human_vy[i])
                predictor.predict()
                pred_list += predictor.pred_list

            mpc_control.curr_pose_cb(robot_x, robot_y)
            mpc_control.obs_cb(pred_list)
            action_mpc = mpc_control.mpc(action_use)
            action_use = ActionXY(np.float32(action_mpc[0]), np.float32(action_mpc[1]))

            action_vr1 = policy.predict_vr(state, phase)
            action_vr1 = action_vr1.cpu()
            action_use_vr1 = ActionXY(np.float32(action_vr1[0][0]), np.float32(action_vr1[0][1]))
            action_vr1 = torch.tensor([[action_use_vr1.vx, action_use_vr1.vy]])

            ob_vr1, full_ob_vr1, reward_vr1, reward1, done_vr1 = env.step(action_use_vr1, False)
            next_joint_state_vr1 = JointState(robot.get_next_full_state(action_use_vr1), ob_vr1)
            full_next_joint_state_vr1 = JointState1(robot.get_next_full_state(action_use_vr1), full_ob_vr1)
            next_state_vr1 = policy.transform(next_joint_state_vr1)
            policy.memory_vr.add(state, action_vr1, reward_vr1, next_state_vr1, done_vr1)
            policy.memory1.add(state, action_vr1, reward1, next_state_vr1, done_vr1)

            if not done_vr1:
                action_vr2 = policy.predict(next_state_vr1, phase)
                action_vr2 = action_vr2.cpu()
                action_use_vr2 = ActionXY(np.float32(action_vr2[0][0]), np.float32(action_vr2[0][1]))
                action_vr2 = torch.tensor([[action_use_vr2.vx, action_use_vr2.vy]])
                ob_vr2, robot_vr2, reward2, done_vr2 = env.step_vr(full_next_joint_state_vr1, action_use_vr2)
                next_joint_state_vr2 = JointState(robot_vr2, ob_vr2)
                next_state_vr2 = policy.transform(next_joint_state_vr2)
                policy.memory1.add(next_state_vr1, action_vr2, reward2, next_state_vr2, done_vr2)

            ob, reward, done, info = env.step(action_use)
            action = torch.tensor([[action_use.vx, action_use.vy]])
            next_joint_state = JointState(robot.get_full_state(), ob)
            next_state = policy.transform(next_joint_state)
            policy.memory2.add(state, action, reward, next_state, done)

            rewards.append(reward)
        total_reward = sum([pow(0.99, t) * reward for t, reward in enumerate(rewards)])
        if policy.memory1.counter + policy.memory2.counter > 1000:
            policy.optim()
        if policy.memory_vr.counter > 200:
            policy.optim_vr()
        if isinstance(info, ReachGoal):
            success_rate = 1.0
            avg_nav_time = env.global_time
        elif isinstance(info, Collision):
            collision_rate = 1.0
            avg_nav_time = env.global_time
        elif isinstance(info, Timeout):
            timeout += 1
            avg_nav_time = env.global_time
        else:
            raise ValueError('Invalid end signal from environment')

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info(
            '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                   total_reward))
        if episode % 1000 == 0 and episode != 0:
            torch.save(policy.actor.state_dict(), rl_weight_file)
    torch.save(policy.actor.state_dict(), rl_weight_file)


if __name__ == '__main__':
    main()

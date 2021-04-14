import ast
import argparse
import logging
import cv2

import os
import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper


def _enjoy():
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.set_mode(False)
    policy.load(filename='ddpg', directory='reinforcement/pytorch/models/')

    obs = env.reset()
    done = False

    while True:
        total_reward = 0
        while not done:
            # print(obs.max())
            # img = np.array(obs).transpose(1,2,0)
            # print(img.shape)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            action = policy.predict(np.array(obs))
            print(action)
            # Perform action
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
        done = False
        print(reward)
        obs = env.reset()

if __name__ == '__main__':
    _enjoy()

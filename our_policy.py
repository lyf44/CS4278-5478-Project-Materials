import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import torch
import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "./gym-duckietown/learning/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./gym-duckietown/learning/reinforcement/pytorch"))

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
import ss_detector

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map5')
parser.add_argument('--seed', type=int, default=4, help='random seed')
parser.add_argument('--load-dir', default='./model/')

args = parser.parse_args()

SEEDS = {
    "map1": [2, 3, 5, 9, 12],
    "map2": [1, 2, 3, 5, 7, 8, 13, 16],
    "map3": [1, 2, 4, 8, 9, 10, 15, 21],
    "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
    "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
}

# please remove this line for your own policy
actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, "duckietown_" + args.map_name + "_v3.pt"), map_location="cpu")
recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

rl_env = make_vec_envs(
    args.map_name,
    [args.seed],
    1,
    None,
    None,
    device="cpu",
    allow_early_resets=False)

rl_obs = rl_env.reset()

env = DuckietownEnv(
    map_name = args.map_name,
    domain_rand = False,
    draw_bbox = False,
    max_steps = args.max_steps,
    seed = args.seed
)
obs = env.reset()
env.render()

total_reward = 0
rl_total_reward = 0
step = 0
while step < args.max_steps:
    with torch.no_grad():
        value, rl_action, _, recurrent_hidden_states = actor_critic.act(rl_obs, recurrent_hidden_states, masks)

    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # print(obs.shape)
    # cv2.imshow("obs_np", obs)
    # cv2.waitKey(1)

    pos1 = ss_detector.detect_stopsign(obs)
    if pos1 is not None:
        pos2 = ss_detector.detect_stopsign_gt(env, pos1)

    # rl_action[0][0] = 0.15
    # print(rl_action)
    rl_obs, rl_reward, done, info = rl_env.step(rl_action)
    masks.fill_(0.0 if done[0] else 1.0)

    action = rl_action[0].numpy()
    action[0] = max(min(action[0], 0.8), 0)
    action[1] = max(min(action[1], 1), -1)
    # print(action)
    obs, reward, done, info = env.step(action)

    rl_total_reward += rl_reward[0][0].item()
    total_reward += reward

    # print('Steps = %s, Timestep Reward=%.3f, rl_total_reward=%.3f, Total Reward=%.3f' % (step, reward, rl_total_reward, total_reward))

    env.render()
    step += 1

    if done:
        break

print("Total Reward", total_reward)

# with torch.no_grad():
#     value, action, _, recurrent_hidden_states = actor_critic.act(obs, recurrent_hidden_states, masks)


# print(action)
# obs, reward, done, info = env.step(action)
# total_reward += reward

# env.render()
# step += 1

# action[0][0] = 0.15
# print(action)
# obs, reward, done, info = env.step(action)
# total_reward += reward

# action[0][0] = 0.3
# print(action)
# obs, reward, done, info = env.step(action)
# total_reward += reward

# action[0][0] = 0.5
# print(action)
# obs, reward, done, info = env.step(action)
# total_reward += reward

# action[0][0] = 0.65
# print(action)
# obs, reward, done, info = env.step(action)
# total_reward += reward

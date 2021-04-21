import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "./gym-duckietown/learning/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./gym-duckietown/learning/reinforcement/pytorch"))

from .a2c_ppo_acktr.envs import make_vec_envs

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map5')
parser.add_argument('--seed', type=int, default=11, help='random seed')
parser.add_argument('--load-dir', default='./model/')

args = parser.parse_args()

SEEDS = {
    "map1": [2, 3, 5, 9, 12],
    "map2": [1, 2, 3, 5, 7, 8, 13, 16],
    "map3": [1, 2, 4, 8, 9, 10, 15, 21],
    "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
    "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
}

env = make_vec_envs(
    args.map_name,
    [2],
    1,
    None,
    None,
    device="cpu",
    allow_early_resets=False)

obs = env.reset()
env.render()

total_reward = 0

# please remove this line for your own policy
actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, "duckietown_" + args.map_name + "_v3.pt"), map_location="cpu")
recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

step = 0
while step < args.max_steps:
    obs = torch.tensor(obs)
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(obs, recurrent_hidden_states, masks)

    obs, reward, done, info = env.step(action)
    total_reward += reward

    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

print("Total Reward", total_reward)

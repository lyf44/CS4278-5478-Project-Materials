import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__)))

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from .a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from .a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

SEEDS = {
    "map1": [2, 3, 5, 9, 12],
    "map2": [1, 2, 3, 5, 7, 8, 13, 16],
    "map3": [1, 2, 4, 8, 9, 10, 15, 21],
    "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
    "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
}

HARD_SEEDS = {
    "map1": [],
    "map2": [2],
    "map3": [8],
    "map4": [2, 4, 7],
    "map5": [2, 8, 9, 16]
}

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='duckietown',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./reinforcement/pytorch/trained_models/ppo/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument('--map-name', default="map1", help='map name')
args = parser.parse_args()

args.det = not args.non_det
# device = torch.device("cuda:0" if args.cuda else "cpu")

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, args.env_name + "_" + args.map_name + ".pt"), map_location="cpu")

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

failed_seeds = []
for i in range(len(SEEDS[args.map_name])):
    env = make_vec_envs(
        args.map_name,
        [SEEDS[args.map_name][i]],
        1,
        None,
        None,
        device="cpu",
        allow_early_resets=False)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    # Get a render function
    # render_func = get_render_func(env)

    obs = env.reset()
    env.render()
    # if render_func is not None:
    #     render_func('human')

    total_reward = 0
    step = 0
    while step <= 500: # 1500
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        print(action)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward[0][0].item()
        masks.fill_(0.0 if done else 1.0)

        if done[0]:
            print("done!!")
            failed_seeds.append(SEEDS[args.map_name][i])
            break

        step += 1

    print(total_reward)
    env.close()

print(failed_seeds)
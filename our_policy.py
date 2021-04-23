import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import torch
import os
import sys
import cv2
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "./gym-duckietown/learning/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./gym-duckietown/learning/reinforcement/pytorch"))

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
import ss_detector
import duckietown_model
from particle_filter import ParticleFilter

# CONSTANTS
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
    "map4": [2, 4, 7, 18],
    "map5": [2, 8, 9, 16]
}
SS_TRACK_THRES = 2.0
NUM_PARTICLES = 100
NUM_RANDOM_PARTICLES = 10
CLAMP_SPEED_DIST = 0.3 # allow some error

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map5')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--load-dir', default='./model/')
parser.add_argument('--no-render', action="store_true", default=False)
args = parser.parse_args()

# please remove this line for your own policy
try:
    actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, "duckietown_" + args.map_name + "_s" + str(args.seed) + ".pt"), map_location="cpu")
    print("load seed-specific model")
except Exception as e:
    print(e)
    actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, "duckietown_" + args.map_name + ".pt"), map_location="cpu")
    print("load default model")

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

if not args.no_render:
    env.render()

pf = None
total_reward = 0
rl_total_reward = 0
step = 0
gt_pos_ss = None
tmp_dist_ss_pf = 0
actions = []
while step < args.max_steps:
    with torch.no_grad():
        value, rl_action, _, recurrent_hidden_states = actor_critic.act(rl_obs, recurrent_hidden_states, masks, deterministic=True)

    # rl_action[0][0] = max(min(rl_action[0][0], 0.7), 0)
    # rl_action[0][1] = max(min(rl_action[0][1], 0.875), -0.875)

    rl_action[0][0] = max(min(rl_action[0][0], 0.8), 0)
    rl_action[0][1] = max(min(rl_action[0][1], 1), -1)

    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # print(obs.shape)
    # cv2.imshow("obs_np", obs)
    # cv2.waitKey(1)

    pos_ss_r = ss_detector.detect_stopsign(obs)

    if pos_ss_r is not None:
        pos_ss_r = duckietown_model.correct_ss_obs(pos_ss_r)
        dist_to_ss = math.sqrt(pos_ss_r[0] ** 2 + pos_ss_r[1] ** 2)
        if dist_to_ss < SS_TRACK_THRES:
            if pf is None:
                print("-----------Stop sign detected, start tracking!!!")
                initial_particles_x = np.random.normal(pos_ss_r[0], duckietown_model.STD_X, NUM_PARTICLES).reshape(-1, 1)
                initial_particles_y = np.random.normal(pos_ss_r[1], duckietown_model.STD_Y, NUM_PARTICLES).reshape(-1, 1)
                initial_particles = np.concatenate((initial_particles_x, initial_particles_y), axis=1)
                print(initial_particles.shape)
                pf = ParticleFilter(initial_particles, duckietown_model.transit_state, duckietown_model.measurement_prob)
            else:
                print("----------Stop sign detected, pf update!!!")
                ss_pos = pf.get_estimate()
                print("predict: {}, meas: {}".format(ss_pos, pos_ss_r), end=" ")
                pf.update(pos_ss_r)
                tmp_dist_ss_pf = math.sqrt(ss_pos[0] ** 2 + ss_pos[1] ** 2)

    if pf is not None:
        ss_pos = pf.get_estimate()
        dist_to_ss = math.sqrt(ss_pos[0] ** 2 + ss_pos[1] ** 2)

        gt_pos_ss = ss_detector.detect_stopsign_gt(env, ss_pos)
        if gt_pos_ss is not None:
            gt_dist_to_ss = math.sqrt(gt_pos_ss[0] ** 2 + gt_pos_ss[1] ** 2)

            print("pf estimate: {}, gt: {}".format(ss_pos, gt_pos_ss))
            print("pf estimate: {}, gt: {}".format(dist_to_ss, gt_dist_to_ss))

        # to prevent particle collapse
        random_particles_x = np.random.normal(ss_pos[0], 0.5, NUM_RANDOM_PARTICLES).reshape(-1, 1)
        random_particles_y = np.random.normal(ss_pos[1], 0.5, NUM_RANDOM_PARTICLES).reshape(-1, 1)
        random_particles = np.concatenate((random_particles_x, random_particles_x), axis=1)
        pf.add_random_samples(random_particles)

        if dist_to_ss <= CLAMP_SPEED_DIST + tmp_dist_ss_pf * 0.2:
            print("----------Close to stop sign, clamp speed to 0.15m/s!!!")
            rl_action[0][0] = max(min(rl_action[0][0], 0.1), 0)
            rl_action[0][1] = max(min(rl_action[0][1], 0.125), -0.125)

        if dist_to_ss >= SS_TRACK_THRES:
            print("-----------Too far from stop sign, stop tracking!!!")
            pf = None

    rl_obs, rl_reward, done, info = rl_env.step(rl_action)
    masks.fill_(0.0 if done[0] else 1.0)

    action = rl_action[0].numpy()
    # action[0] = max(min(action[0], 0.8), 0)
    # action[1] = max(min(action[1], 1), -1)
    print(action)
    actions.append(action)
    obs, reward, done, info = env.step(action)

    rl_total_reward += rl_reward[0][0].item()
    total_reward += reward

    if reward <= -100:
        print("Didnt stop!!")
        break
    # print('Steps = %s, Timestep Reward=%.3f, rl_total_reward=%.3f, Total Reward=%.3f' % (step, reward, rl_total_reward, total_reward))

    if pf is not None:
        pf.predict(action)

    if not args.no_render:
        env.render()
    step += 1

    if done:
        print("Done!!")
        break

print("step_cnt", step)
print("Total Reward", total_reward)

# dump the controls using numpy
if step >= 1500:
    actions = np.array(actions)
    np.savetxt('./control_files/{}_seed{}.txt'.format(args.map_name, args.seed), actions, delimiter=',')
    print("control file saved!!")

import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import torch
import os
import sys
import cv2
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

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
    "map3": [1, 2, 4, 9, 10, 15, 21], # 8
    "map4": [1, 3, 5, 9, 10, 16, 18], # [2, 4, 7]
    "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
}

# # please remove this line for your own policy
# actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, "duckietown_" + args.map_name + ".pt"), map_location="cpu")
# recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
# masks = torch.zeros(1, 1)

# error = []
# gt = []
# est = []
# for i in range(len(SEEDS[args.map_name])):
# # for i in range(1):
#     print("Collecting error information on {} seed {}".format(args.map_name, SEEDS[args.map_name][i]))
#     rl_env = make_vec_envs(
#         args.map_name,
#         [SEEDS[args.map_name][i]],
#         1,
#         None,
#         None,
#         device="cpu",
#         allow_early_resets=False)

#     rl_obs = rl_env.reset()

#     env = DuckietownEnv(
#         map_name = args.map_name,
#         domain_rand = False,
#         draw_bbox = False,
#         max_steps = args.max_steps,
#         seed = SEEDS[args.map_name][i]
#         # seed = 23
#     )
#     obs = env.reset()

#     # env.render()

#     rl_total_reward = 0
#     total_reward = 0
#     step = 0
#     while step <= 500:
#         # input("press")
#         with torch.no_grad():
#             value, rl_action, _, recurrent_hidden_states = actor_critic.act(rl_obs, recurrent_hidden_states, masks)

#         obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
#         # print(obs.shape)
#         # cv2.imshow("obs_np", obs)
#         # cv2.waitKey(1)

#         pos1 = ss_detector.detect_stopsign(obs)
#         if pos1 is not None:
#             new_pos1 = pos1.copy()
#             new_pos1[0] = pos1[0] * 3.84652 + 0.00172
#             new_pos1[1] = pos1[1] * 1.21 + 0.0278
#             pos2 = ss_detector.detect_stopsign_gt(env, pos1)
#             if pos2 is not None and math.sqrt(pos2[0]**2 + pos2[1]**2) < 1.2 and math.sqrt(new_pos1[0]**2 + new_pos1[1]**2) < 1.2:
#                 cur_err = (np.array(pos2) - np.array(new_pos1))[:2]
#                 print(pos1, new_pos1, pos2, cur_err)
#                 error.append(cur_err)
#                 gt.append(np.array(pos2)[:2])
#                 est.append(np.array(pos1)[:2])

#         # rl_action[0][0] = 0.15
#         # print(rl_action)
#         rl_obs, rl_reward, done, info = rl_env.step(rl_action)
#         masks.fill_(0.0 if done[0] else 1.0)

#         action = rl_action[0].numpy()
#         action[0] = max(min(action[0], 0.8), 0)
#         action[1] = max(min(action[1], 1), -1)
#         # print(action)
#         obs, reward, done, info = env.step(action)

#         rl_total_reward += rl_reward[0][0].item()
#         total_reward += reward

#         # print('Steps = %s, Timestep Reward=%.3f, rl_total_reward=%.3f, Total Reward=%.3f' % (step, reward, rl_total_reward, total_reward))

#         # env.render()
#         step += 1

#         if done:
#             break

#     env.close()

# error = np.array(error).reshape(-1, 2)
# gt = np.array(gt).reshape(-1, 2)
# est = np.array(est).reshape(-1, 2)
# file_name = "error_{}.npz".format(args.map_name)
# with open(file_name, 'wb') as f:
#     np.savez(f, error=error, gt=gt, est=est)

file_name = "error_{}.npz".format(args.map_name)
data = np.load(file_name)
error = data["error"]
gt = data["gt"]
est = data["est"]

# mu, std = norm.fit(error[:, 0])

# xmin, xmax = plt.xlim()
# x = np.linspace(-1, 1, 100)
# p = norm.pdf(x, mu, std)
# plt.hist(error[:, 0], density=True, histtype='stepfilled', alpha=0.2)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)
# print(title)
# plt.show()

# mu, std = norm.fit(error[:, 1])

# xmin, xmax = plt.xlim()
# x = np.linspace(-1, 1, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# plt.hist(error[:, 1], density=True, histtype='stepfilled', alpha=0.2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)
# print(title)
# plt.show()

y = gt[:, 0]
x = est[:, 0]
res = scipy.stats.linregress(x, y)
plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.xlabel("Detected x")
plt.ylabel("Ground truth x")
title = "Fit results: slope = %.2f, intercept = %.2f, stderr = %.2f" % (res.slope, res.intercept, res.stderr)
plt.title(title)
plt.show()
print("stderr ", res.stderr)
print(res.slope, res.intercept)

y = gt[:, 1]
x = est[:, 1]
res = scipy.stats.linregress(x, y)
plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.xlabel("Detected y")
plt.ylabel("Ground truth y")
title = "Fit results: slope = %.2f, intercept = %.2f, stderr = %.2f" % (res.slope, res.intercept, res.stderr)
plt.title(title)
plt.show()
print("stderr ", res.stderr)
print(res.slope, res.intercept)
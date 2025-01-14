import copy
import glob
# import os
import time
from collections import deque
import ast
import argparse
import logging
import os
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__)))

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/ppo')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
SEQUENCE_LENGTH = 5

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
    "map4": [2], # [2, 4, 7],
    "map5": [2, 8, 9, 16]
}

def evaluate(actor_critic, args, num_processes, eval_log_dir, device):
    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.obs_rms = obs_rms
    eval_envs = make_vec_envs(args.map_name, HARD_SEEDS[args.map_name], len(HARD_SEEDS[args.map_name]), args.gamma, eval_log_dir, device, True)
    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    steps = 0
    while len(eval_episode_rewards) < 10 and steps <= 1500:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode_reward' in info.keys():
                eval_episode_rewards.append(info['episode_reward'])

        steps += 1

    # eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}, min reward {:.5f}".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.min(eval_episode_rewards)))

    return np.min(eval_episode_rewards)

def main(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Launch the envs with our helper function
    # envs = launch_env()

    # Wrappers
    # envs = ResizeWrapper(envs)
    # envs = NormalizeWrapper(envs)
    # envs = ImgWrapper(envs) # to make the images from 160x120x3 into 3x160x120
    # envs = ActionWrapper(envs)
    # envs = DtRewardWrapper(envs)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Initialized environments, device = {}".format(device))
    # envs = make_vec_envs(args.map_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
    envs = make_vec_envs(args.map_name, SEEDS[args.map_name], args.num_processes, args.gamma, args.log_dir, device, False,
        hard_seeds = HARD_SEEDS[args.map_name], use_hard_seed=True)

    if args.load_model:
        print("loading existing models!!")
        actor_critic, obs_rms = torch.load(os.path.join(args.save_dir, "ppo/v2/", args.env_name + "_" + args.map_name + ".pt"))
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    print(envs.observation_space.shape)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    # actor_critic.multi_gpu()
    actor_critic.to(device)

    obs = envs.reset()
    # print(obs.shape)
    rollouts.obs[0].copy_(torch.tensor(obs))
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    seeds_rewards = [0] * len(SEEDS[args.map_name])
    best_reward = -np.inf
    best_mean_reward = -np.inf
    to_eval = False
    # eval_reward = evaluate(actor_critic, eval_env, args, len(SEEDS[args.map_name]), eval_log_dir, device)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            # env_action = action[0].cpu().numpy()
            # print(env_action)
            # print(action)
            obs, reward, done, infos = envs.step(action)

            # for info in infos:
            #     print(info)
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])
            for info in infos:
                if 'episode_reward' in info.keys():
                    episode_rewards.append(info['episode_reward'])
            for i, done_ in enumerate(done):
                if done_:
                    idx = SEEDS[args.map_name].index(infos[i]["seed_val"])
                    seeds_rewards[idx] = infos[i]['episode_reward']

            # If done then clean the history of observations.store_true
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        # print(episode_rewards)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        # min_reward = np.min(episode_rewards) if len(episode_rewards) > 0 else -np.inf
        # if min_reward > best_reward:
        #     best_reward = min_reward
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
        #     ], os.path.join(save_path, args.env_name + "_best.pt"))
        #     print("Best Model saved!!!, min_reward = {}".format(min_reward))

        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))
            print("Model saved!!!")

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            # writer.add_scalar('reward', np.mean(episode_rewards), total_num_steps)
            mean_reward = np.mean(episode_rewards)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                to_eval = True

        if to_eval or ((args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0)):
            # obs_rms = utils.get_vec_normalize(envs).obs_rms
            # evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
            eval_reward = evaluate(actor_critic, args, len(SEEDS[args.map_name]), eval_log_dir, device)
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + "_best.pt"))
                print("Best Model saved!!!, min_reward = {}".format(eval_reward))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DDPG Args
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--gail', action='store_true', default=False, help='do imitation learning with gail')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int,  default=32, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=20, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-env-steps', type=int, default=3e6, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='duckietown', help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='./reinforcement/pytorch/log/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./reinforcement/pytorch/trained_models/', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=True, help='use a linear schedule on the learning rate')
    parser.add_argument('--load-model', default=False, help='load a model')
    parser.add_argument('--map-name', default="map1", help='map name')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    main(parser.parse_args())

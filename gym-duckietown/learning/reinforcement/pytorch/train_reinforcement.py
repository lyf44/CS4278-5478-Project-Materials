import ast
import argparse
import logging

import os
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/ddpg')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
SEQUENCE_LENGTH = 5

def _train(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

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

    # Set seeds
    seed(args.seed)
    obs_dim = env.observation_space.shape
    state_dim = np.array([obs_dim[0], obs_dim[1], obs_dim[2]*SEQUENCE_LENGTH])
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized DDPG")

    # Evaluate untrained policy
    # evaluations= [evaluate_policy(env, policy)]
    evaluations = []

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    misc = {}
    action = []

    print("Starting training")
    while total_timesteps < args.max_timesteps:

        print("timestep: {} | action: {} | reward: {}".format(total_timesteps, action, reward))

        if done:
            if total_timesteps != 0:
                print(misc['Simulator']['msg'])
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
                writer.add_scalar('reward', episode_reward/10000, episode_num)
                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

                    if args.save_models:
                        policy.save(filename='ddpg', directory=args.model_dir)
                    np.savez("./results/rewards.npz",evaluations)


            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
            action[0] = (action[0] + 1) / 2
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    args.expl_noise,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, misc = env.step(action)
        if action[0] < 0.001:   #Penalise slow actions: helps the bot to figure out that going straight > turning in circles
            reward = -40

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward
        # from PIL import Image
        # obs = np.array(Image.fromarray(obs).resize((np.shape(obs)[0], int(np.shape(obs)[1]/2), int(np.shape(obs)[2]/2))))
        # new_obs = np.array(Image.fromarray(new_obs).resize((int(np.shape(obs)[0]/2), int(np.shape(obs)[1]/2))))
        if episode_timesteps == 0:
            seq = [obs for _ in range(SEQUENCE_LENGTH)]
            obs_seq = np.concatenate(seq)
        else:
            obs_seq = np.concatenate((obs_seq[3:], obs))
        # Store data in replay buffer
        new_obs_seq = np.concatenate((obs_seq[3:], new_obs))
        # print(np.shape(new_obs_seq))
        assert obs_seq.shape == (15, 80, 60)
        assert new_obs_seq.shape == (15, 80, 60)
        replay_buffer.add(obs_seq, new_obs_seq, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    print("Training done, about to save..")
    policy.save(filename='ddpg', directory=args.model_dir)
    print("Finished saving..should return now!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--model-dir', type=str, default='reinforcement/pytorch/models/')

    _train(parser.parse_args())

import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv

map_idx = 1

def launch_env(map, seed=123):
    print(map)
    env = None
    if map is not None:
        # Launch the environment
        # from gym_duckietown.simulator import Simulator
        # env = Simulator(
        #     seed=123, # random seed
        #     map_name="loop_empty",
        #     max_steps=500001, # we don't want the gym to reset itself
        #     domain_rand=0,
        #     camera_width=640,
        #     camera_height=480,
        #     accept_start_angle_deg=4, # start close to straight
        #     full_transparency=True,
        #     distortion=True,
        # )

        env = DuckietownEnv(
            seed=seed, # random seed
            map_name=map,
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )

        # print("Env uses map {}".format(map_idx))
        # env = DuckietownEnv(
        #     seed=seed, # random seed
        #     map_name="map{}".format(map_idx),
        #     max_steps=500001, # we don't want the gym to reset itself
        #     domain_rand=0,
        #     camera_width=640,
        #     camera_height=480,
        #     accept_start_angle_deg=4, # start close to straight
        #     full_transparency=True,
        #     distortion=True,
        # )
        # map_idx += 1
        # if map_idx > 5:
        #     map_idx = 1
    else:
        env = gym.make(id)

    return env


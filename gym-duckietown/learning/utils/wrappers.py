import gym
from gym import spaces
import numpy as np
import cv2

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(64, 64, 3)): # 60, 80
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):

        from PIL import Image
        # print(observation.dtype)
        obs = np.array(Image.fromarray(observation).resize(self.shape[0:2]))
        # print(np.max(obs))
        # print(obs.shape)
        return obs


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        # print(observation.shape)
        obs = observation[10:, :, :]

        img_hsv = cv2.cvtColor(obs.astype(np.float32), cv2.COLOR_RGB2HSV)
        low_val = (0,0,0.5)
        high_val = (80,1,1)
        mask = cv2.inRange(img_hsv, low_val,high_val)
        obs = cv2.bitwise_and(obs, obs, mask=mask) * 255
        obs = obs.astype('uint8')

        # img_hsv = cv2.cvtColor(obs.astype(np.float32), cv2.COLOR_RGB2HSV)
        # print(np.max(obs))
        # obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", obs_bgr)
        # cv2.waitKey(1)

        # from PIL import Image
        # print(observation.shape)
        # obs = np.array(Image.fromarray(obs).resize(self.shape[0:2]))
        # print(np.max(obs))
        obs = cv2.resize(obs, (self.shape[:2]))
        obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", obs_bgr)
        # cv2.waitKey(1)
        obs = obs.transpose(2, 0, 1) # 3x64x64
        # obs = observation.transpose(2,0,1)
        # print(obs.shape)

        return obs


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10 # -40
        elif reward > 0:
            reward += 10
            # reward *= 5
        else:
            reward += 4
            # pass

        return reward


# Deprecated
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [max(0, min(action[0], 0.8)), max(-1.0, min(action[1], 1.0))]
        return action_


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self,
                 env,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102
                 ):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist

    def action(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels
#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from dt_apriltags import Detector
import cv2
from learning.imitation.iil_dagger.teacher import PurePursuitPolicy 

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=True, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

expert = PurePursuitPolicy(env, ref_velocity=0.7)
obs = env.reset()
env.render()
# obs = env.reset()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def _update_pos(pos, angle, wheel_dist, wheelVels, deltaTime):
    """
    pose prediction model from env setup
    """

    Vl, Vr = wheelVels
    l = wheel_dist

    # If the wheel velocities are the same, then there is no rotation
    if Vl == Vr:
        pos = pos + deltaTime * Vl * get_dir_vec(angle)
        return pos, angle

    # Compute the angular rotation velocity about the ICC (center of curvature)
    w = (Vr - Vl) / l

    # Compute the distance to the center of curvature
    r = (l * (Vl + Vr)) / (2 * (Vl - Vr))

    # Compute the rotation angle for this time step
    rotAngle = w * deltaTime

    # Rotate the robot's position around the center of rotation
    r_vec = get_right_vec(angle)
    px, py, pz = pos
    cx = px + r * r_vec[0]
    cz = pz + r * r_vec[2]
    npx, npz = rotate_point(px, pz, cx, cz, rotAngle)
    pos = np.array([npx, py, npz])

    # Update the robot's direction angle
    angle += rotAngle
    return pos, angle

def update(dt, obs):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    # action = np.array([0.0, 0.0])

    # if key_handler[key.UP]:
    #     action = np.array([0.44, 0.0])
    # if key_handler[key.DOWN]:
    #     action = np.array([-0.44, 0])
    # if key_handler[key.LEFT]:
    #     action = np.array([1, +1])
    # if key_handler[key.RIGHT]:
    #     action = np.array([0.8, -1])
    # if key_handler[key.SPACE]:
    #     action = np.array([0, 0])

    # # Speed boost
    # if key_handler[key.LSHIFT]:
    #     action *= 1.5
    # obs = env.reset()
    # env.render()
    action = expert.predict(observation=obs)
    obs, reward, done, info = env.step(action)
    # print(obs)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(obs, estimate_tag_pose=True, camera_params=(305.5718893575089,308.8338858195428,303.0797142544728,231.8845403702499), tag_size=0.08)
    if len(tags) != 0:
        d = np.linalg.norm(np.array(tags[0].pose_t))
        # print(d)
        dist_to_stop = 1000.0
        pos = env.cur_pos
        for obj in env.objects:
            if obj.kind == "sign_stop":
                dist_to_stop = min(dist_to_stop, ((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)
        print("estimated distance: {:.3f}, real distance: {:.3f}".format(d, dist_to_stop))
    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    # if key_handler[key.RETURN]:
    #     from PIL import Image
    #     im = Image.fromarray(obs)

    #     im.save('screen.png')

    if done:
        print('done!')
        obs = env.reset()
        env.render()
        

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate, obs)

# Enter main event loop
pyglet.app.run()

env.close()

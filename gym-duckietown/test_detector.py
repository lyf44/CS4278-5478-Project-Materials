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
import math

WHEEL_DIST = 0.102
DEFAULT_ROBOT_SPEED = 1.20
GAIN = 1.0
TRIM = 0.0
RADIUS = 0.0318
K = 27.0
LIMIT = 1.0

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

expert = PurePursuitPolicy(env, ref_velocity=0.5)
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


def get_dir_vec(cur_angle):
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    return np.array([x, 0, z])

def get_right_vec(cur_angle):
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(cur_angle)
    z = math.cos(cur_angle)
    return np.array([x, 0, z])

def rotate_point(px, py, cx, cy, theta):
    """
    Rotate a 2D point around a center
    """

    dx = px - cx
    dy = py - cy

    new_dx = dx * math.cos(theta) + dy * math.sin(theta)
    new_dy = dy * math.cos(theta) - dx * math.sin(theta)

    return cx + new_dx, cy + new_dy

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

def action_to_command(action):
    vel, angle = action

    # Distance between the wheels
    baseline = WHEEL_DIST

    # assuming same motor constants k for both motors
    k_r = K
    k_l = K

    # adjusting k by gain and trim
    k_r_inv = (GAIN + TRIM) / k_r
    k_l_inv = (GAIN - TRIM) / k_l

    omega_r = (vel + 0.5 * angle * baseline) / RADIUS
    omega_l = (vel - 0.5 * angle * baseline) / RADIUS

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, LIMIT), -LIMIT)
    u_l_limited = max(min(u_l, LIMIT), - LIMIT)

    vels = np.array([u_l_limited, u_r_limited])
    return vels

def update(dt, obs):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """


    action = expert.predict(observation=obs)
    # print(action)
    wheelvels = np.array(action_to_command(action) * DEFAULT_ROBOT_SPEED * 1)
    # pose = _update_pos(pos, angle, WHEEL_DIST, wheelvels, dt)
    if expert.estimated_angle is not None and expert.estimated_pose is not None:
        
        expert.estimated_pose, expert.estimated_angle = _update_pos(expert.estimated_pose, expert.estimated_angle, WHEEL_DIST, wheelvels, dt)
        if expert.estimated_angle < 0:
            expert.estimated_angle =+ 2*math.pi
        elif expert.estimated_angle > 2 * math.pi:
            expert.estimated_angle -= 2 * math.pi
        d = math.sqrt(expert.estimated_pose[0]**2 + expert.estimated_pose[2]**2)
        pos = env.cur_pos
        dist_to_stop = []
        for obj in env.objects:
            if obj.kind == "sign_stop":
                dist_to_stop.append(((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)
        # print("predicted distance: {:.3f}, real distance: {}".format(d, dist_to_stop))
        abs_dist = [abs(dist_to_stop[0] - d), abs(dist_to_stop[1] - d), abs(dist_to_stop[2] - d)]
        # print(min(dist_to_stop))
        # print("estimated distance: {}, real_distance: {}".format(d, min(dist_to_stop)))
        if d <= 0.6:
        # print("pose: {}, angle: {}".format(expert.estimated_pose, expert.estimated_angle))
        # print("estimated distance: {}, real_distance: {}".format(d, min(dist_to_stop)))
            print(dist_to_stop)
            # print("stop region estimation error: {}".format(min(abs_dist)))
    obs, reward, done, info = env.step(action)
    # print(obs)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(obs, estimate_tag_pose=True, camera_params=(305.5718893575089,308.8338858195428,303.0797142544728,231.8845403702499), tag_size=0.075)
    if len(tags) != 0:
        tmp_dist = 1000.0
        idx = None
        # print(len(tags))
        for i in range(len(tags)):
            t = np.array(tags[i].pose_t)
            if math.sqrt(t[0] ** 2 + t[2] ** 2) < tmp_dist:
                tmp_dist = math.sqrt(t[0] ** 2 + t[2] ** 2) 
                idx = i
        t = np.array(tags[idx].pose_t)
        expert.estimated_pose = [-t[2], t[1], -t[0]]
        R = np.array(tags[idx].pose_R)
            
        # theta_x = math.atan2(R[2,1], R[2,2])
        # theta_y = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
        expert.estimated_angle = math.atan2(R[1,0], R[0,0])
        if expert.estimated_angle < 0:
            expert.estimated_angle =+ 2*math.pi
        elif expert.estimated_angle > 2 * math.pi:
            expert.estimated_angle -= 2 * math.pi
        # print("x: {}, y: {}, z: {}".format(abs(theta_x-env.cur_angle), abs(theta_y-env.cur_angle), abs(theta_y-env.cur_angle)))
        # print("theta_z: {:.3f}".format(theta_z))
        d = math.sqrt(t[0]**2 + t[2]**2)
        # print(d)
        dist_to_stop = []
        pos = env.cur_pos
        # print(pos)
        for obj in env.objects:
            if obj.kind == "sign_stop":
                dist_to_stop.append(((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)
        # print("estimated distance: {:.3f}, real distance: {}".format(d, dist_to_stop))
        abs_dist = [abs(dist_to_stop[0] - d), abs(dist_to_stop[1] - d), abs(dist_to_stop[2] - d)]

        # print("pose: {}, angle: {}".format(expert.estimated_pose, expert.estimated_angle))
        # print("estimated distance: {}, real_distance: {}".format(d, min(dist_to_stop)))
        # print("estimation error: {}".format(tags[0].pose_err))
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

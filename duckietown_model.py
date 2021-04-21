import math
import numpy as np
from scipy.stats import norm
from scipy.spatial.transform import Rotation as R

WHEEL_DIST = 0.102
DEFAULT_ROBOT_SPEED = 1.20
GAIN = 1.0
TRIM = 0.0
RADIUS = 0.0318
K = 27.0
LIMIT = 1.0
FRAMERATE = 30
ROBOT_SPEED = 1.2

MEAN_X = 0.27
STD_X = 0.02
MEAN_Y = 0.17
STD_Y = 0.03

norm_x = norm(loc = 0, scale=STD_X)
norm_y = norm(loc = 0, scale=STD_Y)


def get_transformation_matrix(x, y, theta):
    rot = R.from_euler('z', theta)
    T = np.zeros((4,4))
    T[3] = [0, 0, 0, 1]
    T[:3, 3] = [x, y, 0]
    T[:3, :3] = rot.as_matrix()
    return T

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

def update_pos(pos, angle, wheel_dist, wheelVels, deltaTime):
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

def forward_model(pos, theta, action):
    vel, steering = action

    # Distance between the wheels
    baseline = WHEEL_DIST

    # assuming same motor constants k for both motors
    k_r = K
    k_l = K

    # adjusting k by gain and trim
    k_r_inv = (GAIN + TRIM) / k_r
    k_l_inv = (GAIN - TRIM) / k_l

    omega_r = (vel + 0.5 * steering * baseline) / RADIUS
    omega_l = (vel - 0.5 * steering * baseline) / RADIUS

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, LIMIT), -LIMIT)
    u_l_limited = max(min(u_l, LIMIT), -LIMIT)

    vels = np.array([u_l_limited, u_r_limited])
    wheelVels = vels * ROBOT_SPEED * 1

    new_pos, new_theta = update_pos(pos, theta, WHEEL_DIST, wheelVels, 1 / FRAMERATE)

    return new_pos, new_theta

def transit_state(state, action):
    pos = [0, 0, 0]
    theta = 0

    new_pos, new_theta = forward_model(pos, theta, action)

    T_mat = get_transformation_matrix(new_pos[0], new_pos[1], new_theta)
    V_pos_ss_old = np.array([state[0], state[1], 0, 1])
    V_pos_ss_new = np.dot(np.linalg.inv(T_mat) , V_pos_ss_old)
    # print("V_pos_ss_new:", V_pos_ss_new[:2])
    return V_pos_ss_new[:2]

def measurement_prob(state, observation):
    pos = state
    prob_x = norm_x.pdf(observation[0] - pos[0])
    prob_y = norm_y.pdf(observation[1] - pos[1])
    return prob_x * prob_y
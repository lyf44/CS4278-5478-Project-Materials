import cv2
from dt_apriltags import Detector
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

def detect_stopsign(obs):
    '''
    returns robot's position relative to stop sign
    '''

    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(obs, estimate_tag_pose=True, camera_params=(305.5718893575089,308.8338858195428,303.0797142544728,231.8845403702499), tag_size=0.06822)
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

        pos_ss_r = [t[2][0], -t[0][0], 0]
        # print("et_pos_ss_r:", pos_ss_r) # tag's pose relative to camera

        return pos_ss_r
    else:
        return None

def get_transformation_matrix(x, y, theta):
    rot = R.from_euler('z', theta)
    T = np.zeros((4,4))
    T[3] = [0, 0, 0, 1]
    T[:3, 3] = [x, y, 0]
    T[:3, :3] = rot.as_matrix()
    return T

def detect_stopsign_gt(env, et_pos_ss_r):
    dist_to_stop = []
    stopsign_pos = []
    dist_to_est = []
    pos = env.cur_pos
    angle = env.cur_angle + math.radians(90)
    # print(pos, angle)

    T_m_r = get_transformation_matrix(pos[2], pos[0], angle)
    V_et_pos_ss_r = np.array([et_pos_ss_r[0], et_pos_ss_r[1], 0, 1])
    et_pos_ss_m = np.dot(T_m_r , V_et_pos_ss_r)[:3]

    for obj in env.objects:
        if obj.kind == "sign_stop":
            stopsign_pos.append([obj.pos[2], obj.pos[0], obj.pos[1], obj.y_rot])
            dist_to_stop.append(((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)
            dist_to_est.append(((obj.pos[2] - et_pos_ss_m[0]) ** 2 + (obj.pos[0] - et_pos_ss_m[1]) ** 2) ** 0.5)

    min_idx = np.argmin(dist_to_est)
    gt_pos_ss_m = stopsign_pos[min_idx]
    # print("gt_stopsign:", ss_pos)

    V_pos_ss_m = np.array([gt_pos_ss_m[0], gt_pos_ss_m[1], 0, 1])
    V_pos_ss_r = np.dot(np.linalg.inv(T_m_r) , V_pos_ss_m)
    # print("gt_pos_ss_r:", V_pos_ss_r[:3])

    return V_pos_ss_r[:3]

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
    # print(obs.shape)
    # cv2.imshow("obs_np", obs)
    # cv2.waitKey(1)

    # tags = at_detector.detect(obs, estimate_tag_pose=True, camera_params=(305.5718893575089,308.8338858195428,303.0797142544728,231.8845403702499), tag_size=0.035) # 0.06822
    tags = at_detector.detect(obs, estimate_tag_pose=True, camera_params=(220.2460277141687,238.6758484095299,301.8668918355899,227.0880056118307), tag_size=0.035) # 0.0292
    if len(tags) != 0:
        tmp_dist = 1000.0
        idx = None
        # print(len(tags))
        for i in range(len(tags)):
            if tags[i].tag_id == 1: # stop sign
                t = np.array(tags[i].pose_t)
                if math.sqrt(t[0][0] ** 2 + t[2][0] ** 2) < tmp_dist:
                    tmp_dist = math.sqrt(t[0][0] ** 2 + t[2][0] ** 2)
                    idx = i

        if idx is None:
            return None

        t = np.array(tags[idx].pose_t)
        pos_ss_r = [t[2][0], -t[0][0], 0]
        # print(tags[idx].pose_err)
        # print(tags[idx].pose_R)
        print("et_pos_ss_r:", pos_ss_r) # tag's pose relative to camera

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

def clamp_degree(degree):
    if degree >= 360:
        degree -= 360
    elif degree < 0:
        degree += 360

    return degree

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
            r_cur_angle = clamp_degree(math.degrees(env.cur_angle))
            ss_oppo_angle = clamp_degree(obj.y_rot + 180)
            # print(r_cur_angle, ss_oppo_angle)
            if math.fabs(ss_oppo_angle - r_cur_angle) < 60:
                stopsign_pos.append([obj.pos[2], obj.pos[0], obj.pos[1], obj.y_rot])
                dist_to_stop.append(((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)
                dist_to_est.append(((obj.pos[2] - et_pos_ss_m[0]) ** 2 + (obj.pos[0] - et_pos_ss_m[1]) ** 2) ** 0.5)

    if len(dist_to_est) <= 0:
        return None

    min_idx = np.argmin(dist_to_est)
    gt_pos_ss_m = stopsign_pos[min_idx]
    # print("gt_stopsign:", gt_pos_ss_m)

    V_pos_ss_m = np.array([gt_pos_ss_m[0], gt_pos_ss_m[1], 0, 1])
    V_pos_ss_r = np.dot(np.linalg.inv(T_m_r) , V_pos_ss_m)
    # print("gt_pos_ss_r:", V_pos_ss_r[:3])

    return V_pos_ss_r[:3]

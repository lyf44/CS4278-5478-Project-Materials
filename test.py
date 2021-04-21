import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def get_transformation_matrix(x, y, theta):
    rot = R.from_euler('z', theta)
    print(rot.as_matrix())
    T = np.zeros((4,4))
    T[3] = [0, 0, 0, 1]
    T[:3, 3] = [x, y, 0]
    T[:3, :3] = rot.as_matrix()
    print(T)
    return T

if __name__ == "__main__":
    T_m_r = get_transformation_matrix(1, 1, math.radians(90))
    # T_m_ss = get_transformation_matrix(ss_pos[2], ss_pos[0], ss_pos[3])
    V_r_to_ss_m = np.array([1, -1, 0, 1])
    V_r_to_ss_r = np.dot(np.linalg.inv(T_m_r) , V_r_to_ss_m)
    print("gt_stopsign:", V_r_to_ss_r[:3])
# ik_common/common/utils.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def se3_pos_ori_error(T_cur: np.ndarray, T_tar: np.ndarray):
    dp = T_cur[:3, 3] - T_tar[:3, 3]
    R_err = T_cur[:3, :3] @ T_tar[:3, :3].T
    tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(tr))
    return float(np.linalg.norm(dp)), ang

def expm_so3(omega):
    th = np.linalg.norm(omega)
    if th < 1e-12:
        return np.eye(3)
    k = omega / th
    K = np.array([[0,-k[2],k[1]], [k[2],0,-k[0]], [-k[1],k[0],0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

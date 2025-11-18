# ik_common/common/utils.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def se3_pos_ori_error(T_cur: np.ndarray, T_tar: np.ndarray):
    dp = T_cur[:3, 3] - T_tar[:3, 3]
    R_err = T_cur[:3, :3] @ T_tar[:3, :3].T
    tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(tr))
    return float(np.linalg.norm(dp)), ang

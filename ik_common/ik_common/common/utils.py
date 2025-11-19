"""Utility functions for SE(3) errors and basic math helpers."""

from typing import Tuple

import numpy as np


def se3_pos_ori_error(
    T_cur: np.ndarray,
    T_tar: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute position and orientation error between two SE(3) transforms.

    Both transforms are assumed to be 4x4 homogeneous matrices:
        T = [ R  p ]
            [ 0  1 ]

    Args:
        T_cur: Current pose of the end-effector (4x4 homogeneous matrix).
        T_tar: Target/desired pose of the end-effector (4x4 homogeneous matrix).

    Returns:
        pos_err: Euclidean distance between positions (in meters).
        ori_err: Orientation error as rotation angle (in radians).
    """
    # Position error: difference between translations.
    dp = T_cur[:3, 3] - T_tar[:3, 3]

    # Orientation error: relative rotation R_err = R_cur * R_tar^T.
    R_err = T_cur[:3, :3] @ T_tar[:3, :3].T

    # Compute rotation angle from rotation matrix using trace.
    # Clamp the trace value to avoid numerical issues with arccos.
    tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(tr))

    # Return position error (norm) and orientation error angle.
    return float(np.linalg.norm(dp)), ang

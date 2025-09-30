# ik_common/common/base.py
from abc import ABC, abstractmethod
import numpy as np

class IKSolverBase(ABC):
    def __init__(self, kinematics):
        self.kinematics = kinematics
        self.joint_names = kinematics.joint_names

    @abstractmethod
    def solve(self, target_pose: np.ndarray, q_seed=None):
        """
        Returns:
            q (np.ndarray) : solution
            ok (bool)
            info (dict)    : {'iters_total': int, ...}
        """
        pass

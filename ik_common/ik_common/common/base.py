"""Base classes for inverse kinematics (IK) solvers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


class IKSolverBase(ABC):
    """
    Abstract base class for inverse kinematics solvers.

    Concrete IK solvers should inherit from this class and implement
    the `solve` method.
    """

    def __init__(self, kinematics: Any) -> None:
        """
        Initialize the IK solver with a kinematic model.

        Args:
            kinematics: Object that provides forward kinematics, Jacobian,
                joint limits, etc. It must at least expose `joint_names`.
        """
        # Store the kinematics object, which will be used by derived solvers.
        self.kinematics = kinematics

        # Joint name list (e.g. ["joint1", "joint2", ...]).
        self.joint_names = kinematics.joint_names

    @abstractmethod
    def solve(
        self,
        target_pose: np.ndarray,
        q_seed: Optional[Iterable[float]] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Solve inverse kinematics for a given target end-effector pose.

        Args:
            target_pose: 4x4 homogeneous transform of the desired pose
                (position + orientation) in the world frame.
            q_seed: Optional initial guess for the joint configuration.

        Returns:
            q: IK solution as a 1D NumPy array of joint values.
            ok: True if a valid solution was found, False otherwise.
            info: Extra diagnostic information such as iteration count,
                convergence flags, etc.
        """
        raise NotImplementedError

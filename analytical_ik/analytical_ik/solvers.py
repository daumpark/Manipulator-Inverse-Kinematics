"""Analytical inverse kinematics solvers."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# 2-DOF Planar Analytical IK
# -----------------------------------------------------------------------------


@dataclass
class Planar2DParams:
    """
    Parameters for a 2-DOF planar manipulator.

    Attributes:
        L1: Length of the first link.
        L2: Length of the second link.
        joint_limits: Joint limits for (q1, q2) in radians.
    """

    L1: float = 0.35
    L2: float = 0.25
    joint_limits: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    )


class Planar2DAnalyticalIK:
    """
    Analytical IK for a 2-DOF planar manipulator.

    Assumptions:
        - Two revolute joints rotating about the z-axis.
        - Only the (x, y) position of the end-effector is considered.
        - z-position and orientation are ignored.
    """

    def __init__(self, params: Planar2DParams = Planar2DParams()) -> None:
        """Store planar manipulator parameters."""
        self.p = params

    def solve(
        self,
        target_xy: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
    ):
        """
        Solve planar 2-link IK for a target (x, y) position.

        Args:
            target_xy: 2D target position [x, y].
            q_seed: Optional seed configuration for selecting the closest
                solution.

        Returns:
            q_best: Selected joint configuration (2D array) or None.
            solutions: List of all valid solutions
            ok: True if at least one valid solution exists.
            info: Dictionary with extra info such as number of solutions
                or reason for failure.
        """
        x, y = float(target_xy[0]), float(target_xy[1])
        L1, L2 = self.p.L1, self.p.L2

        # Squared distance from origin to target.
        r2 = x * x + y * y

        # Cosine of joint 2 angle using the law of cosines.
        c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)

        # Check reachability with a small tolerance.
        if c2 < -1.0 - 1e-8 or c2 > 1.0 + 1e-8:
            return None, False, {"reason": "unreachable"}

        # Clamp to valid range to avoid numerical issues.
        c2 = np.clip(c2, -1.0, 1.0)

        # Two elbow configurations: elbow-up and elbow-down.
        s2_candidates = [
            np.sqrt(1.0 - c2 * c2),
            -np.sqrt(1.0 - c2 * c2),
        ]

        sols: List[np.ndarray] = []
        for s2 in s2_candidates:
            q2 = np.arctan2(s2, c2)

            # Compute q1 using geometric relations.
            k1 = L1 + L2 * np.cos(q2)
            k2 = L2 * np.sin(q2)
            q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

            # Normalize to [-pi, pi].
            q = np.array(
                [
                    np.arctan2(np.sin(q1), np.cos(q1)),
                    np.arctan2(np.sin(q2), np.cos(q2)),
                ]
            )

            # Check joint limits.
            if self._within_limits(q):
                sols.append(q)

        if not sols:
            return None, False, {"reason": "no_solution_within_limits"}

        # If a seed is not provided, use the zero configuration.
        if q_seed is None:
            q_seed = np.zeros(2)

        # Select the solution closest to the seed in angular distance.
        def _dist(q_sol: np.ndarray) -> float:
            # Wrap the difference to [-pi, pi] before taking the norm.
            return float(
                np.linalg.norm((q_sol - q_seed + np.pi) % (2.0 * np.pi) - np.pi)
            )

        best = min(sols, key=_dist)
        return best, True, {"solutions": len(sols)}

    def fk_points(self, q: np.ndarray) -> np.ndarray:
        """
        Compute 2D joint and end-effector positions for visualization.

        Args:
            q: Joint configuration [q1, q2] in radians.

        Returns:
            (3, 2) array with points:
                index 0: base (0, 0)
                index 1: joint 1 position
                index 2: end-effector position
        """
        L1, L2 = self.p.L1, self.p.L2
        q1, q2 = q

        p0 = np.array([0.0, 0.0])
        p1 = np.array([L1 * np.cos(q1), L1 * np.sin(q1)])
        p2 = p1 + np.array(
            [L2 * np.cos(q1 + q2), L2 * np.sin(q1 + q2)],
        )

        return np.stack([p0, p1, p2], axis=0)

    def _within_limits(self, q: np.ndarray) -> bool:
        """
        Check if the given configuration is within the joint limits.

        Args:
            q: Joint configuration [q1, q2].

        Returns:
            True if both joints are within their configured limits.
        """
        lo1, hi1 = self.p.joint_limits[0]
        lo2, hi2 = self.p.joint_limits[1]
        return (lo1 <= q[0] <= hi1) and (lo2 <= q[1] <= hi2)

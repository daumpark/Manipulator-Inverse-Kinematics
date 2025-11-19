"""Numerical inverse kinematics solvers using Jacobian-based methods."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin

from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel


class _JacobianBase(IKSolverBase):
    """
    Base class for Jacobian-based numerical IK solvers.

    This class provides:
        - Common hyperparameters (tolerance, gains, etc.).
        - SE(3) error computation in the LOCAL frame.
        - Optional position/orientation weighting utilities.
    """

    def __init__(self, kinematics: KinematicModel) -> None:
        """
        Initialize the base solver with a kinematic model.

        Args:
            kinematics: KinematicModel instance providing FK/Jacobian/limits.
        """
        super().__init__(kinematics)

        # Default initial configuration (in radians).
        self.q0 = np.deg2rad([0, 30, -30, 0, 0, 0])

        # Maximum number of iterations for the batch solve interface.
        self.max_iter: int = 150

        # Position and orientation tolerances.
        self.tol_pos: float = 1e-3  # [m]
        self.tol_rot: float = np.deg2rad(1.0)  # [rad]

        # Optional weights and step scaling.
        self.alpha: float = 0.7
        self.w_pos: float = 1.0
        self.w_rot: float = 0.7

    def _se3_err_local(
        self,
        T_current: np.ndarray,
        T_target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SE(3) error in the LOCAL frame.

        This implements:
            e = log( T_current^{-1} * T_target )
              = [v, w], where v is translational part, w rotational part.

        Args:
            T_current: 4x4 homogeneous transform of current EE pose.
            T_target: 4x4 homogeneous transform of target EE pose.

        Returns:
            6D error vector [vx, vy, vz, wx, wy, wz].
        """
        dMi = pin.SE3(T_current).actInv(pin.SE3(T_target))
        return pin.log(dMi).vector  # [v, w]

    def _weighted(
        self,
        e6: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optionally apply different weights to position and orientation errors.

        Args:
            e6: Raw 6D error vector [v, w].

        Returns:
            e_weighted: Weighted error vector.
            W: Weight matrix used (6x6 diagonal).
        """
        W = np.diag([self.w_pos] * 3 + [self.w_rot] * 3)
        return W @ e6, W


class JacobianTranspose(_JacobianBase):
    """
    Jacobian transpose IK (CLIK-style).

    Update rule:
        q_{k+1} = q_k + dt * J(q_k)^T * (Kp * e)

    Here, step() performs a single control-cycle-style update.
    """

    def __init__(self, kinematics: KinematicModel) -> None:
        """Initialize Jacobian transpose IK solver."""
        super().__init__(kinematics)
        self.Kp: float = 10.0
        # In real-time control, dt can be interpreted as the control period.
        self.dt: float = 0.01

    # ---------------------------------------------------------------------
    # Real-time style single-step update
    # ---------------------------------------------------------------------
    def step(
        self,
        q: np.ndarray,
        target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Perform a single IK update step.

        Args:
            q: Current joint configuration (shape (n,)).
            target_pose: Desired EE pose as 4x4 homogeneous transform.

        Returns:
            q_next: Updated joint configuration after one step.
            reached: True if the error is within tolerances.
            info: Dictionary with diagnostic info (pos_err, rot_err, method).
        """
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, dtype=float))

        # Current EE pose and 6D error.
        T_current, _ = kin.forward_kinematics(q)
        e6 = self._se3_err_local(T_current, target_pose)
        pos_err = float(np.linalg.norm(e6[:3]))
        rot_err = float(np.linalg.norm(e6[3:]))
        reached = (pos_err < self.tol_pos) and (rot_err < self.tol_rot)

        # Jacobian and update.
        J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        dq = J.T @ (self.Kp * e6)  # Interpreted as joint velocity.
        q_next = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))

        info: Dict[str, Any] = {
            "method": "JT",
            "pos_err": pos_err,
            "rot_err": rot_err,
        }
        return q_next, reached, info

    # ---------------------------------------------------------------------
    # Batch-style solve: repeatedly call step()
    # ---------------------------------------------------------------------
    def solve(
        self,
        target_pose: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Run iterative IK until convergence or max_iter is reached.

        This wraps step() in a loop, mainly for offline use.

        Args:
            target_pose: 4x4 homogeneous transform of desired EE pose.
            q_seed: Optional initial joint configuration; if None, q0 is used.

        Returns:
            q: Final joint configuration after iterations.
            ok: True if solution converged within tolerances.
            info: Dictionary with iteration count, method, and history.
        """
        kin = self.kinematics
        if q_seed is None:
            q = self.q0.copy()
        else:
            q = np.asarray(q_seed, dtype=float).copy()

        q = kin.clamp(q)
        q_hist: List[np.ndarray] = [q.copy()]
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())
            if reached:
                info: Dict[str, Any] = {
                    "iters_total": iters,
                    "method": info_step["method"],
                    "q_hist": np.asarray(q_hist, dtype=float),
                }
                return q, True, info

        # Convergence failure.
        info = {
            "iters_total": iters,
            "method": "JT",
            "q_hist": np.asarray(q_hist, dtype=float),
        }
        return q, False, info


class JacobianPinv(_JacobianBase):
    """
    Jacobian pseudoinverse IK.

    Update rule:
        dq = J^{+} * (Kp * e)
        q_{k+1} = q_k + dt * dq
    """

    def __init__(self, kinematics: KinematicModel) -> None:
        """Initialize pseudoinverse-based IK solver."""
        super().__init__(kinematics)
        self.Kp: float = 10.0
        self.dt: float = 0.01

    @staticmethod
    def _pinv(
        J: np.ndarray,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute a numerically robust pseudoinverse via SVD.

        Args:
            J: Jacobian matrix (m x n).
            tol: Singular values below this threshold are treated as zero.

        Returns:
            J_pinv: Pseudoinverse of J (n x m).
        """
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        S_inv = np.diag([1.0 / s if s > tol else 0.0 for s in S])
        return Vt.T @ S_inv @ U.T

    def step(
        self,
        q: np.ndarray,
        target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Perform one IK step using pseudoinverse.

        Args:
            q: Current joint configuration.
            target_pose: Desired EE pose as 4x4 homogeneous transform.

        Returns:
            q_next: Updated joint configuration.
            reached: True if error is within tolerance.
            info: Dictionary with diagnostic info.
        """
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, dtype=float))

        T_current, _ = kin.forward_kinematics(q)
        e6 = self._se3_err_local(T_current, target_pose)
        pos_err = float(np.linalg.norm(e6[:3]))
        rot_err = float(np.linalg.norm(e6[3:]))
        reached = (pos_err < self.tol_pos) and (rot_err < self.tol_rot)

        J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        J_pinv = self._pinv(J)
        dq = J_pinv @ (self.Kp * e6)
        q_next = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))

        info: Dict[str, Any] = {
            "method": "Jpinv",
            "pos_err": pos_err,
            "rot_err": rot_err,
        }
        return q_next, reached, info

    def solve(
        self,
        target_pose: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Run iterative pseudoinverse IK until convergence or max_iter.

        Args:
            target_pose: 4x4 desired EE pose.
            q_seed: Optional initial joint configuration.

        Returns:
            q: Final joint configuration.
            ok: True if converged, False otherwise.
            info: Dictionary including iteration count and joint history.
        """
        kin = self.kinematics
        if q_seed is None:
            q = self.q0.copy()
        else:
            q = np.asarray(q_seed, dtype=float).copy()

        q = kin.clamp(q)
        q_hist: List[np.ndarray] = [q.copy()]
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())
            if reached:
                info: Dict[str, Any] = {
                    "iters_total": iters,
                    "method": info_step["method"],
                    "q_hist": np.asarray(q_hist, dtype=float),
                }
                return q, True, info

        info = {
            "iters_total": iters,
            "method": "Jpinv",
            "q_hist": np.asarray(q_hist, dtype=float),
        }
        return q, False, info


class JacobianDLS(_JacobianBase):
    """
    Damped least squares (DLS) IK.

    Update rule:
        dq = J^T ( J J^T + λ^2 I )^{-1} (Kp * e)
        q_{k+1} = q_k + dt * dq

    Damping helps to avoid instability near singular configurations.
    """

    def __init__(self, kinematics: KinematicModel) -> None:
        """Initialize a damped least squares IK solver."""
        super().__init__(kinematics)
        self.Kp: float = 10.0
        self.dt: float = 0.01

        # Damping coefficient λ.
        self.lmbda: float = 0.05

    def step(
        self,
        q: np.ndarray,
        target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Perform one IK step using damped least squares.

        Args:
            q: Current joint configuration.
            target_pose: Target EE pose as 4x4 homogeneous transform.

        Returns:
            q_next: Updated joint configuration.
            reached: True if error is within tolerance.
            info: Dictionary with diagnostic info.
        """
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, dtype=float))

        T_current, _ = kin.forward_kinematics(q)
        e6 = self._se3_err_local(T_current, target_pose)
        pos_err = float(np.linalg.norm(e6[:3]))
        rot_err = float(np.linalg.norm(e6[3:]))
        reached = (pos_err < self.tol_pos) and (rot_err < self.tol_rot)

        J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        JJt = J @ J.T
        A = JJt + (self.lmbda**2) * np.eye(J.shape[0])
        J_damped_pinv = J.T @ np.linalg.inv(A)
        dq = J_damped_pinv @ (self.Kp * e6)
        q_next = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))

        info: Dict[str, Any] = {
            "method": "DLS",
            "pos_err": pos_err,
            "rot_err": rot_err,
        }
        return q_next, reached, info

    def solve(
        self,
        target_pose: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Run iterative DLS IK until convergence or max_iter.

        Args:
            target_pose: Desired EE pose (4x4 transform).
            q_seed: Optional initial joint configuration.

        Returns:
            q: Final joint configuration.
            ok: True if converged, False otherwise.
            info: Dictionary with iteration stats and joint history.
        """
        kin = self.kinematics
        if q_seed is None:
            q = self.q0.copy()
        else:
            q = np.asarray(q_seed, dtype=float).copy()

        q = kin.clamp(q)
        q_hist: List[np.ndarray] = [q.copy()]
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())
            if reached:
                info: Dict[str, Any] = {
                    "iters_total": iters,
                    "method": info_step["method"],
                    "q_hist": np.asarray(q_hist, dtype=float),
                }
                return q, True, info

        info = {
            "iters_total": iters,
            "method": "DLS",
            "q_hist": np.asarray(q_hist, dtype=float),
        }
        return q, False, info

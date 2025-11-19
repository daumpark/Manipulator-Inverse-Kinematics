"""Redundant IK solvers with nullspace and joint-limit handling.

Includes:
    - NullspacePositionOnly: Position-only CLIK with joint-centering nullspace.
    - WeightedCLIK: CLIK with joint-limit weighted damping.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel


# --------------------------------------------------------------------------- #
# Common utilities                                                            #
# --------------------------------------------------------------------------- #
def _damped_pinv(J: np.ndarray, lam: float = 1e-2):
    """Compute a damped pseudo-inverse of J.

    Uses SVD-based damped least-squares:
        J^# = V * diag(s / (s^2 + λ^2)) * U^T

    Args:
        J: Jacobian matrix with shape (m, n).
        lam: Damping factor λ.

    Returns:
        J_damped: Damped pseudo-inverse of J with shape (n, m).
        svd: Tuple (U, S, Vt) from the SVD of J.
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Sd = S / (S**2 + lam**2)
    J_damped = Vt.T @ np.diag(Sd) @ U.T
    return J_damped, (U, S, Vt)


def _projector(J: np.ndarray, lam: float = 1e-2):
    """Compute nullspace projector and damped pseudo-inverse.

    P = I - J^# J

    Args:
        J: Task Jacobian (m, n).
        lam: Damping factor for pseudo-inverse.

    Returns:
        P: Nullspace projector (n, n).
        Jp: Damped pseudo-inverse of J (n, m).
    """
    Jp, _ = _damped_pinv(J, lam)
    n = J.shape[1]
    P = np.eye(n) - Jp @ J
    return P, Jp


def _joint_centering_grad(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Compute a simple joint-centering gradient.

    This penalizes deviation from the middle of the joint range with a
    quadratic potential, so the gradient points towards the center.

    Args:
        q: Current joint configuration (n,).
        lower: Lower joint limits (n,).
        upper: Upper joint limits (n,).

    Returns:
        Gradient wrt q that pushes joints towards the mid-range.
    """
    mid = 0.5 * (lower + upper)
    rng = np.maximum(upper - lower, 1e-6)
    return (mid - q) / (rng**2)


def _jl_grad_dariush(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Joint-limit gradient as used in Dariush-style potentials.

    This corresponds to the gradient of a barrier-like function where:
        - The cost is small near the joint center.
        - The gradient magnitude grows as we approach joint limits.

    Args:
        q: Current joint configuration (n,).
        lower: Joint lower limits (n,).
        upper: Joint upper limits (n,).
        eps: Small positive number to prevent division by zero.

    Returns:
        grad: ∂H/∂q (n,), approximately zero in the middle and large near
              joint limits.
    """
    q = np.asarray(q, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)

    rng = np.maximum(hi - lo, eps)
    num = (rng**2) * (2.0 * q - hi - lo)
    den = (
        4.0
        * np.maximum(hi - q, eps) ** 2
        * np.maximum(q - lo, eps) ** 2
    )
    grad = num / den
    return grad


# --------------------------------------------------------------------------- #
# 1) NullspacePositionOnly                                                    #
# --------------------------------------------------------------------------- #
class NullspacePositionOnly(IKSolverBase):
    r"""Position-only CLIK with joint-centering nullspace term.

    Control law (continuous version):

        dq = J_pos^# * (Kp * e_pos)
             + k_ns * (I - J_pos^# J_pos) * z

    where:
        - J_pos is the position-only Jacobian.
        - e_pos is the positional error (in local frame).
        - z is a joint-centering gradient.
        - (I - J_pos^# J_pos) projects z into the nullspace of J_pos.

    The public API is:
        - step(q, target_pose): one CLIK update step.
        - solve(target_pose, q_seed): repeated calls to step() until convergence.
    """

    def __init__(self, kinematics: KinematicModel):
        """Initialize solver parameters and default configuration."""
        super().__init__(kinematics)

        # Default initial configuration (in degrees → radians).
        self.q0 = np.deg2rad([0.0, 30.0, -30.0, 0.0, 0.0, 0.0])

        # Iteration and convergence parameters.
        self.max_iter = 1000
        self.tol_pos = 1e-3  # [m]

        # CLIK gains and time step.
        self.Kp = 1.0
        self.dt = 0.02  # [s]

        # Damping for pseudo-inverse.
        self.lam = 1e-2

        # Nullspace gain applied to joint-centering term.
        self.k_ns = 0.2

    # ------------------------------------------------------------------ #
    # Real-time style interface: one control step                        #
    # ------------------------------------------------------------------ #
    def step(self, q: np.ndarray, target_pose: np.ndarray):
        """Perform a single CLIK step for position-only IK.

        Args:
            q: Current joint configuration (n,).
            target_pose: Desired end-effector pose as 4x4 homogeneous matrix.
                Only the translation part is used here.

        Returns:
            q_next: Updated joint configuration.
            reached: True if position error is below tolerance.
            info: Dict containing diagnostic information such as:
                - 'method': solver name.
                - 'pos_err': current position error (norm).
        """
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, float))

        # Forward kinematics and local-frame position error.
        T, _ = kin.forward_kinematics(q)
        R = T[:3, :3]

        # Compute error in the local frame of the end-effector.
        e = R.T @ (target_pose[:3, 3] - T[:3, 3])
        pos_err = np.linalg.norm(e)
        reached = pos_err < self.tol_pos

        # Position-only Jacobian in LOCAL frame.
        J6 = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        J = J6[:3, :]

        # Nullspace projector and pseudo-inverse.
        P, Jp = _projector(J, self.lam)

        # Joint-centering gradient.
        z = _joint_centering_grad(q, kin.lower, kin.upper)

        # CLIK + nullspace update.
        dq = Jp @ (self.Kp * e) + self.k_ns * (P @ z)

        # Integrate with simple Euler step and clamp to joint limits.
        q_next = kin.clamp(q + dq * self.dt)

        info = {
            "method": "NullspacePos",
            "pos_err": pos_err,
        }
        return q_next, reached, info

    # ------------------------------------------------------------------ #
    # Batch-style solve: call step() multiple times                      #
    # ------------------------------------------------------------------ #
    def solve(self, target_pose: np.ndarray, q_seed: np.ndarray | None = None):
        """Iteratively solve IK by repeatedly calling step().

        Args:
            target_pose: Target pose (4x4 homogeneous).
            q_seed: Optional initial configuration. If None, uses self.q0.

        Returns:
            q_sol: Final configuration (clamped to joint limits).
            ok: True if converged within max_iter.
            info: Dict with keys:
                - 'iters_total': total number of iterations.
                - 'method': solver identifier.
                - 'q_hist': array with the trajectory of q.
        """
        kin = self.kinematics

        if q_seed is None:
            q = self.q0.copy()
        else:
            q = np.asarray(q_seed, float).copy()

        q = kin.clamp(q)
        q_hist = [q.copy()]

        for it in range(self.max_iter):
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())

            if reached:
                info = {
                    "iters_total": it + 1,
                    "method": info_step["method"],
                    "q_hist": np.array(q_hist),
                }
                return kin.clamp(q), True, info

        info = {
            "iters_total": self.max_iter,
            "method": "NullspacePos",
            "q_hist": np.array(q_hist),
        }
        return kin.clamp(q), False, info


# --------------------------------------------------------------------------- #
# 2) Weighted-CLIK (joint-limit weighted)                                     #
# --------------------------------------------------------------------------- #
class WeightedCLIK(IKSolverBase):
    r"""Weighted CLIK with joint-limit-dependent weighting matrix.

    Effective pseudo-inverse:

        J* = W^{-1} J^T (J W^{-1} J^T + λ^2 I)^{-1}

    where:
        - W is a diagonal weighting matrix that becomes larger near joint
          limits, effectively reducing motion in those joints.
        - step() implements a single CLIK update with this weighted inverse.
    """

    def __init__(self, kinematics: KinematicModel):
        """Initialize solver parameters and default configuration."""
        super().__init__(kinematics)

        self.q0 = np.deg2rad([0.0, 30.0, -30.0, 0.0, 0.0, 0.0])

        self.max_iter = 1000
        self.tol_pos = 1e-3
        self.Kp = 1.0
        self.dt = 0.02
        self.lam = 1e-2

        # Scaling for the joint-limit weighting (currently not used directly).
        self.w_scale = 1.0

        # History of joint-limit gradient, used to detect approaching limits.
        self._g_prev: np.ndarray | None = None

    def _W(self, q: np.ndarray) -> np.ndarray:
        """Compute joint-limit-dependent weighting matrix W(q).

        The diagonal entries grow when the joint is moving closer to its limit,
        based on the Dariush-style gradient. This follows the idea that joints
        near limits should move less.

        Args:
            q: Current joint configuration (n,).

        Returns:
            W: Diagonal weighting matrix (n, n).
        """
        grad = np.abs(
            _jl_grad_dariush(
                q,
                self.kinematics.lower,
                self.kinematics.upper,
            ),
        )

        if self._g_prev is None:
            delta = np.zeros_like(grad)
        else:
            delta = grad - self._g_prev

        # When gradient is increasing (approaching limits),
        # increase the weight for that joint.
        w_diag = np.where(delta >= 0.0, 1.0 + grad, 1.0)

        self._g_prev = grad.copy()
        return np.diag(w_diag)

    # ------------------------------------------------------------------ #
    # Real-time style interface: one control step                        #
    # ------------------------------------------------------------------ #
    def step(self, q: np.ndarray, target_pose: np.ndarray):
        """Perform a single weighted CLIK step for position-only IK.

        Args:
            q: Current joint configuration (n,).
            target_pose: Desired end-effector pose (4x4 homogeneous).

        Returns:
            q_next: Updated joint configuration.
            reached: True if position error is below tolerance.
            info: Dict with 'method' and 'pos_err'.
        """
        kin = self.kinematics
        q = np.asarray(q, float)

        # Forward kinematics and local-frame position error.
        T, _ = kin.forward_kinematics(q)
        R = T[:3, :3]
        e = R.T @ (target_pose[:3, 3] - T[:3, 3])
        pos_err = np.linalg.norm(e)
        reached = pos_err < self.tol_pos

        # Position-only Jacobian in LOCAL frame.
        J = kin.jacobian(
            q,
            ref_frame=pin.ReferenceFrame.LOCAL,
        )[:3, :]

        # Joint-limit weighting.
        W = self._W(q)
        Winv = np.linalg.inv(W)

        # Weighted damped least-squares pseudo-inverse.
        A = J @ Winv @ J.T + (self.lam**2) * np.eye(3)
        J_star = Winv @ J.T @ np.linalg.inv(A)

        # CLIK update.
        dq = J_star @ (self.Kp * e)
        q_next = q + dq * self.dt

        info = {
            "method": "W-CLIK",
            "pos_err": pos_err,
        }
        return q_next, reached, info

    # ------------------------------------------------------------------ #
    # Batch-style solve                                                  #
    # ------------------------------------------------------------------ #
    def solve(self, target_pose: np.ndarray, q_seed: np.ndarray | None = None):
        """Iteratively solve IK using weighted CLIK steps.

        Args:
            target_pose: Target pose (4x4 homogeneous).
            q_seed: Optional initial configuration.

        Returns:
            q_sol: Final configuration.
            ok: True if converged.
            info: Dict with iteration statistics and history.
        """
        # Reset joint-limit gradient history at the start of a batch solve.
        self._g_prev = None

        if q_seed is None:
            q = self.q0.copy()
        else:
            q = np.asarray(q_seed, float).copy()

        q_hist = [q.copy()]

        for it in range(self.max_iter):
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())

            if reached:
                info = {
                    "iters_total": it + 1,
                    "method": info_step["method"],
                    "q_hist": np.array(q_hist),
                }
                return q, True, info

        info = {
            "iters_total": self.max_iter,
            "method": "W-CLIK",
            "q_hist": np.array(q_hist),
        }
        return q, False, info

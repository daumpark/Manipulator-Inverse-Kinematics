"""Redundant IK solvers with nullspace and joint-limit handling.

Includes:
    - NullspacePositionOnly: Position-only CLIK with joint-centering nullspace.
    - WeightedCLIK: CLIK with joint-limit weighted damping.
    - CTP_SVF_SD: Continuous Task Priority + Singular Value Filtering + Selective
      Damping, following Colomé & Torras / joint-limit avoidance literature.
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


def _svf_transform(s: np.ndarray, nu: float = 10.0, s0: float = 5e-3):
    """Apply Singular Value Filtering (SVF) to singular values.

    Based on Colomé & Torras:
        h_{ν,σ0}(σ) = (σ^3 + νσ^2 + 2σ + 2σ0) / (σ^2 + νσ + 2)

    Args:
        s: Singular values of the Jacobian.
        nu: Shape factor ν that controls the filtering behavior.
        s0: Minimum singular value bound σ0.

    Returns:
        Filtered singular values h(s).
    """
    return (s**3 + nu * s**2 + 2.0 * s + 2.0 * s0) / (s**2 + nu * s + 2.0)


def _filtered_pinv_svf(
    J: np.ndarray,
    nu: float = 10.0,
    s0: float = 5e-3,
):
    """Compute pseudo-inverse with SVF-based singular value filtering.

    Args:
        J: Jacobian (m, n).
        nu: SVF shape parameter.
        s0: SVF minimum singular value bound.

    Returns:
        J_filtered: Filtered pseudo-inverse of J (n, m).
        svf_data: Tuple (U, S, Sh, Vt) where Sh are filtered singular values.
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Sh = _svf_transform(S, nu=nu, s0=s0)
    inv_Sh = 1.0 / np.maximum(Sh, 1e-9)
    J_filtered = Vt.T @ np.diag(inv_Sh) @ U.T
    return J_filtered, (U, S, Sh, Vt)


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

        kin = self.kinematics
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


# --------------------------------------------------------------------------- #
# 3) CTP + SVF + SD                                                           #
# --------------------------------------------------------------------------- #
class CTP_SVF_SD(IKSolverBase):
    r"""CTP + SVF + SD redundant IK solver.

    Combines three ideas:
        - CTP (Continuous Task Priority): joint-limit avoidance as a primary
          task, with the main task projected into the remaining subspace.
        - SVF (Singular Value Filtering): robust pseudo-inverse near
          singularities.
        - SD (Selective Damping): limit excessive joint motion in the
          SVD modes (task space components).

    Notation roughly follows the original paper's equations (Eq. 18, 30, 34, 44).
    """

    def __init__(self, kinematics: KinematicModel):
        """Initialize solver parameters and default configuration."""
        super().__init__(kinematics)

        self.q0 = np.deg2rad([0.0, 30.0, -30.0, 0.0, 0.0, 0.0])

        self.max_iter = 1000
        self.tol_pos = 1e-3
        self.Kp = 1.0
        self.dt = 0.02

        # SVF parameters.
        self.nu = 10.0       # Shape factor (Section IV).
        self.sigma0 = 5e-3   # Minimum singular value bound.

        # Selective Damping parameter.
        self.gamma_max = 0.2  # Max joint step per iteration (rad).

        # CTP (joint-limit avoidance) parameters.
        self.lambda_jl = 0.5   # Joint-limit avoidance gain (Eq. 34).
        self.jl_buffer = 0.15  # Use 15% of total range as activation buffer.

    def _compute_activation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute joint-limit activation matrix H.

        Based on Section VI-C, Eq. (30) in the paper:
        - H is a diagonal matrix.
        - Each diagonal entry grows smoothly from 0 to 1 as the joint
          approaches its limits, within a buffer zone.

        Args:
            q: Current joint configuration (n,).

        Returns:
            H: Activation matrix (n, n) with values in [0, 1].
        """
        lower = self.kinematics.lower
        upper = self.kinematics.upper

        rng = upper - lower
        buffer = rng * self.jl_buffer

        h_diag = np.zeros_like(q)

        for i, val in enumerate(q):
            dist_min = val - lower[i]
            dist_max = upper[i] - val
            min_buf = buffer[i]

            # Near lower limit.
            if dist_min < min_buf:
                # Cubic smooth step from 0 (safe) to 1 (at limit).
                ratio = 1.0 - dist_min / np.maximum(min_buf, 1e-6)
                h_diag[i] = ratio**2 * (3.0 - 2.0 * ratio)

            # Near upper limit.
            elif dist_max < min_buf:
                ratio = 1.0 - dist_max / np.maximum(min_buf, 1e-6)
                h_diag[i] = ratio**2 * (3.0 - 2.0 * ratio)

            # Safe zone.
            else:
                h_diag[i] = 0.0

        return np.diag(h_diag)

    def _sd_clip(
        self,
        Vt: np.ndarray,
        dq_raw: np.ndarray,
        gamma_max: float,
    ) -> np.ndarray:
        """Apply Selective Damping in the SVD mode space.

        - Transform dq_raw into the right-singular-vector space (mode space).
        - Clip each component to [-gamma_max, gamma_max].
        - The caller maps it back via Vt.T.

        Args:
            Vt: Right singular vectors of the Jacobian (from SVD).
            dq_raw: Raw joint velocity update (n,).
            gamma_max: Max allowed step per mode.

        Returns:
            coeff_clipped: Clipped coefficients in mode space.
        """
        # Transform to mode coordinates.
        coeff = Vt @ dq_raw

        # Clip each mode independently.
        coeff_clipped = np.clip(coeff, -gamma_max, gamma_max)
        return coeff_clipped

    # ------------------------------------------------------------------ #
    # Real-time style interface: one control step                        #
    # ------------------------------------------------------------------ #
    def step(self, q: np.ndarray, target_pose: np.ndarray):
        r"""Perform one CTP + SVF + SD update step.

        Continuous Task Priority structure (simplified):

            Δθ = Δθ_1  +  J_2_proj^# (e_2 - J_2 Δθ_1)

        Here:
            - Δθ_1 is the joint-limit avoidance term (primary task).
            - J_2_proj is the Jacobian for the main task, modified by
              availability weights.
            - SVF and SD are applied on J_2 for robustness and bounded motion.

        Args:
            q: Current joint configuration (n,).
            target_pose: Desired end-effector pose (4x4 homogeneous).

        Returns:
            q_next: Updated joint configuration (clamped).
            reached: True if position error is below tolerance.
            info: Dict with method name, position error, and activation norm.
        """
        kin = self.kinematics
        q = np.asarray(q, float)

        # 1) Forward Kinematics & secondary-task error (position-only).
        T, _ = kin.forward_kinematics(q)
        R = T[:3, :3]
        e_pos = R.T @ (target_pose[:3, 3] - T[:3, 3])
        pos_err = np.linalg.norm(e_pos)
        reached = pos_err < self.tol_pos

        # 2) CTP primary task: joint-limit avoidance.
        # Δθ_jl = H * λ_jl * (q_center - q)
        H = self._compute_activation_matrix(q)
        q_center = 0.5 * (kin.lower + kin.upper)
        dq_jl = H @ (self.lambda_jl * (q_center - q))

        # 3) Secondary task Jacobian with CTP (availability weights).
        # J_mod = J * (I - H): joints fully activated (H=1) are removed.
        J = kin.jacobian(
            q,
            ref_frame=pin.ReferenceFrame.LOCAL,
        )[:3, :]
        I = np.eye(len(q))
        W_inv = I - H
        J_mod = J @ W_inv

        # 4) SVF: robust pseudo-inverse of J_mod.
        J_svf, (U, S, Sh, Vt) = _filtered_pinv_svf(
            J_mod,
            nu=self.nu,
            s0=self.sigma0,
        )

        # 5) Effective secondary-task error:
        # e_eff = Kp * e_pos - J * Δθ_jl
        e_eff = (self.Kp * e_pos) - (J @ dq_jl)

        # 6) Raw secondary-task step.
        dq_task_raw = J_svf @ e_eff

        # 7) Selective Damping in SVD mode space.
        coeff = self._sd_clip(Vt, dq_task_raw, self.gamma_max)
        dq_task = Vt.T @ coeff

        # 8) Final update and integration.
        dq_total = dq_jl + dq_task
        q_next = kin.clamp(q + dq_total * self.dt)

        info = {
            "method": "CTP+SVF+SD",
            "pos_err": pos_err,
            "H_norm": np.linalg.norm(np.diag(H)),
        }
        return q_next, reached, info

    # ------------------------------------------------------------------ #
    # Batch-style solve                                                  #
    # ------------------------------------------------------------------ #
    def solve(self, target_pose: np.ndarray, q_seed: np.ndarray | None = None):
        """Iteratively solve IK using CTP + SVF + SD steps.

        Args:
            target_pose: Target end-effector pose (4x4 homogeneous).
            q_seed: Optional initial configuration.

        Returns:
            q_sol: Final configuration (clamped).
            ok: True if converged.
            info: Dict with iteration statistics and history.
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
            "method": "CTP+SVF+SD",
            "q_hist": np.array(q_hist),
        }
        return kin.clamp(q), False, info

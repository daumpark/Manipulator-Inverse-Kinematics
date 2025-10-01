# redundant_ik/solvers.py
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel

# ---------- 공통 유틸 ----------
def _damped_pinv(J, lam=1e-2):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Sd = S / (S**2 + lam**2)
    return Vt.T @ np.diag(Sd) @ U.T, (U, S, Vt)

def _projector(J, lam=1e-2):
    Jp, _ = _damped_pinv(J, lam)
    n = J.shape[1]
    return np.eye(n) - Jp @ J, Jp

def _svf_transform(s, nu=10.0, s0=5e-3):
    # Colomé & Torras: h_{nu,s0}(σ) = (σ^3 + νσ^2 + 2σ + 2s0)/(σ^2 + νσ + 2)
    return (s**3 + nu*s**2 + 2.0*s + 2.0*s0) / (s**2 + nu*s + 2.0)

def _filtered_pinv_svf(J, nu=10.0, s0=5e-3):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Sh = _svf_transform(S, nu=nu, s0=s0)
    return Vt.T @ np.diag(1.0/np.maximum(Sh, 1e-9)) @ U.T, (U, S, Sh, Vt)

def _joint_centering_grad(q, lower, upper):
    mid = 0.5*(lower+upper); rng = np.maximum(upper-lower, 1e-6)
    return (mid - q) / (rng**2)

# --- solvers.py 상단 유틸 영역에 추가 ---
def _jl_grad_dariush(q, lower, upper, eps=1e-9):
    q  = np.asarray(q, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    rng = np.maximum(hi - lo, eps)
    num = (rng**2) * (2.0*q - hi - lo)
    den = 4.0 * np.maximum(hi - q, eps)**2 * np.maximum(q - lo, eps)**2
    return num / den  # = ∂H/∂q  (중앙 0, 한계에서 커짐)

# ---------- 1) NullspacePositionOnly ----------
class NullspacePositionOnly(IKSolverBase):
    """
    dq = J_pos# * (Kp * e_pos) + k_ns * (I - J_pos# J_pos) * z
      - z: joint-centering gradient
    """
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.q0 = np.deg2rad([55,0,205,0,85,0])
        self.max_iter = 1000
        self.tol_pos = 1e-3
        self.Kp = 1.0
        self.dt = 0.02
        self.lam = 1e-2
        self.k_ns = 0.2

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        for it in range(self.max_iter):
            T, _ = kin.forward_kinematics(q)
            R = T[:3,:3]
            e = R.T @ (target_pose[:3,3] - T[:3,3])
            if np.linalg.norm(e) < self.tol_pos:
                return kin.clamp(q), True, {'iters_total': it+1, 'method':'NullspacePos'}
            J6 = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            J = J6[:3,:]  # position-only
            P, Jp = _projector(J, self.lam)
            z = _joint_centering_grad(q, kin.lower, kin.upper)
            dq = Jp @ (self.Kp * e) + self.k_ns * (P @ z)
            # 간단 backtracking
            alpha = 1.0
            for _ in range(5):
                q_try = kin.clamp(q + alpha*dq*self.dt)
                T_try, _ = kin.forward_kinematics(q_try)
                if np.linalg.norm(target_pose[:3,3] - T_try[:3,3]) < np.linalg.norm(e):
                    q = q_try; break
                alpha *= 0.5
        return kin.clamp(q), False, {'iters_total': self.max_iter, 'method':'NullspacePos'}

# ---------- 3) Weighted-CLIK (joint-limit 가중) ----------
class WeightedCLIK(IKSolverBase):
    """
    J* = W^{-1} J^T (J W^{-1} J^T + λ^2 I)^-1
    - W: joint-limit 가까울수록 큰 가중 (속도 억제)
    """
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.q0 = np.deg2rad([0,30,-30,0,0,0])
        self.max_iter = 1000
        self.tol_pos = 1e-3
        self.Kp = 1.0
        self.dt = 0.02
        self.lam = 1e-2
        self.w_scale = 1.0

    def _W(self, q):
        g = np.abs(_jl_grad_dariush(q, self.kinematics.lower, self.kinematics.upper))
        if self._g_prev is None:
            delta = np.zeros_like(g)
        else:
            delta = g - self._g_prev
        w = np.where(delta >= 0.0, 1.0 + g, 1.0)  # 논문 식(10)
        self._g_prev = g.copy()
        return np.diag(w)

    def solve(self, target_pose, q_seed=None):
        self._g_prev = None
        kin = self.kinematics
        lam = self.lam
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        for it in range(self.max_iter):
            T, _ = kin.forward_kinematics(q)
            R = T[:3,:3]
            e = R.T @ (target_pose[:3,3] - T[:3,3])
            error = np.linalg.norm(e)
            print(error)
            if error < self.tol_pos:
                return kin.clamp(q), True, {'iters_total': it+1, 'method':'W-CLIK'}
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)[:3,:]
            W = self._W(q)
            Winv = np.linalg.inv(W)
            A = J @ Winv @ J.T + (lam**2)*np.eye(3)
            Jstar = Winv @ J.T @ np.linalg.inv(A)
            dq = Jstar @ (self.Kp * e)
            q = kin.clamp(q + dq * self.dt)
        return kin.clamp(q), False, {'iters_total': self.max_iter, 'method':'W-CLIK'}

# ---------- 4) CTP + SVF + SD ----------
class CTP_SVF_SD(IKSolverBase):
    """
    연속 과업 전환(CTP) + SVF 기반 pinv + 선택 감쇠(모드별 클립)
    """
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.q0 = np.deg2rad([55,0,205,0,85,0])
        self.max_iter = 1000
        self.tol_pos = 1e-3
        self.Kp = 1.0
        self.dt = 0.02
        self.lam = 1e-3
        self.nu = 10.0
        self.sigma0 = 5e-3
        self.gamma_max = 0.2

    def _sd_clip(self, Vt, dq, gamma_max):
        """
        Vt: shape (3, n)  from SVD(J)
        dq: shape (n,)    candidate joint update
        return: clipped modal coefficients (3,)
        """
        coeff = Vt @ dq                 # (3,)  관절공간 모드 좌표
        coeff = np.clip(coeff, -gamma_max, gamma_max)
        return coeff

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        for it in range(self.max_iter):
            T, _ = kin.forward_kinematics(q)
            R = T[:3,:3]
            e = R.T @ (target_pose[:3,3] - T[:3,3])
            if np.linalg.norm(e) < self.tol_pos:
                return kin.clamp(q), True, {'iters_total': it+1, 'method':'CTP_SVF_SD'}
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)[:3,:]
            # SVF pinv
            Jsf, (U,S,Sh,Vt) = _filtered_pinv_svf(J, nu=self.nu, s0=self.sigma0)  # U(3x3), Vt(3xn)
            raw = Jsf @ (self.Kp * e)                    # (n,)
            # 선택 감쇠: 모드별 클립 후 재조합
            comp = self._sd_clip(Vt, raw, self.gamma_max)  # (3,)
            dq = Vt.T @ comp                             # (n,)
            q = kin.clamp(q + dq * self.dt)
        return kin.clamp(q), False, {'iters_total': self.max_iter, 'method':'CTP_SVF_SD'}

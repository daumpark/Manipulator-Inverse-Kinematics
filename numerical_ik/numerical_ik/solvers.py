# numerical_ik/solvers.py
import numpy as np
import pinocchio as pin
from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel

class _JacobianBase(IKSolverBase):
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.q0 = np.deg2rad([0,30,-30,0,0,0])
        self.max_iter = 150
        self.tol_pos = 1e-3
        self.tol_rot = np.deg2rad(1.0)
        self.alpha = 0.7
        self.w_pos = 1.0
        self.w_rot = 0.7

    def _se3_err_local(self, T_current, T_target):
        dMi = pin.SE3(T_current).actInv(pin.SE3(T_target))
        return pin.log(dMi).vector  # [v, w]

    def _weighted(self, e6):
        W = np.diag([self.w_pos]*3 + [self.w_rot]*3)
        return W @ e6, W

class JacobianTranspose(_JacobianBase):
    """ dq = J^T (dx + Kp e) """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.Kp = 10.0
        self.dt = 0.01

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        iters = 0
        for it in range(self.max_iter):
            iters = it+1
            T_c, _ = kin.forward_kinematics(q)
            e6 = self._se3_err_local(T_c, target_pose)
            pe, re = np.linalg.norm(e6[:3]), np.linalg.norm(e6[3:])
            if pe < self.tol_pos and re < self.tol_rot:
                return q, True, {'iters_total': iters, 'method':'JT'}
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            dq = J.T @ (self.Kp * e6)
            q = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))
        return q, False, {'iters_total': iters, 'method':'JT'}

class JacobianPinv(_JacobianBase):
    """ dq = J^+ (dx + Kp e)  (pseudo-inverse) """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.Kp = 10.0
        self.dt = 0.01

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        iters = 0
        for it in range(self.max_iter):
            iters = it+1
            T_c, _ = kin.forward_kinematics(q)
            e6 = self._se3_err_local(T_c, target_pose)
            pe, re = np.linalg.norm(e6[:3]), np.linalg.norm(e6[3:])
            if pe < self.tol_pos and re < self.tol_rot:
                return q, True, {'iters_total': iters, 'method':'Jpinv'}
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            # Moore-Penrose
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            S_inv = np.diag([1/s if s>1e-6 else 0.0 for s in S])
            J_pinv = Vt.T @ S_inv @ U.T
            dq = J_pinv @ (self.Kp * e6)
            q = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))
        return q, False, {'iters_total': iters, 'method':'Jpinv'}

class JacobianDLS(_JacobianBase):
    """ DLS: dq = J^T (J J^T + λ^2 I)^-1 (dx + Kp e) """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.Kp = 10.0
        self.dt = 0.01
        self.lmbda = 0.05

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        iters = 0
        for it in range(self.max_iter):
            iters = it+1
            T_c, _ = kin.forward_kinematics(q)
            e6 = self._se3_err_local(T_c, target_pose)
            pe, re = np.linalg.norm(e6[:3]), np.linalg.norm(e6[3:])
            if pe < self.tol_pos and re < self.tol_rot:
                return q, True, {'iters_total': iters, 'method':'DLS'}
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            J_dpinv = J.T @ np.linalg.inv(J @ J.T + (self.lmbda**2)*np.eye(6))
            dq = J_dpinv @ (self.Kp * e6)
            q = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))
        return q, False, {'iters_total': iters, 'method':'DLS'}
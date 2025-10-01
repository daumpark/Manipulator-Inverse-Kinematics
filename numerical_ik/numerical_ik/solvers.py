# numerical_ik/solvers.py
import numpy as np
import pinocchio as pin
from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel

class _JacobianBase(IKSolverBase):
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.q0 = np.deg2rad(np.array([55.0, 0.0, 205.0, 0.0, 85.0, 0.0], float))
        self.max_iter = 1000
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
    """ dq = alpha * J^T * e """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.alpha = 0.7

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
                return kin.clamp(q), True, {'iters_total': iters, 'method':'JT'}
            eW, _ = self._weighted(e6)
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            dq = self.alpha * (J.T @ eW)
            # simple backtracking
            alpha = 1.0
            for _ in range(5):
                q_try = kin.clamp(q + alpha*dq)
                T_try, _ = kin.forward_kinematics(q_try)
                e_try = self._se3_err_local(T_try, target_pose)
                if np.linalg.norm(e_try) < np.linalg.norm(e6):
                    q = q_try; break
                alpha *= 0.5
        return kin.clamp(q), False, {'iters_total': iters, 'method':'JT'}

class JacobianPinv(_JacobianBase):
    """ dq = J^# * e  (pseudo-inverse) """
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
                return kin.clamp(q), True, {'iters_total': iters, 'method':'Jpinv'}
            eW, _ = self._weighted(e6)
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            # Moore-Penrose
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            S_inv = np.diag([1/s if s>1e-6 else 0.0 for s in S])
            J_pinv = Vt.T @ S_inv @ U.T
            dq = J_pinv @ eW
            # line search
            alpha = self.alpha
            for _ in range(5):
                q_try = kin.clamp(q + alpha*dq)
                T_try, _ = kin.forward_kinematics(q_try)
                if np.linalg.norm(self._se3_err_local(T_try, target_pose)) < np.linalg.norm(e6):
                    q = q_try; break
                alpha *= 0.5
        return kin.clamp(q), False, {'iters_total': iters, 'method':'Jpinv'}

class JacobianDLS(_JacobianBase):
    """ DLS: dq = J^T (J J^T + λ^2 I)^-1 e """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.lmbda = 0.05

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        lam = self.lmbda
        iters = 0
        for it in range(self.max_iter):
            iters = it+1
            T_c, _ = kin.forward_kinematics(q)
            e6 = self._se3_err_local(T_c, target_pose)
            pe, re = np.linalg.norm(e6[:3]), np.linalg.norm(e6[3:])
            if pe < self.tol_pos and re < self.tol_rot:
                return kin.clamp(q), True, {'iters_total': iters, 'method':'DLS'}
            eW, W = self._weighted(e6)
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T
            dq_nom = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), eW)
            alpha = self.alpha
            improved=False
            for _ in range(5):
                q_try = kin.clamp(q + alpha*dq_nom)
                T_try, _ = kin.forward_kinematics(q_try)
                if np.linalg.norm(W @ self._se3_err_local(T_try, target_pose)) < np.linalg.norm(W @ e6):
                    q = q_try; improved = True; break
                alpha *= 0.5
            if not improved:
                lam *= 2.0
        return kin.clamp(q), False, {'iters_total': iters, 'method':'DLS'}
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
        """
        log( T_current^{-1} * T_target ) in LOCAL frame
        e = [v, w]
        """
        dMi = pin.SE3(T_current).actInv(pin.SE3(T_target))
        return pin.log(dMi).vector  # [v, w]

    def _weighted(self, e6):
        """
        옵션: 위치/자세 가중치 줄 때 사용 가능
        """
        W = np.diag([self.w_pos]*3 + [self.w_rot]*3)
        return W @ e6, W


class JacobianTranspose(_JacobianBase):
    """
    Jacobian Transpose IK (CLIK-style)
      q_{k+1} = q_k + dt * J(q_k)^T * (Kp * e)
    여기서 step()은 "한 제어 주기" 동안의 업데이트만 수행한다.
    """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.Kp = 10.0
        # 실시간 제어에서 dt는 제어 주기(초)로 해석 가능
        self.dt = 0.01

    # --- 실시간 제어용 한 스텝 업데이트 ---
    def step(self, q, target_pose):
        """
        q: 현재 관절각 (np.ndarray, shape (n,))
        target_pose: 4x4 hom. transform (SE3)
        return:
          q_next: 업데이트된 관절각
          reached: (bool) 오차가 tol 안에 들어왔는지
          info: dict (pos_err, rot_err 등)
        """
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, float))

        # 현재 EE pose와 에러
        T_c, _ = kin.forward_kinematics(q)
        e6 = self._se3_err_local(T_c, target_pose)
        pe = np.linalg.norm(e6[:3])
        re = np.linalg.norm(e6[3:])
        reached = (pe < self.tol_pos) and (re < self.tol_rot)

        # 자코비안 및 업데이트
        J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        dq = J.T @ (self.Kp * e6)  # 사실상 qdot
        q_next = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))

        info = {
            'method': 'JT',
            'pos_err': pe,
            'rot_err': re,
        }
        return q_next, reached, info

    # --- 기존 batch solve는 step()을 여러 번 호출하는 래퍼로 유지 ---
    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        q_hist = [q.copy()]
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())
            if reached:
                info = {
                    'iters_total': iters,
                    'method': info_step['method'],
                    'q_hist': np.array(q_hist),
                }
                return q, True, info

        # 수렴 실패
        info = {
            'iters_total': iters,
            'method': 'JT',
            'q_hist': np.array(q_hist),
        }
        return q, False, info


class JacobianPinv(_JacobianBase):
    """
    Jacobian Pseudoinverse IK:
      dq = J^+ * (Kp * e)
      q_{k+1} = q_k + dt * dq
    """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.Kp = 10.0
        self.dt = 0.01

    def _pinv(self, J, tol=1e-6):
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        S_inv = np.diag([1/s if s > tol else 0.0 for s in S])
        return Vt.T @ S_inv @ U.T

    def step(self, q, target_pose):
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, float))

        T_c, _ = kin.forward_kinematics(q)
        e6 = self._se3_err_local(T_c, target_pose)
        pe = np.linalg.norm(e6[:3])
        re = np.linalg.norm(e6[3:])
        reached = (pe < self.tol_pos) and (re < self.tol_rot)

        J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        J_pinv = self._pinv(J)
        dq = J_pinv @ (self.Kp * e6)
        q_next = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))

        info = {
            'method': 'Jpinv',
            'pos_err': pe,
            'rot_err': re,
        }
        return q_next, reached, info

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        q_hist = [q.copy()]
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())
            if reached:
                info = {
                    'iters_total': iters,
                    'method': info_step['method'],
                    'q_hist': np.array(q_hist),
                }
                return q, True, info

        info = {
            'iters_total': iters,
            'method': 'Jpinv',
            'q_hist': np.array(q_hist),
        }
        return q, False, info


class JacobianDLS(_JacobianBase):
    """
    Damped Least Squares IK:
      dq = J^T ( J J^T + λ^2 I )^{-1} (Kp * e)
      q_{k+1} = q_k + dt * dq
    """
    def __init__(self, kin: KinematicModel):
        super().__init__(kin)
        self.Kp = 10.0
        self.dt = 0.01
        self.lmbda = 0.05

    def step(self, q, target_pose):
        kin = self.kinematics
        q = kin.clamp(np.asarray(q, float))

        T_c, _ = kin.forward_kinematics(q)
        e6 = self._se3_err_local(T_c, target_pose)
        pe = np.linalg.norm(e6[:3])
        re = np.linalg.norm(e6[3:])
        reached = (pe < self.tol_pos) and (re < self.tol_rot)

        J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
        JJt = J @ J.T
        A = JJt + (self.lmbda**2) * np.eye(J.shape[0])
        J_dpinv = J.T @ np.linalg.inv(A)
        dq = J_dpinv @ (self.Kp * e6)
        q_next = kin.clamp(pin.integrate(kin.model, q, dq * self.dt))

        info = {
            'method': 'DLS',
            'pos_err': pe,
            'rot_err': re,
        }
        return q_next, reached, info

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        q_hist = [q.copy()]
        iters = 0

        for it in range(self.max_iter):
            iters = it + 1
            q, reached, info_step = self.step(q, target_pose)
            q_hist.append(q.copy())
            if reached:
                info = {
                    'iters_total': iters,
                    'method': info_step['method'],
                    'q_hist': np.array(q_hist),
                }
                return q, True, info

        info = {
            'iters_total': iters,
            'method': 'DLS',
            'q_hist': np.array(q_hist),
        }
        return q, False, info

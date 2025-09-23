import numpy as np
from abc import ABC, abstractmethod
import pinocchio as pin
import os
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from rclpy.logging import get_logger

# ================================================================
#  Kinematic Model (URDF-driven FK/Jacobian) + PiPER geometry
# ================================================================
class KinematicModel:
    """
    Loads PiPER 6-DoF URDF and exposes:
      - forward_kinematics(q): (4x4 T_ee, [T_joint1..T_joint6])
      - jacobian(q): 6x6 geometric Jacobian at EE
      - joint_names, lower/upper limits (rad)
    Also estimates 'd6' (distance from joint6 origin to EE along joint6 axis)
    to support classic wrist-center decoupling used by FABRIK position-only IK.
    """
    def __init__(self):
        # Locate URDF from installed package; fall back to local path if needed.
        try:
            desc_share = get_package_share_directory('piper_description')
            urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
        except Exception:
            urdf_file = '/mnt/data/piper_no_gripper_description.urdf'  # fallback

        if not os.path.exists(urdf_file):
            raise FileNotFoundError(f"PiPER URDF not found: {urdf_file}")

        self.urdf_file = urdf_file

        # Build Pinocchio model
        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_file)
        self.model = self.robot.model
        self.data = self.robot.data

        # EE frame detection (prefer link6, else last frame)
        self.ee_frame_id = None
        self.ee_link_name = "link6"
        try:
            fid = self.model.getFrameId(self.ee_link_name)
            if isinstance(fid, (int, np.integer)) and (0 <= fid < self.model.nframes):
                self.ee_frame_id = fid
        except Exception:
            self.ee_frame_id = None
        if self.ee_frame_id is None:
            last_id = self.model.nframes - 1
            if last_id >= 0:
                self.ee_frame_id = last_id

        # Actuated joint names (first 6 one-DoF joints)
        names = []
        for i in range(1, 7):
            nm = f"joint{i}"
            try:
                if self.model.getJointId(nm) > 0:
                    names.append(nm)
            except Exception:
                pass
        if len(names) != 6:
            names = [j.name for j in self.model.joints if j.nq > 0][:6]
        self.joint_names = names

        # Joint limits from URDF
        lower, upper = [], []
        root = ET.parse(urdf_file).getroot()
        lim_map = {}
        for j in root.findall('joint'):
            nm = j.attrib.get('name', '')
            lim = j.find('limit')
            if lim is not None and 'lower' in lim.attrib and 'upper' in lim.attrib:
                lim_map[nm] = (float(lim.attrib['lower']), float(lim.attrib['upper']))
        for nm in self.joint_names:
            lo, hi = lim_map.get(nm, (-np.inf, np.inf))
            lower.append(lo); upper.append(hi)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

        self.logger = get_logger('ik_solvers')

        # Estimate d6 at zero configuration (project EE offset onto joint6 axis)
        try:
            qzero = np.zeros(6, dtype=float)
            self._full_fk(qzero)
            j6_name = self.joint_names[-1]
            j6_id = self.model.getJointId(j6_name)
            Tj6 = self.data.oMi[j6_id]
            Tee = self.data.oMf[self.ee_frame_id]
            # Use joint-6 axis (z of joint-6 frame) in world:
            z = Tj6.rotation[:, 2]
            off = Tee.translation - Tj6.translation
            self.d6 = float(np.dot(z, off))
        except Exception:
            self.d6 = 0.0  # safe fallback

    # ---------- Core geometry helpers ----------
    def _full_fk(self, q):
        q = np.asarray(q, dtype=float).flatten()
        if q.size != 6:
            raise ValueError("FK expects 6 joint values (rad).")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def forward_kinematics(self, q):
        """
        FK using the URDF (pinocchio).
        Returns:
          T_ee (4x4), [T_joint1 ... T_joint6] (list of 4x4)
        """
        self._full_fk(q)

        # Joint transforms (world w.r.t. each joint)
        Ts = []
        for nm in self.joint_names:
            jid = self.model.getJointId(nm)
            Ts.append(self.data.oMi[jid].homogeneous.copy())

        # EE transform (robust)
        fid = self.ee_frame_id
        if not (0 <= fid < len(self.data.oMf)):
            fid = len(self.data.oMf) - 1
        T_ee = self.data.oMf[fid].homogeneous.copy()
        return T_ee, Ts

    def jacobian(self, q, ref_frame=pin.ReferenceFrame.LOCAL):
        q = np.asarray(q, dtype=float).flatten()
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.ee_frame_id
        if not (0 <= fid < self.model.nframes):
            fid = self.model.nframes - 1
        J = pin.computeFrameJacobian(self.model, self.data, q, fid, ref_frame)
        return J

    # ---------- Utilities ----------
    def clamp(self, q):
        return np.minimum(np.maximum(q, self.lower), self.upper)


# ================================================================
#  Base Class
# ================================================================
class IKSolverBase(ABC):
    def __init__(self, kinematics: KinematicModel):
        self.kinematics = kinematics
        self.joint_names = kinematics.joint_names

    @abstractmethod
    def solve(self, target_pose: np.ndarray, q_seed=None):
        """
        :param target_pose: 4x4 homogeneous transform (base_link -> EE target)
        :param q_seed: optional 6-dim seed (np.ndarray)
        :return: (q[6], reachable: bool)
        """
        pass

# ================================================================
#  Jacobian IK (Damped Least Squares, with limits)
# ================================================================
class JacobianIKSolver(IKSolverBase):
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.q0 = np.deg2rad(np.array([55.0, 0.0, 205.0, 0.0, 85.0, 0.0], dtype=float))
        self.max_iter = 150
        self.tol_pos = 1e-3
        self.tol_rot = np.deg2rad(1.0)
        self.lmbda = 0.05
        self.alpha = 0.7
        self.w_pos = 1.0
        self.w_rot = 0.7

    def _se3_error_local(self, T_current, T_target):
        dMi = pin.SE3(T_current).actInv(pin.SE3(T_target))
        err = pin.log(dMi).vector
        return err

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        lam = self.lmbda
        for _ in range(self.max_iter):
            T_c, _ = kin.forward_kinematics(q)
            err6 = self._se3_error_local(T_c, target_pose)
            pe, re = np.linalg.norm(err6[:3]), np.linalg.norm(err6[3:])
            if pe < self.tol_pos and re < self.tol_rot:
                return kin.clamp(q), True

            W = np.diag([self.w_pos]*3 + [self.w_rot]*3)
            e = W @ err6
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T
            dq_nom = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), e)

            # backtracking line search
            alpha = self.alpha
            for _ in range(5):
                q_try = kin.clamp(q + alpha * dq_nom)
                T_try, _ = kin.forward_kinematics(q_try)
                e_try = self._se3_error_local(T_try, target_pose)
                if np.linalg.norm(W @ e_try) < np.linalg.norm(e):
                    q = q_try
                    break
                alpha *= 0.5
            else:
                lam *= 2.0

        return kin.clamp(q), False

class FabrikRIKSolver(IKSolverBase):
    """
    FABRIK-R (Santos et al., 2021) position-only IK for a 6-DoF serial chain.
    - Uses world-frame 1-DoF joint axes from Pinocchio geometric Jacobian.
    - Forward/Backward passes follow the paper's plane construction rules:
      Forward:  p_prev = p[i+1] (EE쪽), p_next = (기본) p[i-1] (베이스쪽 앵커)
      Backward: p_prev = p[i]   (베이스쪽), p_next = (기본) p[i+2] (EE쪽 앵커)
    """

    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.max_iter = 80
        self.tol = 1e-3
        lo, hi = kinematics.lower, kinematics.upper
        mid = np.where(np.isfinite(lo + hi), 0.5 * (lo + hi), 0.0)
        self.q0 = np.where(np.isfinite(mid), mid, 0.0)
        self.axis_parallel_tol = 0.995  # |dot| >= tol -> nearly parallel
        self.debug = False              # True로 두면 불변량 체크를 프린트

    # ---------- helpers ----------
    def _axes_world(self, q):
        # WORLD-frame joint axes from the WORLD geometric Jacobian (angular rows)
        Jw = self.kinematics.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
        A = []
        for j in range(6):
            a = np.array(Jw[3:, j]).reshape(3,)
            n = np.linalg.norm(a)
            A.append(a / n if n > 1e-12 else np.array([0., 0., 1.]))
        return A  # [a1..a6], unit

    def _chain_positions(self, q):
        Tee, Ts = self.kinematics.forward_kinematics(q)
        ps = [T[:3, 3].copy() for T in Ts]  # p1..p6
        # wrist center (URDF에 따라 d6≈0이면 EE와 동일)
        if abs(self.kinematics.d6) > 1e-9:
            a6 = self._axes_world(q)[5]
            ps.append(Tee[:3, 3] - self.kinematics.d6 * a6)
        else:
            ps.append(Tee[:3, 3].copy())
        return ps  # [p1..p6, p_wc] 길이 7

    def _link_lengths(self, ps):
        # L[i] = |p[i+1]-p[i]| for i in [0..5]
        return [float(np.linalg.norm(ps[i+1] - ps[i])) for i in range(6)]

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    @staticmethod
    def _any_perp(n):
        n = FabrikRIKSolver._unit(n)
        if abs(n[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])
        v = np.cross(n, a)
        vn = np.linalg.norm(v)
        return v / vn if vn > 1e-12 else np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _rodrigues(v, axis, theta):
        # rotate vector v around unit axis by theta (rad)
        a = FabrikRIKSolver._unit(axis)
        c, s = np.cos(theta), np.sin(theta)
        return v*c + np.cross(a, v)*s + a*np.dot(a, v)*(1.0 - c)

    # --- anchor 선택: s = p_next - p_prev 정의용 p_next 인덱스 ---
    def _choose_anchor_for_s_forward(self, i, axes):
        # Forward: p_prev = p[i+1] (EE쪽), s는 기본적으로 베이스쪽 정보를 쓰는 게 안정적.
        # 1) 베이스쪽으로 축이 다른 첫 관절
        for j in range(i-1, -1, -1):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        # 2) 없다면 EE쪽으로
        for j in range(i+1, 6):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        # 3) 그래도 없으면 가장 가까운 베이스쪽 이웃
        return max(i-1, 0)

    def _choose_anchor_for_s_backward(self, i, axes):
        # Backward: p_prev = p[i] (베이스쪽), s는 기본적으로 EE쪽 정보를 쓰는 게 안정적.
        # 1) EE쪽으로 축이 다른 첫 관절
        for j in range(i+1, 6):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        # 2) 없다면 베이스쪽으로
        for j in range(i-1, -1, -1):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        # 3) 그래도 없으면 가장 가까운 EE쪽 이웃
        return min(i+1, 5)

    # --- 식(6) 해법: A cos(2θ) + B sin(2θ) = C ---
    @staticmethod
    def _solve_theta_eq6(K1, K2, K3):
        # (K1-K2)cos(2θ) + K3 sin(2θ) = -K2
        A = K1 - K2
        B = K3
        C = -K2
        R = float(np.hypot(A, B))
        if R < 1e-12:
            return [0.0]
        rhs = np.clip(C / R, -1.0, 1.0)
        phi = float(np.arctan2(B, A))
        alpha = float(np.arccos(rhs))
        return [0.5 * (phi + alpha), 0.5 * (phi - alpha)]

    def _define_planes_and_rotate(self, p_prev, p_i, p_next, a_prev, d_prev):
        """
        - Π_prev: plane through p_prev with normal a_prev (hinge plane).
          먼저 p_i를 Π_prev 원(반지름 d_prev) 위의 임시점 p_hat으로 투영.
        - Π_i:  En(법선)을 s = p_next - p_prev 에 직교하도록 정하고(식(2)),
                En은 El=a_prev을 축으로 Ev(=p_prev->p_hat) 를 각 θ만큼 회전한 결과(식(3)),
                θ는 식(6)으로 결정.
        - 그 후 p_hat을 El 축으로 θ만큼 돌려 p_i_new를 얻는다.
        """
        El = self._unit(a_prev)

        # Ev: p_prev -> p_i 를 Π_prev로 사영한 방향
        v = p_i - p_prev
        v_perp = v - El * np.dot(El, v)
        if np.linalg.norm(v_perp) < 1e-9:
            v_perp = self._any_perp(El)
        Ev = self._unit(v_perp)
        p_hat = p_prev + d_prev * Ev

        # s = p_next - p_prev  (방향 중요!)
        s = p_next - p_prev
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            # 앵커가 나쁜 경우: Π_prev 유지 (θ=0)
            return p_hat, Ev

        s_u = s / ns
        t = np.cross(El, Ev)

        # K1 = s·Ev,  K2 = (El·Ev)(s·El),  K3 = s·(El×Ev)
        K1 = float(np.dot(s_u, Ev))
        K2 = float(np.dot(El, Ev) * np.dot(s_u, El))
        K3 = float(np.dot(s_u, t))

        thetas = self._solve_theta_eq6(K1, K2, K3)

        # 후보 중 p_i에 가장 가까운 것을 선택
        best_p = None
        best_u = None
        best_cost = 1e18
        for th in thetas:
            u_rot = self._rodrigues(Ev, El, th)
            p_rot = p_prev + d_prev * u_rot
            cost = np.linalg.norm(p_rot - p_i)
            if cost < best_cost:
                best_cost, best_p, best_u = cost, p_rot, u_rot

        if best_p is None:  # 수치 문제가 있으면 퇴각
            return p_hat, Ev
        return best_p, best_u

    # ---------- 불변량 체크 (옵션) ----------
    def _check_invariants(self, phase, i, p, L, a_prev, center_idx, tol=1e-6):
        if not self.debug:
            return
        # 링크 길이 보존
        ell = np.linalg.norm(p[i+1] - p[i])
        if abs(ell - L[i]) > tol:
            print(f"[{phase}] link {i} length violation: {ell:.6g} vs {L[i]:.6g}")
        # Π_prev 직교(업데이트된 방향과 a_prev 직교)
        center = p[center_idx]
        v = (p[i] - center) if phase == "F" else (p[i+1] - center)
        dot_prev = abs(np.dot(self._unit(v), self._unit(a_prev)))
        if dot_prev > 1e-3:
            print(f"[{phase}] plane Π_prev violated at i={i}: |v·a_prev|={dot_prev:.3e}")

    # ---------- FABRIK-R passes ----------
    def _forward_pass(self, p, axes, L, target_wc):
        # end fixed to target
        p[6] = target_wc.copy()
        # i = 5..0 : update p[i], center is p[i+1]
        for i in range(5, -1, -1):
            a_prev = axes[i]
            idx_for_s = self._choose_anchor_for_s_forward(i, axes)
            # 회전 중심 = p[i+1], 업데이트 대상 = p[i], s-anchor = p[idx_for_s]
            p_new, _ = self._define_planes_and_rotate(
                p_prev=p[i+1],
                p_i=p[i],
                p_next=p[idx_for_s],
                a_prev=a_prev,
                d_prev=L[i]
            )
            p[i] = p_new
            self._check_invariants("F", i, p, L, a_prev, center_idx=i+1)

    def _backward_pass(self, p, axes, L, base):
        # base fixed
        p[0] = base.copy()
        # i = 0..5 : update p[i+1], center is p[i]
        for i in range(0, 6):
            a_prev = axes[i]
            idx_for_s = self._choose_anchor_for_s_backward(i, axes)
            p_new, _ = self._define_planes_and_rotate(
                p_prev=p[i],
                p_i=p[i+1],
                p_next=p[idx_for_s],
                a_prev=a_prev,
                d_prev=L[i]
            )
            p[i+1] = p_new
            self._check_invariants("B", i, p, L, a_prev, center_idx=i)

    # ---------- q recovery ----------
    @staticmethod
    def _signed_angle_around_axis(v_from, v_to, axis):
        a = v_to * 0.0 + axis
        a = a / (np.linalg.norm(a) + 1e-12)
        vf = v_from - a * np.dot(a, v_from)
        vt = v_to   - a * np.dot(a, v_to)
        nf = np.linalg.norm(vf); nt = np.linalg.norm(vt)
        if nf < 1e-12 or nt < 1e-12:
            return 0.0
        vf /= nf; vt /= nt
        s = np.dot(a, np.cross(vf, vt))
        c = np.clip(np.dot(vf, vt), -1.0, 1.0)
        return float(np.arctan2(s, c))

    def _positions_to_q(self, q_in, p_des):
        kin = self.kinematics
        q = q_in.copy()
        axes_world = self._axes_world(q)
        for i in range(6):
            Tee, Ts = kin.forward_kinematics(q)
            pj = Ts[i][:3, 3]
            aj = axes_world[i]
            if i < 5:
                pj_next_cur = Ts[i+1][:3, 3]
            else:
                if abs(kin.d6) > 1e-9:
                    pj_next_cur = Tee[:3, 3] - kin.d6 * axes_world[5]
                else:
                    pj_next_cur = Tee[:3, 3]
            v_cur = pj_next_cur - pj
            v_des = p_des[i+1] - p_des[i]
            if np.linalg.norm(v_des) < 1e-12 or np.linalg.norm(v_cur) < 1e-12:
                continue
            dtheta = self._signed_angle_around_axis(v_cur, v_des, aj)
            # 약간 보수적으로 업데이트하면 안정적
            q[i] = np.clip(q[i] + 0.8 * dtheta, kin.lower[i], kin.upper[i])
            axes_world = self._axes_world(q)  # 다음 관절에 영향 주므로 갱신
        return q

    # ---------- main ----------
    def solve(self, target_pose: np.ndarray, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        # 초기 체인/길이/베이스
        p = self._chain_positions(q)
        L = self._link_lengths(p)  # 링크 길이는 고정 상수로 사용
        base = p[0].copy()

        # wrist center target (d6≈0이면 EE 위치)
        p_t = target_pose[:3, 3]
        R_t = target_pose[:3, :3]
        target_wc = p_t - kin.d6 * R_t[:, 2] if abs(kin.d6) > 1e-6 else p_t

        # 도달 가능성 체크
        if np.linalg.norm(target_wc - base) > (np.sum(L) + 1e-6):
            return kin.clamp(q), False

        ok = False
        for _ in range(self.max_iter):
            # 현재 q에서 축 추출(이번 iteration 동안 고정)
            axes = self._axes_world(q)

            # q로부터 현재 p 재구축 (누적오차 방지)
            p = self._chain_positions(q)

            # FABRIK-R passes
            self._forward_pass(p, axes, L, target_wc)
            self._backward_pass(p, axes, L, base)

            # q를 p에 맞추어 갱신
            q = self._positions_to_q(q, p)

            # 수렴 검사(워리스트 센터)
            Tee, _ = kin.forward_kinematics(q)
            axes_now = self._axes_world(q)
            wc_now = Tee[:3, 3] - kin.d6 * axes_now[5] if abs(kin.d6) > 1e-9 else Tee[:3, 3]
            if np.linalg.norm(wc_now - target_wc) < self.tol:
                ok = True
                break

        return kin.clamp(q), ok


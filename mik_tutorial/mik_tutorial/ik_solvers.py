import numpy as np
from abc import ABC, abstractmethod
import pinocchio as pin
import os
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from rclpy.logging import get_logger


# ================================================================
#  Kinematic Model (URDF-driven FK/Jacobian)
# ================================================================
class KinematicModel:
    """
    PiPER 6-DoF URDF 로딩:
      - forward_kinematics(q): (4x4 T_ee, [T_joint1..T_joint6])
      - jacobian(q): 6x6 geometric Jacobian at EE
      - joint_names, lower/upper limits (rad)
    또한 EE 프레임에서 joint6 원점까지의 고정 오프셋 r_ee_to_j6_ee(3,) 저장
    (FABRIK에서 '워리스트 타깃'을 정확히 계산하기 위해 사용)
    """
    def __init__(self):
        try:
            desc_share = get_package_share_directory('piper_description')
            urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
        except Exception:
            urdf_file = '/mnt/data/piper_no_gripper_description.urdf'  # fallback

        if not os.path.exists(urdf_file):
            raise FileNotFoundError(f"PiPER URDF not found: {urdf_file}")

        self.urdf_file = urdf_file

        # Pinocchio
        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_file)
        self.model = self.robot.model
        self.data = self.robot.data

        # EE frame
        self.ee_joint_name = "joint6"
        self.ee_frame_id = self.model.getFrameId(self.ee_joint_name)

        # Joint names (6 DoF)
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

        # Limits from URDF
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
            lower.append(lo)
            upper.append(hi)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

        self.logger = get_logger('ik_solvers')

        # --- EE 프레임에서 joint6 원점까지의 고정 오프셋 저장 ---
        try:
            qzero = np.zeros(6, dtype=float)
            self._full_fk(qzero)
            j6_name = self.joint_names[-1]
            self.j6_id = self.model.getJointId(j6_name)

            Tj6 = self.data.oMi[self.j6_id]       # base→joint6
            Tee = self.data.oMf[self.ee_frame_id] # base→EE

            # EE frame에서 joint6 원점까지 벡터 r_ee_to_j6_ee
            r_world = Tj6.translation - Tee.translation     # world
            R_ee    = Tee.rotation
            self.r_ee_to_j6_ee = R_ee.T @ r_world           # (3,)
        except Exception:
            self.j6_id = None
            self.r_ee_to_j6_ee = np.zeros(3)  # 폴백

        # 기본 joint 역할(필요 시 노드에서 override)
        # pivot: 구형관절(회전 3축 역할), hinge: 1축 회전, pris: 직동
        self.joint_roles = ["pivot", "hinge", "hinge", "pivot", "hinge", "pivot"]

    # ---------- Core geometry ----------
    def _full_fk(self, q):
        q = np.asarray(q, dtype=float).flatten()
        if q.size != 6:
            raise ValueError("FK expects 6 joint values (rad).")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def forward_kinematics(self, q):
        self._full_fk(q)
        Ts = []
        for nm in self.joint_names:
            jid = self.model.getJointId(nm)
            Ts.append(self.data.oMi[jid].homogeneous.copy())
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
        return pin.computeFrameJacobian(self.model, self.data, q, fid, ref_frame)

    # ---------- Utilities ----------
    def clamp(self, q):
        return np.minimum(np.maximum(q, self.lower), self.upper)

    def set_joint_roles(self, roles):
        assert len(roles) == 6 and all(r in ("pivot", "hinge", "pris") for r in roles)
        self.joint_roles = list(roles)

    def describe_joint_roles(self):
        return " / ".join(f"J{i+1}:{r}" for i, r in enumerate(getattr(self, "joint_roles", [])))


# ================================================================
#  Base Class
# ================================================================
class IKSolverBase(ABC):
    def __init__(self, kinematics: KinematicModel):
        self.kinematics = kinematics
        self.joint_names = kinematics.joint_names
        # solve() 호출 후 여기 담김(시각화용)
        self.debug = None

    @abstractmethod
    def solve(self, target_pose: np.ndarray, q_seed=None):
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
        err = pin.log(dMi).vector  # [vx,vy,vz, wx,wy,wz]
        return err

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        # ---- debug buffers ----
        ee_positions = []
        pos_errs = []
        ori_errs = []

        lam = self.lmbda
        for it in range(self.max_iter):
            T_c, _ = kin.forward_kinematics(q)
            err6 = self._se3_error_local(T_c, target_pose)
            pe, re = np.linalg.norm(err6[:3]), np.linalg.norm(err6[3:])

            # debug record
            ee_positions.append(T_c[:3, 3].copy())
            pos_errs.append(float(pe))
            ori_errs.append(float(re))

            if pe < self.tol_pos and re < self.tol_rot:
                self.debug = {
                    "iters": it + 1,
                    "ee_positions": ee_positions,
                    "pos_errs": pos_errs,
                    "ori_errs": ori_errs,
                }
                return kin.clamp(q), True

            W = np.diag([self.w_pos] * 3 + [self.w_rot] * 3)
            e = W @ err6
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T
            dq_nom = J.T @ np.linalg.solve(JJt + (lam**2) * np.eye(6), e)

            # backtracking line search
            alpha = self.alpha
            improved = False
            for _ in range(5):
                q_try = kin.clamp(q + alpha * dq_nom)
                T_try, _ = kin.forward_kinematics(q_try)
                e_try = self._se3_error_local(T_try, target_pose)
                if np.linalg.norm(W @ e_try) < np.linalg.norm(e):
                    q = q_try
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lam *= 2.0

        # debug on exit
        self.debug = {
            "iters": self.max_iter,
            "ee_positions": ee_positions,
            "pos_errs": pos_errs,
            "ori_errs": ori_errs,
        }
        return kin.clamp(q), False


# ================================================================
#  FABRIK (Constrained, revolute-only plane projection + limits)
#  -> cfabrik의 알고리즘을 ROS/Pinocchio 체인으로 이식
# ================================================================
class FABRIKSolver(IKSolverBase):
    """
    핵심 아이디어(이식 포인트):
      - 체인 좌표 p0..p6 (joint1..joint6 + wrist_center) 구성
      - Backward(목표에서 거꾸로) → Forward(기준점 고정)로 FABRIK 한 사이클
      - 각 스텝에서 '부모 링크 방향' 변화에 맞춰 힌지 축/평면(u,v)을 동적으로 이송
      - 목표점을 힌지 평면으로 투영하고 (관절한계 있으면 각 φ를 클램프)
      - 새 체인으로부터 q를 복원(축 주위 서명각)
      - EE 자세는 '워리스트 위치 오차'가 작아졌을 때 Jacobian 소정 보정으로 맞춤
    """
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        # tolerances
        self.max_iter = 120
        self.tol_pos = 1e-3
        self.tol_rot = np.deg2rad(1.0)

        # position update controls
        self.q_gain = 0.9
        self.q_reg = 0.02
        self.smooth_q = 0.30
        self.max_step_deg = 6.0

        # gating/orientation tweak
        self.orient_gate_mul = 5.0      # wrist err가 tol_pos*이 값보다 작을 때 자세 보정 on
        self.orient_lambda = 0.05       # 작은 DLS 감쇠로 자세만 보정
        self.orient_steps = 2           # 한 반복에서 적용할 회전 보정 스텝 수

        # neutral(mid-range) as default start
        lo, hi = kinematics.lower, kinematics.upper
        mid = np.where(np.isfinite(lo + hi), 0.5 * (lo + hi), 0.0)
        self.q_mid = np.where(np.isfinite(mid), mid, 0.0)
        self.q0 = self.q_mid.copy()

        # debug
        self.debug = None

    # ---------- small helpers (ported from cfabrik) ----------
    @staticmethod
    def _norm(v):
        return float(np.linalg.norm(v))

    @staticmethod
    def _unit(v, fallback=None):
        vv = np.asarray(v, dtype=float).reshape(-1)
        n = np.linalg.norm(vv)
        if n < 1e-12:
            if fallback is None:
                fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            ff = np.asarray(fallback, dtype=float).reshape(-1)
            fn = np.linalg.norm(ff)
            return ff / (fn if fn > 1e-12 else 1.0)
        return vv / n

    @staticmethod
    def _any_perp(axis):
        axis = FABRIKSolver._unit(axis)
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(axis, ref)
        return FABRIKSolver._unit(perp)

    @staticmethod
    def _rot_between(u0, u):
        a = FABRIKSolver._unit(u0).astype(float)
        b = FABRIKSolver._unit(u).astype(float)
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = float(np.dot(a, b))
        if s < 1e-12:
            if c > 0:
                return np.eye(3)
            axis = FABRIKSolver._any_perp(a)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=float)
            return np.eye(3) + 2 * K @ K
        axis = v / s
        theta = np.arctan2(s, c)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=float)
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # ---------- chain & axis extraction ----------
    def _frame_arrays(self, q):
        """현재 q에서 p[0..6], link_lengths[0..5], axes_world[0..5], parent_dir0[0..6] 생성"""
        kin = self.kinematics
        Tee, Ts = kin.forward_kinematics(q)

        # joint origins
        p = [T[:3, 3].copy() for T in Ts]  # p[0..5]
        # wrist center = joint6 origin (정확)
        p_wc = Ts[-1][:3, 3].copy()
        p.append(p_wc)  # p[6]

        # link lengths (고정)
        L = [float(np.linalg.norm(p[i + 1] - p[i])) for i in range(6)]

        # world joint axes via Jacobian rotational part
        Jw = kin.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
        axes = []
        for j in range(6):
            w = np.array(Jw[3:, j]).reshape(3,)
            if np.linalg.norm(w) < 1e-10:
                axes.append(None)
            else:
                axes.append(self._unit(w))

        # neutral parent directions (현재 q를 기준으로)
        parent_dir0 = []
        # base(=p[0]의 부모)를 p[-1]로 쓰지 말고, p[0]-base를 근사: 첫 링크 방향
        d0 = p[0] - (p[0] - self._unit(p[1] - p[0]) * L[0])
        parent_dir0.append(self._unit(p[0] - d0))
        for i in range(6):
            parent_dir0.append(self._unit(p[i + 1] - p[i]))

        # hinge plane bases u0,v0 (부모→자식 기준)
        hinge_u0, hinge_v0 = [], []
        for j in range(6):
            ax = axes[j] if axes[j] is not None else self._any_perp(p[j + 1] - p[j])
            d = p[j + 1] - p[j]
            d_plane = d - np.dot(d, ax) * ax
            if self._norm(d_plane) < 1e-12:
                u0 = self._any_perp(ax)
            else:
                u0 = self._unit(d_plane)
            v0 = self._unit(np.cross(ax, u0))
            hinge_u0.append(u0)
            hinge_v0.append(v0)

        return p, L, axes, hinge_u0, hinge_v0, parent_dir0, Tee

    def _transport_plane_basis(self, j, parent_dir0_j, parent_dir_now_j, axis_world_j, u0_j, v0_j):
        """부모 링크 방향 변화(parent_dir0→now)에 맞춰 (axis,u0,v0) 회전 이송."""
        if axis_world_j is None:
            # 직동/알 수 없음: u,v는 그냥 현재 방향으로 재정의
            ax_now = self._unit(parent_dir_now_j)
            u_now = self._any_perp(ax_now)
            v_now = self._unit(np.cross(ax_now, u_now))
            return u_now, v_now, ax_now
        R = self._rot_between(parent_dir0_j, parent_dir_now_j)
        ax_now = self._unit(R @ axis_world_j)
        u_now = self._unit(R @ u0_j)
        v_now = self._unit(np.cross(ax_now, u_now))
        return u_now, v_now, ax_now

    # ---------- FABRIK primitives ----------
    def _backward_global(self, p_in, L, target):
        out = [pi.copy() for pi in p_in]
        out[-1] = target.copy()
        for k in range(len(out) - 2, -1, -1):
            v = out[k] - out[k + 1]
            dir_v = self._unit(v)
            out[k] = out[k + 1] + L[k] * dir_v
        return out

    def _fabrik_subchain(self, p_in, L, anchor_idx, target):
        out = [pi.copy() for pi in p_in]
        out[-1] = target.copy()
        for k in range(len(out) - 2, anchor_idx, -1):
            v = out[k] - out[k + 1]
            dir_v = self._unit(v)
            out[k] = out[k + 1] + L[k] * dir_v
        for k in range(anchor_idx, len(out) - 1):
            v = out[k + 1] - out[k]
            dir_v = self._unit(v)
            out[k + 1] = out[k] + L[k] * dir_v
        return out

    # ---------- q reconstruction ----------
    @staticmethod
    def _signed_angle_around_axis(v_from, v_to, axis):
        a = axis / (np.linalg.norm(axis) + 1e-12)
        vf = v_from - a * np.dot(a, v_from)
        vt = v_to - a * np.dot(a, v_to)
        nf = np.linalg.norm(vf)
        nt = np.linalg.norm(vt)
        if nf < 1e-12 or nt < 1e-12:
            return 0.0
        vf /= nf
        vt /= nt
        s = np.dot(a, np.cross(vf, vt))
        c = np.clip(np.dot(vf, vt), -1.0, 1.0)
        return float(np.arctan2(s, c))

    def _positions_to_q(self, q_in, p_des, gain=None):
        """현재 q에서의 프레임들과 p_des 체인을 비교해 축 주위 각도 변화량으로 q 갱신"""
        kin = self.kinematics
        roles = getattr(kin, "joint_roles", ["hinge"] * 6)
        q = q_in.copy()

        Tee, Ts = kin.forward_kinematics(q)
        # world rotational axes
        pin.computeJointJacobians(kin.model, kin.data, q)
        pin.updateFramePlacements(kin.model, kin.data)
        fid = kin.ee_frame_id
        Jw = pin.computeFrameJacobian(kin.model, kin.data, q, fid, pin.ReferenceFrame.WORLD)

        wc_cur = Ts[-1][:3, 3].copy()
        wc_des = p_des[6]
        g = self.q_gain if gain is None else float(gain)

        for i in range(6):
            a = Jw[3:, i].reshape(3,)
            if np.linalg.norm(a) < 1e-10:
                continue  # prismatic skip
            a = a / np.linalg.norm(a)
            pj = Ts[i][:3, 3]
            if roles[i] == "pivot":
                r_cur = wc_cur - pj
                r_des = wc_des - p_des[i]
                r_cur -= a * np.dot(a, r_cur)
                r_des -= a * np.dot(a, r_des)
                if np.linalg.norm(r_cur) < 1e-12 or np.linalg.norm(r_des) < 1e-12:
                    v_cur = (Ts[i + 1][:3, 3] if i < 5 else wc_cur) - pj
                    v_des = p_des[i + 1] - p_des[i]
                    dtheta = self._signed_angle_around_axis(v_cur, v_des, a)
                else:
                    dtheta = self._signed_angle_around_axis(r_cur, r_des, a)
            else:  # hinge
                v_cur = (Ts[i + 1][:3, 3] if i < 5 else wc_cur) - pj
                v_des = p_des[i + 1] - p_des[i]
                if np.linalg.norm(v_des) < 1e-12 or np.linalg.norm(v_cur) < 1e-12:
                    continue
                dtheta = self._signed_angle_around_axis(v_cur, v_des, a)

            q[i] = np.clip(q[i] + g * float(dtheta), kin.lower[i], kin.upper[i])

        # weak regularization toward mid-range
        if self.q_reg > 0.0:
            q = kin.clamp((1.0 - self.q_reg) * q + self.q_reg * self.q_mid)
        return q

    # ---------- main ----------
    def solve(self, target_pose: np.ndarray, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        # chain @ current q
        p, L, axes0, u0_list, v0_list, parent_dir0, Tee = self._frame_arrays(q)
        base_p = p[0].copy()
        base_R = np.eye(3)  # base 회전은 고정(필요시 확장)

        # EE target
        p_t = target_pose[:3, 3].copy()
        R_t = target_pose[:3, :3].copy()

        # --- 타깃 워리스트: EE 타깃에서 joint6 오프셋(EE frame)을 회전/이동 ---
        r_ee_to_j6 = getattr(kin, "r_ee_to_j6_ee", np.zeros(3))
        target_wc = p_t + R_t @ r_ee_to_j6

        # reachability (워리스트 타깃 기준)
        if np.linalg.norm(target_wc - base_p) > (np.sum(L) + 1e-6):
            self.debug = None
            return kin.clamp(q), False

        max_step = np.deg2rad(self.max_step_deg)
        local_gain = self.q_gain

        # ---- debug buffers ----
        ee_positions, pos_errs, ori_errs, wrist_errs, gates = [], [], [], [], []

        # helper to compute EE & wrist error
        def eval_all(qv, p_list=None):
            Tee_, Ts_ = kin.forward_kinematics(qv)
            dp = float(np.linalg.norm(Tee_[:3, 3] - p_t))
            M = Tee_[:3, :3] @ R_t.T
            tr = np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)
            ang = float(np.arccos(tr))
            if p_list is None:
                wc = Ts_[-1][:3, 3]
            else:
                wc = p_list[6]
            werr = float(np.linalg.norm(wc - target_wc))
            return Tee_[:3, 3].copy(), dp, ang, (dp + ang), werr

        # initial record
        ee_pos, pe, oe, cost_prev, werr = eval_all(q, p)
        ee_positions.append(ee_pos)
        pos_errs.append(pe)
        ori_errs.append(oe)
        wrist_errs.append(werr)
        gates.append(False)

        ok = False
        for it in range(self.max_iter):
            # ---------- Backward (global) ----------
            p_b = self._backward_global(p, L, target_wc)

            # ---------- Forward with constraints ----------
            cur = [pi.copy() for pi in p_b]
            cur[0] = base_p.copy()  # anchor base
            # 동적 축 이송 + 힌지 평면 투영
            for j in range(1, 7):
                parent = cur[j - 1]
                ideal = p_b[j]
                # 현재 부모 링크 방향(now) 산출
                grandparent = base_p if (j - 1) == 0 else cur[j - 2]
                parent_dir_now = parent - grandparent
                parent_dir_now = self._unit(parent_dir_now, parent_dir0[j - 1])

                u_now, v_now, ax_now = self._transport_plane_basis(
                    j - 1, parent_dir0[j - 1], parent_dir_now, axes0[j - 1], u0_list[j - 1], v0_list[j - 1]
                )

                # 힌지 평면으로 투영(관절 한계는 q에서 클램프하므로 여기선 평면 투영만)
                v = ideal - parent
                v_plane = v - np.dot(v, ax_now) * ax_now
                if self._norm(v_plane) < 1e-12:
                    v_plane = u_now
                dir_on_plane = self._unit(v_plane)

                cur[j] = parent + L[j - 1] * dir_on_plane

                # 서브체인 정련
                if j < 6:
                    cur = self._fabrik_subchain(cur, L, j, target_wc)

            # 체인 블렌딩(살짝 이완)
            alpha_relax = 0.30
            p_prev = [pi.copy() for pi in p]
            p = [(1.0 - alpha_relax) * p_prev[i] + alpha_relax * cur[i] for i in range(7)]

            # ---------- q update from chain ----------
            q_prev = q.copy()
            q_prop = self._positions_to_q(q_prev, p, gain=local_gain)

            # step limit & smoothing
            dq = q_prop - q_prev
            dq = np.clip(dq, -max_step, max_step)
            q_step = kin.clamp(q_prev + dq)
            if self.smooth_q > 0.0:
                q = kin.clamp((1.0 - self.smooth_q) * q_step + self.smooth_q * q_prev)
            else:
                q = q_step

            # ---------- optional: orientation-only DLS tweak (gate) ----------
            ee_pos, pe, oe, cost_now, werr = eval_all(q)
            fix_orientation = (werr < self.orient_gate_mul * self.tol_pos)
            if fix_orientation and oe > self.tol_rot * 0.5:
                for _ in range(self.orient_steps):
                    Tee_now, _ = kin.forward_kinematics(q)
                    dMi = pin.SE3(Tee_now).actInv(pin.SE3(target_pose))
                    err6 = pin.log(dMi).vector
                    e_rot = err6[3:]  # only rotation
                    J_loc = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
                    Jr = J_loc[3:, :]
                    JJt = Jr @ Jr.T
                    lam = self.orient_lambda
                    dq_or = Jr.T @ np.linalg.solve(JJt + (lam**2) * np.eye(3), e_rot)
                    dq_or = np.clip(dq_or, -max_step, max_step)
                    q = kin.clamp(q + dq_or)
                ee_pos, pe, oe, cost_now, werr = eval_all(q)

            # 적응적 gain
            if cost_now > cost_prev:
                # 후퇴
                q = kin.clamp(0.5 * (q_prev + q))
                ee_pos, pe, oe, cost_now, werr = eval_all(q)
                local_gain = max(0.4 * local_gain, 0.15 * self.q_gain)
            else:
                local_gain = min(max(local_gain, 0.5 * self.q_gain) * 1.05, self.q_gain)

            # debug rec
            ee_positions.append(ee_pos)
            pos_errs.append(pe)
            ori_errs.append(oe)
            wrist_errs.append(werr)
            gates.append(bool(fix_orientation))

            # convergence
            if pe < self.tol_pos and oe < self.tol_rot:
                ok = True
                break

            cost_prev = cost_now
            # 새 축/기저는 다음 반복의 기준 q에서 다시 산출
            p, L, axes0, u0_list, v0_list, parent_dir0, Tee = self._frame_arrays(q)
            p[-1] = target_wc  # wrist 기준점은 현재 목표 유지

        # save debug (마지막 체인 좌표도 저장)
        self.debug = {
            "iters": len(ee_positions),
            "ee_positions": ee_positions,
            "pos_errs": pos_errs,
            "ori_errs": ori_errs,
            "wrist_errs": wrist_errs,
            "gates": gates,                   # True면 orientation 보정 on
            "final_chain_p": p,               # [p0..p6]
        }
        return kin.clamp(q), ok

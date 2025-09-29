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
    - URDF 로딩 및 Pinocchio 모델 구성
    - forward_kinematics(q) → (T_ee, [T_joint1..T_joint6])
    - jacobian(q)
    - joint axis / type 파싱 (revolute/continuous/prismatic, axis xyz)
    - EE→joint6 오프셋 r_ee_to_j6_ee 계산
    - chain_points(q) → p0..p6 월드 좌표
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

        # EE frame (joint6를 EE로 취급)
        self.ee_joint_name = "joint6"
        try:
            self.ee_frame_id = self.model.getFrameId(self.ee_joint_name)
        except Exception:
            self.ee_frame_id = self.model.nframes - 1

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

        # --- URDF 파싱: limits / axis / type ---
        lower, upper = [], []
        axis_map_local = {}
        type_map = {}

        root = ET.parse(urdf_file).getroot()
        lim_map = {}
        for j in root.findall('joint'):
            nm = j.attrib.get('name', '')
            jtype = j.attrib.get('type', '')
            type_map[nm] = jtype

            # limits
            lim = j.find('limit')
            if lim is not None and 'lower' in lim.attrib and 'upper' in lim.attrib:
                lim_map[nm] = (float(lim.attrib['lower']), float(lim.attrib['upper']))
            elif jtype == 'continuous':
                lim_map[nm] = (-np.inf, np.inf)

            # axis
            ax = j.find('axis')
            if ax is not None and 'xyz' in ax.attrib:
                xyz = np.fromstring(ax.attrib['xyz'], sep=' ', dtype=float)
                if np.linalg.norm(xyz) < 1e-12:
                    xyz = np.array([0.0, 0.0, 1.0])
                axis_map_local[nm] = xyz / np.linalg.norm(xyz)

        for nm in self.joint_names:
            lo, hi = lim_map.get(nm, (-np.inf, np.inf))
            lower.append(lo); upper.append(hi)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.joint_axis_local = axis_map_local  # local frame axis
        self.joint_type = type_map              # 'revolute'/'continuous'/'prismatic'/...

        self.logger = get_logger('ik_solvers')

        # --- EE frame에서 joint6 원점까지의 고정 오프셋 ---
        try:
            qzero = np.zeros(6, dtype=float)
            self._full_fk(qzero)
            j6_name = self.joint_names[-1]
            self.j6_id = self.model.getJointId(j6_name)

            Tj6 = self.data.oMi[self.j6_id]       # base→joint6
            Tee = self.data.oMf[self.ee_frame_id] # base→EE

            r_world = Tj6.translation - Tee.translation     # world
            R_ee    = Tee.rotation
            self.r_ee_to_j6_ee = R_ee.T @ r_world           # (3,)
        except Exception:
            self.j6_id = None
            self.r_ee_to_j6_ee = np.zeros(3)

    # ---------- Core geometry ----------
    def _full_fk(self, q):
        q = np.asarray(q, dtype=float).flatten()
        if q.size != 6:
            raise ValueError("FK expects 6 joint values (rad or m).")
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

    def joint_axis_world(self, q, joint_name):
        """URDF axis(xyz)을 joint 로컬→월드로 회전시켜 월드 기준 축 반환."""
        self._full_fk(q)
        jid = self.model.getJointId(joint_name)
        Rw = self.data.oMi[jid].rotation
        a_local = self.joint_axis_local.get(joint_name, np.array([0.0, 0.0, 1.0]))
        a_world = Rw @ a_local
        n = np.linalg.norm(a_world)
        return a_world if n < 1e-12 else a_world / n

    def chain_points(self, q):
        """p0..p6 월드 좌표 (필요 시 사용)."""
        self._full_fk(q)
        pts = [np.zeros(3, dtype=float)]  # p0
        for nm in self.joint_names:
            jid = self.model.getJointId(nm)
            pts.append(self.data.oMi[jid].translation.copy())
        return np.asarray(pts)  # (7,3)


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
        Returns:
            q (np.ndarray): solution joint vector
            ok (bool): success flag
            info (dict): at least {'iters_total': int} 등
        """
        pass


# ================================================================
#  Jacobian IK (DLS)
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

        iters = 0
        lam = self.lmbda
        for it in range(self.max_iter):
            iters = it + 1
            T_c, _ = kin.forward_kinematics(q)
            err6 = self._se3_error_local(T_c, target_pose)
            pe, re = np.linalg.norm(err6[:3]), np.linalg.norm(err6[3:])

            if pe < self.tol_pos and re < self.tol_rot:
                return kin.clamp(q), True, {"iters_total": iters}

            W = np.diag([self.w_pos]*3 + [self.w_rot]*3)
            e = W @ err6
            J = kin.jacobian(q, ref_frame=pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T
            dq_nom = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), e)

            alpha = self.alpha
            improved = False
            for _ in range(5):
                q_try = kin.clamp(q + alpha*dq_nom)
                T_try, _ = kin.forward_kinematics(q_try)
                e_try = self._se3_error_local(T_try, target_pose)
                if np.linalg.norm(W @ e_try) < np.linalg.norm(e):
                    q = q_try; improved = True; break
                alpha *= 0.5
            if not improved:
                lam *= 2.0

        return kin.clamp(q), False, {"iters_total": iters}


# ================================================================
#  FABRIK (URDF + Pinocchio, p0/p1 앵커링, 각도/이동만 갱신)
# ================================================================
class FABRIKSolver(IKSolverBase):
    """
    - p0: 월드 원점 (고정)
    - p1: joint1 원점 (앵커)
    - p2..p6: FABRIK-R으로 위치만 업데이트
    - 그 결과를 축기반 각도/슬라이드로만 반영 (orientation 제약 없음)
    """
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.max_iter_fabrik = 80
        self.tol_fabrik = 1e-3
        self.q0 = np.deg2rad(np.array([55.0, 0.0, 205.0, 0.0, 85.0, 0.0], dtype=float))
        self.align_passes = 3
        self.tol_align = 2e-3

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        return v if n < 1e-12 else v / n

    @staticmethod
    def _fabrik_place(prev, curr, length):
        d = np.linalg.norm(curr - prev)
        if d < 1e-12:
            return prev.copy()
        return prev + (curr - prev) * (length / d)

    def _project_point_to_plane(self, p, n, p0):
        n = self._normalize(n)
        return p - np.dot(p - p0, n) * n

    def _make_vertical_plane_through(self, base_point, through_point):
        z = np.array([0.0, 0.0, 1.0])
        v = through_point - base_point
        v_xy = np.array([v[0], v[1], 0.0])
        if np.linalg.norm(v_xy) < 1e-9:
            normal = np.array([1.0, 0.0, 0.0])
        else:
            normal = np.cross(z, v_xy)
        return self._normalize(normal), base_point.copy()

    def _points_from_q(self, q):
        return self.kinematics.chain_points(q)

    def _fabrik_run(self, points, lengths, target):
        pts = points.copy()
        p0 = pts[0].copy()
        p1_anchor = pts[1].copy()

        # ✅ 이미 타깃과 거의 같으면 반복 0회로 바로 반환
        if np.linalg.norm(pts[6] - target) < self.tol_fabrik:
            return pts, 0

        phi_normal, phi_point = self._make_vertical_plane_through(p0, pts[5])
        fabrik_iters = 0

        for _ in range(self.max_iter_fabrik):
            fabrik_iters += 1
            # 고정
            pts[0] = p0
            pts[1] = p1_anchor

            # Forward
            pts[6] = target.copy()
            pts[5] = self._fabrik_place(pts[6], pts[5], lengths[5])

            omega_n, omega_p = self._make_vertical_plane_through(p0, pts[5])
            for i in [4, 3, 2]:
                proj = self._project_point_to_plane(pts[i], omega_n, omega_p)
                pts[i] = self._fabrik_place(pts[i+1], proj, lengths[i])

            # Backward
            pts[0] = p0
            pts[1] = p1_anchor
            phi_normal, phi_point = self._make_vertical_plane_through(p0, pts[5])
            for i in [1, 2, 3, 4, 5]:
                prev = pts[i]
                nxt = pts[i+1]
                if i+1 in (2, 3, 4):
                    nxt = self._project_point_to_plane(nxt, phi_normal, phi_point)
                pts[i+1] = self._fabrik_place(prev, nxt, lengths[i])

            if np.linalg.norm(pts[6] - target) < self.tol_fabrik:
                break

        return pts, fabrik_iters

    def _align_q_with_points(self, q_init, p_targets):
        """
        FABRIK 위치해(p_targets: p0..p6)를 실제 q로 반영.
        - 실제로 의미 있게 바뀔 때만 카운트/업데이트
        - 한 패스에서 아무 변화도 없으면 즉시 종료
        """
        kin = self.kinematics
        q = kin.clamp(q_init.copy())
        eps = 1e-10
        align_steps = 0
        angle_tol = 1e-6    # [rad]
        len_tol   = 1e-6    # [m]

        def pick_best_ref_index(i, a_world, pts):
            base = pts[i]
            best_k, best_norm = i+1, -1.0
            for k in range(i+1, 7):
                v = pts[k] - base
                v_perp = v - a_world * np.dot(a_world, v)
                n = np.linalg.norm(v_perp)
                if n > best_norm:
                    best_norm = n
                    best_k = k
            return best_k, best_norm

        for _ in range(self.align_passes):
            kin._full_fk(q)
            pass_changed = False

            for i in range(1, 6):
                nm = kin.joint_names[i-1]
                jtype = kin.joint_type.get(nm, 'revolute')

                a = kin.joint_axis_world(q, nm)
                a /= (np.linalg.norm(a) + 1e-15)

                k_ref, spread = pick_best_ref_index(i, a, p_targets)
                if spread < 1e-6:
                    continue

                pts_cur = self._points_from_q(q)
                p_i_cur, p_k_cur = pts_cur[i], pts_cur[k_ref]
                r_cur = p_k_cur - p_i_cur
                r_tgt = p_targets[k_ref] - p_targets[i]

                if jtype in ('revolute', 'continuous'):
                    r_perp = r_cur - a * np.dot(a, r_cur)
                    t_perp = r_tgt - a * np.dot(a, r_tgt)
                    nr = np.linalg.norm(r_perp); nt = np.linalg.norm(t_perp)
                    if nr < eps or nt < eps:
                        continue
                    cross_val = np.cross(r_perp, t_perp)
                    theta = np.arctan2(np.dot(a, cross_val), np.dot(r_perp, t_perp))
                    # ✅ 변화가 의미 있을 때만 적용/카운트
                    if abs(theta) > angle_tol:
                        q[i-1] = np.clip(q[i-1] + theta, kin.lower[i-1], kin.upper[i-1])
                        pass_changed = True
                        align_steps += 1

                elif jtype == 'prismatic':
                    p_ip1_cur = pts_cur[i+1]
                    r_cur_seg = p_ip1_cur - p_i_cur
                    r_tgt_seg = p_targets[i+1] - p_targets[i]
                    delta = np.dot(r_tgt_seg, a) - np.dot(r_cur_seg, a)
                    if abs(delta) > len_tol:
                        q[i-1] = np.clip(q[i-1] + delta, kin.lower[i-1], kin.upper[i-1])
                        pass_changed = True
                        align_steps += 1

                if pass_changed:
                    kin._full_fk(q)  # 다음 조인트 계산에 최신 상태 반영

            # ✅ 이 패스에서 아무 변화가 없으면 조기 종료
            if not pass_changed:
                break

        return kin.clamp(q), align_steps

    def solve(self, target_pose: np.ndarray, q_seed=None):
        kin = self.kinematics

        # 초기 상태
        q0 = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        pts0 = self._points_from_q(q0)
        lengths = np.linalg.norm(pts0[1:] - pts0[:-1], axis=1)

        # EE → joint6(워리스트) 타깃 변환
        Tt = np.asarray(target_pose, dtype=float)
        p_ee = Tt[:3, 3]; R_ee = Tt[:3, :3]
        r_ee_to_j6 = kin.r_ee_to_j6_ee if kin.r_ee_to_j6_ee is not None else np.zeros(3)
        p6_target = p_ee + R_ee @ r_ee_to_j6

        # FABRIK 위치 해 (필요시 0회)
        sol_pts, fabrik_iters = self._fabrik_run(pts0, lengths, p6_target)

        # 각도/이동만 반영 (실제 변화만 카운트)
        q_final, align_steps = self._align_q_with_points(q0, sol_pts)

        # 성공 판정(포지션 전용)
        pts_final = self._points_from_q(q_final)
        pos_err_chain = np.linalg.norm(pts_final[2:7] - sol_pts[2:7], axis=1).max()
        fabrik_ok = (np.linalg.norm(sol_pts[6] - p6_target) < self.tol_fabrik)
        align_ok = (pos_err_chain < self.tol_align)
        ok = bool(fabrik_ok and align_ok)

        total_iters = fabrik_iters + align_steps
        if total_iters == 0:
            total_iters = 1  # Jacobian과 동일 관례(즉시 수렴 시 1)

        return kin.clamp(q_final), ok, {
            "iters_total": int(total_iters),
            "iters_fabrik": int(fabrik_iters),
            "iters_align": int(align_steps),
        }

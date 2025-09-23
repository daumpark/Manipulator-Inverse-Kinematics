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
      - jacobian(q): 6x6 geometric Jacobian at EE (LOCAL frame)
      - joint_names, lower/upper limits (rad)
    Also estimates 'd6' (distance from joint6 origin to EE along joint6 z-axis)
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

        # Estimate d6 at zero configuration (project EE offset onto joint6 z)
        try:
            qzero = np.zeros(6, dtype=float)
            self._full_fk(qzero)
            j6_name = self.joint_names[-1]
            j6_id = self.model.getJointId(j6_name)
            Tj6 = self.data.oMi[j6_id]
            Tee = self.data.oMf[self.ee_frame_id]
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

# ================================================================
#  FABRIK (Pure, position-only, with revolute-axis constraints)
#     - Uses wrist-center decoupling via estimated d6
#     - Maps FABRIK-updated segment directions to joint angles by
#       rotating about each joint's local z-axis (assumes URDF axes).
#     - No Jacobian/DLS involved.
# ================================================================
class FabrikPureIKSolver(IKSolverBase):
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.max_iter = 60
        self.tol = 1e-3
        # seed near mid-range
        lo, hi = kinematics.lower, kinematics.upper
        mid = np.where(np.isfinite(lo+hi), 0.5*(lo+hi), 0.0)
        self.q0 = np.where(np.isfinite(mid), mid, 0.0)

    # --- helpers ---
    def _chain_positions(self, q):
        """
        Returns p[0..6] world positions: base (0) then joint1..joint6 (6)
        """
        _, Ts = self.kinematics.forward_kinematics(q)
        ps = [np.zeros(3, dtype=float)]
        for T in Ts:
            ps.append(T[:3, 3].copy())
        return ps

    def _link_lengths(self, ps):
        return [float(np.linalg.norm(ps[i+1]-ps[i])) for i in range(6)]

    @staticmethod
    def _signed_angle_around_axis(v_from, v_to, axis):
        """Return signed angle to rotate v_from -> v_to around 'axis' (radians)"""
        # project onto plane perpendicular to axis
        a = axis / (np.linalg.norm(axis) + 1e-12)
        vf = v_from - a * np.dot(a, v_from)
        vt = v_to   - a * np.dot(a, v_to)
        n = np.linalg.norm(vf) * np.linalg.norm(vt)
        if n < 1e-12:
            return 0.0
        vf = vf / (np.linalg.norm(vf) + 1e-12)
        vt = vt / (np.linalg.norm(vt) + 1e-12)
        s = np.dot(a, np.cross(vf, vt))
        c = np.clip(np.dot(vf, vt), -1.0, 1.0)
        return float(np.arctan2(s, c))

    def _apply_fabrik_pass(self, p, L, target, base):
        """One forward/backward FABRIK pass on positions (p is list length 7)."""
        # forward: set end to target, pull back
        p[6] = target.copy()
        for i in range(5, -1, -1):
            r = np.linalg.norm(p[i+1] - p[i])
            if r < 1e-12:
                continue
            lam = L[i] / r
            p[i] = (1 - lam) * p[i+1] + lam * p[i]
        # backward: set base fixed, push forward
        p[0] = base.copy()
        for i in range(0, 6):
            r = np.linalg.norm(p[i+1] - p[i])
            if r < 1e-12:
                continue
            lam = L[i] / r
            p[i+1] = (1 - lam) * p[i] + lam * p[i+1]

    def _positions_to_q(self, q, p_des):
        """
        Given desired segment directions from p_des, rotate each joint about its
        local z-axis to align current segment toward desired, updating q in-place.
        """
        kin = self.kinematics
        # iterate joints sequentially with FK refresh to propagate downstream
        for i in range(6):
            Tee, Ts = kin.forward_kinematics(q)
            # current vectors and axes
            pj = Ts[i][:3, 3]
            aj = Ts[i][:3, :3] @ np.array([0.0, 0.0, 1.0])
            if i < 5:
                pj_next = Ts[i+1][:3, 3]
            else:
                pj_next = Tee[:3, 3]  # end-effector position
            v_cur = pj_next - pj
            v_des = p_des[i+1] - p_des[i]
            if np.linalg.norm(v_des) < 1e-12 or np.linalg.norm(v_cur) < 1e-12:
                continue
            dtheta = self._signed_angle_around_axis(v_cur, v_des, aj)
            q[i] = np.clip(q[i] + dtheta, kin.lower[i], kin.upper[i])
        return q

    def solve(self, target_pose: np.ndarray, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        # Current chain geometry
        ps = self._chain_positions(q)
        L = self._link_lengths(ps)

        # Reachability test for position
        base = ps[0].copy()
        R_t = target_pose[:3, :3]
        p_t = target_pose[:3, 3]
        # wrist-center if d6 available
        p_goal = p_t - kin.d6 * R_t[:, 2] if abs(kin.d6) > 1e-6 else p_t
        if np.linalg.norm(p_goal - base) > (sum(L) + 1e-6):
            return kin.clamp(q), False

        # Iterations
        for _ in range(self.max_iter):
            # build fresh positions from current q
            ps = self._chain_positions(q)
            p = [pi.copy() for pi in ps]

            # FABRIK forward/backward on positions for wrist-center
            self._apply_fabrik_pass(p, L, p_goal, base)

            # Convert desired positions to joint updates (axis-constrained)
            q = kin.clamp(self._positions_to_q(q, p))

            # check convergence on EE position
            Tee, _ = kin.forward_kinematics(q)
            pe = np.linalg.norm(Tee[:3, 3] - p_t)
            if pe < self.tol:
                return kin.clamp(q), True

        return kin.clamp(q), False

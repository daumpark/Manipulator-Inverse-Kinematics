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
    Loads PiPER 6-DoF URDF and exposes:
      - forward_kinematics(q): (4x4 T_ee, [T_joint1..T_joint6])
      - jacobian(q): 6x6 geometric Jacobian at EE
      - joint_names, lower/upper limits (rad)
    Also estimates 'd6' (distance from joint6 origin to EE along joint6 axis)
    to support wrist-center decoupling.
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

        # Build Pinocchio model
        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_file)
        self.model = self.robot.model
        self.data = self.robot.data

        # EE frame: prefer 'link6', else last frame
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

        # Joint names (first 6 one-DoF joints)
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

        # Limits
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

        # Estimate d6
        try:
            qzero = np.zeros(6, dtype=float)
            self._full_fk(qzero)
            j6_name = self.joint_names[-1]
            j6_id = self.model.getJointId(j6_name)
            Tj6 = self.data.oMi[j6_id]
            Tee = self.data.oMf[self.ee_frame_id]
            z = Tj6.rotation[:, 2]  # joint6 local z in world
            off = Tee.translation - Tj6.translation
            self.d6 = float(np.dot(z, off))
        except Exception:
            self.d6 = 0.0

        # Default joint roles (can be overridden)
        self.joint_roles = ["pivot","hinge","hinge","pivot","hinge","pivot"]

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
        assert len(roles) == 6 and all(r in ("pivot","hinge","pris") for r in roles)
        self.joint_roles = list(roles)

    def describe_joint_roles(self):
        return " / ".join(f"J{i+1}:{r}" for i,r in enumerate(getattr(self, "joint_roles", [])))


# ================================================================
#  Base Class
# ================================================================
class IKSolverBase(ABC):
    def __init__(self, kinematics: KinematicModel):
        self.kinematics = kinematics
        self.joint_names = kinematics.joint_names

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
        err = pin.log(dMi).vector  # [vx,vy,vz, wx,wy,wz], where ||w|| = angle
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
#  FABRIK (adapted for serial robots: hinge/pivot/pris)
# ================================================================
class FABRIKSolver(IKSolverBase):
    """
    FABRIK (adapted for serial robots: hinge/pivot/prismatic) with robust damping.
    - Orientation gating: far → position first, near → pose matching
    - Position relax (blend), per-joint step cap, q smoothing, cost backtracking
    """
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        # --- tolerances ---
        self.max_iter = 120
        self.tol_pos = 1e-3
        self.tol_rot = np.deg2rad(1.0)

        # --- update controls ---
        self.q_gain = 0.9         # base joint update gain
        self.q_reg  = 0.02        # weak regularization to mid-range
        self.smooth_q = 0.30      # EMA smoothing of q per iteration (0~1)
        self.relax_pos = 0.30     # blend new chain positions p with previous (0~1)
        self.max_step_deg = 6.0   # per-joint per-iter cap (deg)
        self.orient_gate_mul = 5.0  # start enforcing orientation when pos_err < mul*tol_pos

        self.axis_parallel_tol = 0.9
        lo, hi = kinematics.lower, kinematics.upper
        mid = np.where(np.isfinite(lo + hi), 0.5 * (lo + hi), 0.0)
        self.q_mid = np.where(np.isfinite(mid), mid, 0.0)
        self.q0 = self.q_mid.copy()
        self.estimate_ee_orientation_when_pos_only = True

    # ---------- utils ----------
    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    @staticmethod
    def _any_perp(n):
        n = FABRIKSolver._unit(n)
        c = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
        v = np.cross(n, c)
        vn = np.linalg.norm(v)
        return v / (vn + 1e-12)

    def _frame_arrays(self, q):
        kin = self.kinematics
        Tee, Ts = kin.forward_kinematics(q)

        p = [T[:3, 3].copy() for T in Ts]  # p[0..5]
        R = [T[:3, :3].copy() for T in Ts]

        # wrist center
        if abs(kin.d6) > 1e-9:
            Jw = kin.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
            a6 = np.array(Jw[3:, 5]).reshape(3,)
            a6 = self._unit(a6) if np.linalg.norm(a6) > 1e-12 else R[-1][:, 2]
            p_wc = Tee[:3, 3] - kin.d6 * a6
        else:
            p_wc = Tee[:3, 3].copy()
        p.append(p_wc)  # p[6]

        # world axes & link lengths
        Jw = kin.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
        axes = []
        is_pris = []
        for j in range(6):
            w = np.array(Jw[3:, j]).reshape(3,)
            if np.linalg.norm(w) < 1e-10:
                is_pris.append(True)
                z_link = self._unit(p[j+1] - p[j])
                axes.append(self._any_perp(z_link))
            else:
                is_pris.append(False)
                axes.append(self._unit(w))

        L = [float(np.linalg.norm(p[i+1] - p[i])) for i in range(6)]

        x = []; y = []; z = []
        for i in range(6):
            x_i = self._unit(axes[i])
            z_link = self._unit(p[i+1] - p[i]) if np.linalg.norm(p[i+1] - p[i]) > 1e-12 else R[i][:, 2]
            z_i = self._unit(z_link - x_i * np.dot(x_i, z_link))
            if np.linalg.norm(z_i) < 1e-9:
                z_i = self._any_perp(x_i)
            y_i = self._unit(np.cross(z_i, x_i))
            x.append(x_i); y.append(y_i); z.append(z_i)

        R_ee = Tee[:3, :3]
        x.append(R_ee[:, 0].copy()); y.append(R_ee[:, 1].copy()); z.append(R_ee[:, 2].copy())
        return p, x, y, z, L, np.array(axes), np.array(is_pris)

    def _axes_world(self, q):
        kin = self.kinematics
        Jw = kin.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
        A = []
        for j in range(6):
            w = np.array(Jw[3:, j]).reshape(3,)
            if np.linalg.norm(w) < 1e-10:
                A.append(np.zeros(3))
            else:
                A.append(w / np.linalg.norm(w))
        return np.vstack(A)

    def _wrist_center(self, q, Tee=None, axes_world=None):
        kin = self.kinematics
        if Tee is None:
            Tee, _ = kin.forward_kinematics(q)
        if abs(getattr(kin, "d6", 0.0)) < 1e-9:
            return Tee[:3, 3]
        if axes_world is None:
            axes_world = self._axes_world(q)
        a6 = axes_world[5]
        if np.linalg.norm(a6) < 1e-12:
            a6 = Tee[:3, :3][:, 2]
        return Tee[:3, 3] - kin.d6 * (a6 / (np.linalg.norm(a6) + 1e-12))

    # ---------- forward/backward ----------
    def _forward_stage(self, p, x, y, z, L, types, target_p, target_R, fix_orientation=True):
        p[6] = target_p.copy()
        if fix_orientation and target_R is not None:
            x[6] = self._unit(target_R[:, 0]); y[6] = self._unit(target_R[:, 1]); z[6] = self._unit(target_R[:, 2])

        for i in range(5, -1, -1):
            if types[i] == "pris":
                z_i = self._unit(z[i+1]); di = p[i+1] - p[i]
                l_i = float(np.dot(di, z_i))
                p[i] = p[i+1] - z_i * l_i
                y[i] = self._unit(np.cross(z_i, x[i])); z[i] = z_i
                continue

            if types[i] == "hinge":
                di = p[i+1] - p[i]
                x_next = self._unit(x[i+1])
                v = di - x_next * float(np.dot(di, x_next))
                z_i = self._unit(v) if np.linalg.norm(v) >= 1e-12 else self._unit(z[i+1])
                p[i] = p[i+1] - z_i * L[i]
                x[i] = x_next
                y[i] = self._unit(np.cross(z_i, x[i])); z[i] = z_i
            else:  # pivot
                z_i = self._unit(z[i+1])
                p[i] = p[i+1] - z_i * L[i]
                j = max(i-2, 0)
                z_im1 = self._unit((p[i] - p[j])) if np.linalg.norm(p[i] - p[j]) > 1e-12 else self._unit(z[i])
                x_i = np.cross(z_i, z_im1)
                if np.linalg.norm(x_i) < 1e-12:
                    x_i = x[i+1]
                x_i = self._unit(x_i)
                if np.dot(x_i, x[i+1]) < 0:
                    x_i = -x_i
                x[i] = x_i
                y[i] = self._unit(np.cross(z_i, x_i)); z[i] = z_i

    def _backward_stage(self, p, x, y, z, L, types, base_p, base_R):
        p[0] = base_p.copy()
        if base_R is not None:
            x[0] = self._unit(base_R[:, 0]); y[0] = self._unit(base_R[:, 1]); z[0] = self._unit(base_R[:, 2])

        for i in range(0, 6):
            if types[i] == "pris":
                z_ip1 = self._unit(z[i])
                p[i+1] = p[i] + z_ip1 * L[i]
                x[i+1] = x[i]
                y[i+1] = self._unit(np.cross(z_ip1, x[i+1])); z[i+1] = z_ip1
                continue

            if types[i] == "hinge":
                di = p[i+1] - p[i]
                x_i = self._unit(x[i])
                v = di - x_i * float(np.dot(di, x_i))
                z_ip1 = self._unit(v) if np.linalg.norm(v) >= 1e-12 else self._unit(z[i])
                p[i+1] = p[i] + z_ip1 * L[i]
                x[i+1] = x_i
                y[i+1] = self._unit(np.cross(z_ip1, x[i+1])); z[i+1] = z_ip1
            else:  # pivot
                z_ip1 = self._unit(z[i])
                p[i+1] = p[i] + z_ip1 * L[i]
                j = min(i+3, 6)
                z_ip2 = self._unit((p[j] - p[i+1])) if np.linalg.norm(p[j] - p[i+1]) > 1e-12 else self._unit(z[i+1])
                x_ip1 = np.cross(z_ip1, z_ip2)
                if np.linalg.norm(x_ip1) < 1e-12:
                    x_ip1 = x[i]
                x[i+1] = self._unit(x_ip1)
                y[i+1] = self._unit(np.cross(z_ip1, x[i+1])); z[i+1] = z_ip1

    # ---------- q 복원 ----------
    @staticmethod
    def _signed_angle_around_axis(v_from, v_to, axis):
        a = axis / (np.linalg.norm(axis) + 1e-12)
        vf = v_from - a * np.dot(a, v_from)
        vt = v_to   - a * np.dot(a, v_to)
        nf = np.linalg.norm(vf); nt = np.linalg.norm(vt)
        if nf < 1e-12 or nt < 1e-12:
            return 0.0
        vf /= nf; vt /= nt
        s = np.dot(a, np.cross(vf, vt))
        c = np.clip(np.dot(vf, vt), -1.0, 1.0)
        return float(np.arctan2(s, c))

    def _positions_to_q(self, q_in, p_des, gain=None):
        kin = self.kinematics
        roles = getattr(kin, "joint_roles", ["hinge"]*6)
        q = q_in.copy()

        Tee, Ts = kin.forward_kinematics(q)
        axes_world = self._axes_world(q)
        wc_cur = self._wrist_center(q, Tee, axes_world)
        wc_des = p_des[6]
        g = self.q_gain if gain is None else float(gain)

        for i in range(6):
            a = axes_world[i]
            if np.linalg.norm(a) < 1e-10:
                continue  # prismatic skip
            a = a / np.linalg.norm(a)
            pj = Ts[i][:3, 3]

            if roles[i] == "pivot":
                # twist around axis using projected wrist vector
                r_cur = wc_cur - pj
                r_des = wc_des - p_des[i]
                r_cur -= a * np.dot(a, r_cur)
                r_des -= a * np.dot(a, r_des)
                if np.linalg.norm(r_cur) < 1e-12 or np.linalg.norm(r_des) < 1e-12:
                    v_cur = (Ts[i+1][:3,3] if i < 5 else wc_cur) - pj
                    v_des = p_des[i+1] - p_des[i]
                    dtheta = self._signed_angle_around_axis(v_cur, v_des, a)
                else:
                    dtheta = self._signed_angle_around_axis(r_cur, r_des, a)
            else:  # hinge
                v_cur = (Ts[i+1][:3,3] if i < 5 else wc_cur) - pj
                v_des = p_des[i+1] - p_des[i]
                if np.linalg.norm(v_des) < 1e-12 or np.linalg.norm(v_cur) < 1e-12:
                    continue
                dtheta = self._signed_angle_around_axis(v_cur, v_des, a)

            q[i] = np.clip(q[i] + g * float(dtheta), kin.lower[i], kin.upper[i])

        # weak regularization toward mid-range
        if self.q_reg > 0.0:
            q = kin.clamp((1.0 - self.q_reg) * q + self.q_reg * self.q_mid)
        return q

    # ---------- 메인 ----------
    def solve(self, target_pose: np.ndarray, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        # base info
        p, x, y, z, L, axes, is_pris = self._frame_arrays(q)
        base_p = p[0].copy()
        _, Ts = kin.forward_kinematics(q)
        base_R = Ts[0][:3, :3].copy()

        p_t = target_pose[:3, 3].copy()
        R_t = target_pose[:3, :3].copy()
        target_wc = p_t - kin.d6 * R_t[:, 2] if abs(kin.d6) > 1e-6 else p_t

        if np.linalg.norm(target_wc - base_p) > (np.sum(L) + 1e-6):
            return kin.clamp(q), False

        max_step = np.deg2rad(self.max_step_deg)
        local_gain = self.q_gain

        # helper to compute EE cost
        def ee_cost(qv):
            Tee, _ = kin.forward_kinematics(qv)
            dp = float(np.linalg.norm(Tee[:3,3] - p_t))
            M = Tee[:3,:3] @ R_t.T
            tr = np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)
            ang = float(np.arccos(tr))
            return dp, ang, dp + ang  # simple additive cost

        # initial cost
        pe, oe, cost_prev = ee_cost(q)

        ok = False
        for _ in range(self.max_iter):
            # orientation gating: enforce R only when close in position
            fix_orientation = (pe < self.orient_gate_mul * self.tol_pos)

            # store previous chain to relax later
            p_prev = [pi.copy() for pi in p]

            types = kin.joint_roles  # fixed per model
            self._forward_stage(p, x, y, z, L, types, target_wc, R_t if fix_orientation else None, fix_orientation)
            self._backward_stage(p, x, y, z, L, types, base_p, base_R)

            # position relaxation (blend)
            if self.relax_pos > 0.0:
                a = float(self.relax_pos)
                for i in range(len(p)):
                    p[i] = (1.0 - a) * p_prev[i] + a * p[i]

            # propose q update
            q_prev = q.copy()
            q_prop = self._positions_to_q(q_prev, p, gain=local_gain)

            # step cap per joint
            dq = q_prop - q_prev
            dq = np.clip(dq, -max_step, max_step)
            q_step = kin.clamp(q_prev + dq)

            # smoothing
            if self.smooth_q > 0.0:
                q = kin.clamp((1.0 - self.smooth_q) * q_step + self.smooth_q * q_prev)
            else:
                q = q_step

            # evaluate cost, backtrack if worse
            pe, oe, cost_now = ee_cost(q)
            if cost_now > cost_prev:
                # backtrack & reduce local gain a bit
                q = kin.clamp(0.5 * (q_prev + q))
                local_gain = max(0.4 * local_gain, 0.15 * self.q_gain)
                # recalc cost after backtrack
                pe, oe, cost_now = ee_cost(q)
            else:
                # gentle recovery of gain
                local_gain = min(max(local_gain, 0.5*self.q_gain) * 1.05, self.q_gain)

            # convergence
            if pe < self.tol_pos and oe < self.tol_rot:
                ok = True
                break

            # refresh frames for next loop
            p, x, y, z, L, axes, is_pris = self._frame_arrays(q)
            cost_prev = cost_now

        return kin.clamp(q), ok


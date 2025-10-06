# heuristic_ik/solvers.py
import numpy as np
from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel

# =========================
# Dual Quaternion utilities
# =========================
# Quaternion is [w, x, y, z]
def _qmul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def _qconj(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]], float)

def _qnormalize(q):
    n = np.linalg.norm(q)
    return q if n < 1e-15 else q / n

def _quat_from_axis_angle(axis, theta):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-15:
        return np.array([1.,0.,0.,0.], float)
    a = a / n
    half = 0.5 * theta
    s = np.sin(half)
    return np.array([np.cos(half), a[0]*s, a[1]*s, a[2]*s], float)

def _rotmat_from_quat(q):
    # for debug/optionally building t = (I-R) p
    w,x,y,z = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], float)
    return R

class DualQuaternion:
    """
    Rigid transform as a unit dual quaternion:
      Q = qr + ε qd,  where qr is unit quaternion (rotation), qd encodes translation t via
      qd = 0.5 * (0, t) * qr
    """
    __slots__ = ("qr","qd")
    def __init__(self, qr, qd):
        self.qr = _qnormalize(np.asarray(qr, float))
        self.qd = np.asarray(qd, float)

    @staticmethod
    def from_rt(R, t):
        # R: 3x3 rotation, t: 3
        # get qr from R via robust formula (through axis-angle would also be fine)
        # Here we rely on R being orthonormal (Pinocchio guarantees)
        # Convert R -> quaternion (fast)
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = np.sqrt(tr+1.0)*2.0
            w = 0.25*S
            x = (R[2,1] - R[1,2]) / S
            y = (R[0,2] - R[2,0]) / S
            z = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S; x = 0.25*S
            y = (R[0,1] + R[1,0]) / S; z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S; x = (R[0,1] + R[1,0]) / S
            y = 0.25*S; z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S; x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S; z = 0.25*S
        qr = _qnormalize(np.array([w,x,y,z], float))
        t_quat = np.array([0., t[0], t[1], t[2]], float)
        qd = 0.5 * _qmul(t_quat, qr)
        return DualQuaternion(qr, qd)

    @staticmethod
    def pure_rotation_about_axis_through_point(axis, theta, point):
        """
        Pure rotation around axis 'axis' by angle theta, axis passing through 'point' (world).
        The rigid transform is: x' = p + R (x - p)  =>  R rotation, translation t = (I - R) p
        """
        qr = _quat_from_axis_angle(axis, theta)
        R = _rotmat_from_quat(qr)
        t = (np.eye(3) - R) @ np.asarray(point, float)
        t_quat = np.array([0., t[0], t[1], t[2]], float)
        qd = 0.5 * _qmul(t_quat, qr)
        return DualQuaternion(qr, qd)

    def normalize(self):
        n = np.linalg.norm(self.qr)
        if n < 1e-15:
            self.qr = np.array([1.,0.,0.,0.], float)
            self.qd = np.array([0.,0.,0.,0.], float)
            return self
        self.qr /= n; self.qd /= n
        return self

    def conj(self):
        return DualQuaternion(_qconj(self.qr), _qconj(self.qd))

    def mul(self, other):
        qr = _qmul(self.qr, other.qr)
        qd = _qmul(self.qr, other.qd) + _qmul(self.qd, other.qr)
        return DualQuaternion(qr, qd)

    def transform_point(self, p):
        # p' = r p r* + t, with t = 2 * qd * r*
        r = self.qr; d = self.qd
        p_q = np.array([0., p[0], p[1], p[2]], float)
        r_conj = _qconj(r)
        pr = _qmul(_qmul(r, p_q), r_conj)
        t_q = _qmul(d, r_conj)
        t = 2.0 * t_q[1:]
        return pr[1:] + t


# =======================================================
# Legacy FABRIK_R (kept as-is so you can switch if wanted)
# =======================================================
class FABRIK_R(IKSolverBase):
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.max_iter_fabrik = 100
        self.tol_fabrik = 1e-3
        self.q0 = np.deg2rad([0,30,-30,0,0,0])
        self.align_passes = 1
        self.tol_align = 2e-3
        self.max_rounds = 10
        self.prev_plane_n = None
        self.last_q = None
        self.is_joint_limits = False

    @staticmethod
    def _norm(v):
        n = np.linalg.norm(v)
        return v if n < 1e-12 else v / n

    def _place(self, prev, curr, L):
        d = np.linalg.norm(curr - prev)
        return prev.copy() if d < 1e-12 else prev + (curr - prev) * (L / d)

    def _project_plane(self, p, n, p0):
        n = self._norm(n)
        return p - np.dot(p - p0, n) * n

    def _points(self, q):
        return self.kinematics.chain_points(q)

    def _base_axis(self, q):
        kin = self.kinematics
        a = kin.joint_axis_world(q, kin.joint_names[0])
        return self._norm(a)

    def _vert_plane_general(self, p0, pts, q_cur):
        a1 = self._base_axis(q_cur)
        bestn = -1.0
        best_v_perp = None
        base = p0
        for k in (5, 4, 3, 2):
            v = pts[k] - base
            v_perp = v - a1 * np.dot(a1, v)
            nrm = np.linalg.norm(v_perp)
            if nrm > bestn:
                bestn = nrm
                best_v_perp = v_perp

        if bestn < 1e-9 or best_v_perp is None:
            if self.prev_plane_n is not None:
                nrm = self.prev_plane_n
            else:
                tmp = np.array([1.0, 0, 0]) if abs(a1[0]) < 0.9 else np.array([0, 1.0, 0])
                nrm = self._norm(np.cross(a1, tmp))
        else:
            nrm = self._norm(np.cross(a1, best_v_perp))

        self.prev_plane_n = nrm
        return nrm, p0

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _fabrik(self, q_for_plane, pts_init, L, target):
        pts = pts_init.copy()
        p0 = pts[0].copy()

        to_tgt = target - p0
        dist = np.linalg.norm(to_tgt)
        sumL = float(np.sum(L))
        if dist > sumL + 1e-12:
            dirv = self._norm(to_tgt)
            for i in range(1, 7):
                pts[i] = pts[i - 1] + L[i - 1] * dirv
            return pts, 1

        it = 0
        for _ in range(self.max_iter_fabrik):
            it += 1
            pts[6] = target.copy()
            pts[5] = self._place(pts[6], pts[5], L[5])
            n_plane, p_plane = self._vert_plane_general(p0, pts, q_for_plane)
            for i in (4, 3, 2):
                proj = self._project_plane(pts[i], n_plane, p_plane) if n_plane is not None else pts[i]
                pts[i] = self._place(pts[i + 1], proj, L[i])
            pts[1] = self._place(pts[2], pts[1], L[1])
            pts[0] = p0.copy()

            pts[0] = p0.copy()
            n_plane, p_plane = self._vert_plane_general(p0, pts, q_for_plane)
            for i in range(0, 6):
                nxt = pts[i + 1]
                if (i + 1) in (2, 3, 4) and n_plane is not None:
                    nxt = self._project_plane(nxt, n_plane, p_plane)
                pts[i + 1] = self._place(pts[i], nxt, L[i])

            if np.linalg.norm(pts[6] - target) < self.tol_fabrik:
                break

        return pts, it

    def _align(self, q_init, P_target):
        kin = self.kinematics
        q = q_init.copy()
        eps = 1e-10
        steps = 0

        def best_k(i, a, pts):
            base = pts[i]; best = i + 1; bestn = -1.0
            for k in range(i + 1, 7):
                v = pts[k] - base
                v_perp = v - a * np.dot(a, v)
                n = np.linalg.norm(v_perp)
                if n > bestn:
                    bestn = n; best = k
            return best, bestn

        for _ in range(self.align_passes):
            kin._full_fk(q)
            changed = False
            for i in range(1, 6):
                nm = kin.joint_names[i - 1]
                jtype = kin.joint_type.get(nm, 'revolute')
                a = kin.joint_axis_world(q, nm)
                a = a / (np.linalg.norm(a) + 1e-15)

                k, spread = best_k(i, a, P_target)
                if spread < 1e-6:
                    continue

                pts_cur = self.kinematics.chain_points(q)
                p_i, p_k = pts_cur[i], pts_cur[k]
                r_cur = p_k - p_i
                r_tgt = P_target[k] - P_target[i]

                if jtype in ('revolute', 'continuous'):
                    r_p = r_cur - a * np.dot(a, r_cur)
                    t_p = r_tgt - a * np.dot(a, r_tgt)
                    if np.linalg.norm(r_p) < eps or np.linalg.norm(t_p) < eps:
                        continue
                    th = np.arctan2(np.dot(a, np.cross(r_p, t_p)),
                                    np.dot(r_p, t_p))
                    if abs(th) > 1e-6:
                        q[i - 1] = q[i - 1] + th
                        if jtype == 'revolute' and self.is_joint_limits:
                            q[i - 1] = np.clip(q[i - 1], kin.lower[i - 1], kin.upper[i - 1])
                        else:
                            q[i - 1] = self._wrap_angle(q[i - 1])
                        changed = True; steps += 1
                        kin._full_fk(q)

                elif jtype == 'prismatic':
                    pts_cur = self.kinematics.chain_points(q)
                    p_ip1 = pts_cur[i + 1]
                    rseg_cur = p_ip1 - p_i
                    rseg_tgt = P_target[i + 1] - P_target[i]
                    delta = np.dot(rseg_tgt, a) - np.dot(rseg_cur, a)
                    if abs(delta) > 1e-6:
                        q[i-1] = q[i-1] + delta
                        changed = True; steps += 1
                        kin._full_fk(q)

            if not changed:
                break

        return q, steps

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        if q_seed is not None:
            q = np.asarray(q_seed, float).copy()
        elif self.last_q is not None:
            q = self.last_q.copy()
        else:
            q = self.q0.copy()

        Tt = np.asarray(target_pose, float)
        p_ee = Tt[:3, 3]; R_ee = Tt[:3, :3]
        r = kin.r_ee_to_j6_ee if kin.r_ee_to_j6_ee is not None else np.zeros(3)
        p6 = p_ee + R_ee @ r

        tot_fab = 0; tot_align = 0
        ok = False
        q_best = q.copy()
        best_err = np.inf

        for _ in range(self.max_rounds):
            P0 = kin.chain_points(q)
            L = np.linalg.norm(P0[1:] - P0[:-1], axis=1)
            P_sol, it_fab = self._fabrik(q_for_plane=q, pts_init=P0, L=L, target=p6)
            tot_fab += it_fab

            q_new, steps = self._align(q, P_sol)
            tot_align += steps

            P_fin = kin.chain_points(q_new)
            pos_err_chain = np.linalg.norm(P_fin[2:7] - P_sol[2:7], axis=1).max()
            pos_err_p6 = np.linalg.norm(P_sol[6] - p6)
            ok = (pos_err_p6 < self.tol_fabrik) and (pos_err_chain < self.tol_align)

            score = pos_err_p6 + pos_err_chain
            if score < best_err:
                best_err = score
                q_best = q_new.copy()

            if ok:
                q = q_new
                break

            if steps == 0 and it_fab <= 1:
                q = q_new
                break

            q = q_new

        q_fin = q if ok else q_best
        self.last_q = q_fin.copy()
        stats = {
            'iters_total': int(max(1, tot_fab + tot_align)),
            'iters_fabrik': int(tot_fab),
            'iters_align': int(tot_align)
        }
        return q_fin, bool(ok), stats


# ==========================================
# NEW: DQ-FABRIK (faithful dual-quaternion)
# ==========================================
class DQ_FABRIK(IKSolverBase):
    """
    Faithful to the DQ-FABRIK paper:
      - Represent local joint updates as **pure-rotation dual quaternions**
      - Positions-only FABRIK to get target chain (wrist center p6)
      - Axis alignment via **DQ rotation about joint axis through joint point**
      - Orientation stage via **swing (align EE z) + twist (align EE x about z)** using DQ
    No Jacobian; no numerical differentiation.
    """

    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        # position loop
        self.max_iter_fabrik = 120
        self.tol_fabrik = 1e-3
        # alignment
        self.align_passes = 2       # few passes are still faithful; can be 1
        self.tol_align = 2e-3
        self.max_rounds = 8
        # orientation (swing/twist)
        self.ori_max_iter = 30
        self.ori_tol_rad = np.deg2rad(1.0)
        self.ori_step = 1.0
        # misc
        self.q0 = np.deg2rad([0, 30, -30, 0, 0, 0])
        self.last_q = None
        self.is_joint_limits = True

    # ---- helpers ----
    @staticmethod
    def _norm(v):
        n = np.linalg.norm(v)
        return v if n < 1e-12 else v / n

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _place(self, prev, curr, L):
        d = np.linalg.norm(curr - prev)
        return prev.copy() if d < 1e-12 else prev + (curr - prev) * (L / d)

    # ------------- FABRIK (positions-only) -------------
    def _fabrik_positions(self, pts_init, L, target, base_lock=True):
        pts = pts_init.copy()
        base = pts[0].copy()
        if np.linalg.norm(target - base) > float(np.sum(L)) + 1e-12:
            dirv = self._norm(target - base)
            for i in range(1, 7):
                pts[i] = pts[i-1] + L[i-1] * dirv
            return pts, 1

        it = 0
        for _ in range(self.max_iter_fabrik):
            it += 1
            # Backward
            pts[6] = target.copy()
            for i in range(5, -1, -1):
                pts[i] = self._place(pts[i+1], pts[i], L[i])
            # Forward
            if base_lock:
                pts[0] = base.copy()
            for i in range(0, 6):
                pts[i+1] = self._place(pts[i], pts[i+1], L[i])
            if np.linalg.norm(pts[6] - target) < self.tol_fabrik:
                break
        return pts, it

    # ----- Axis alignment using **pure-rotation DQ** about joint axis -----
    def _align_axes_dq(self, q_init, P_target, p6_target):
        """
        Few passes of base->tip sweeps. Each joint update is done as a pure-rotation DQ
        about the joint axis through the joint point. We still update the *joint angle*
        (Pinocchio FK), but the computed transform matches the DQ formulation.
        """
        kin = self.kinematics
        q = q_init.copy()
        eps = 1e-12
        total_steps = 0

        for _ in range(self.align_passes):
            kin._full_fk(q)
            changed = False

            for i in range(1, 6):  # joints 1..5 (index 0..4)
                nm = kin.joint_names[i - 1]
                jtype = kin.joint_type.get(nm, 'revolute')
                if jtype not in ('revolute', 'continuous'):
                    continue

                pts_cur = kin.chain_points(q)       # world points 0..6
                p_i = pts_cur[i]
                p6_cur = pts_cur[6]
                a = kin.joint_axis_world(q, nm)
                a = a / (np.linalg.norm(a) + 1e-15)

                # Desired vectors relative to joint i
                p_i_tar = P_target[i]
                r_cur = p6_cur - p_i
                r_tar = p6_target - p_i_tar

                # project to plane ⟂ a  (pure rotation about 'a' should align these)
                r_p = r_cur - a * np.dot(a, r_cur)
                t_p = r_tar - a * np.dot(a, r_tar)

                # degeneracy guards: if too small, fall back to segment i->i+1
                if np.linalg.norm(r_p) < 1e-9 or np.linalg.norm(t_p) < 1e-9:
                    r_cur2 = pts_cur[i + 1] - p_i
                    r_tar2 = P_target[i + 1] - p_i_tar
                    r_p = r_cur2 - a * np.dot(a, r_cur2)
                    t_p = r_tar2 - a * np.dot(a, r_tar2)
                    if np.linalg.norm(r_p) < 1e-9 or np.linalg.norm(t_p) < 1e-9:
                        continue

                # minimal rotation angle (about 'a') from r_p -> t_p (same as paper's u->v)
                num = np.dot(a, np.cross(r_p, t_p))
                den = np.dot(r_p, t_p)
                th = np.arctan2(num, den)
                if abs(th) < 1e-9:
                    continue

                # --- This is where we use a *pure-rotation DQ* (axis through p_i) ---
                # Build DQ and (conceptually) apply it to downstream; in practice, we
                # just update q[i-1] by th which is exactly this rotation in the model.
                dq = DualQuaternion.pure_rotation_about_axis_through_point(a, th, p_i).normalize()
                # Note: we don't need to explicitly move points with dq; FK(q) will.

                # Apply as a joint angle update, respecting limits if needed
                q[i - 1] = q[i - 1] + th
                if jtype == 'revolute' and self.is_joint_limits:
                    q[i - 1] = np.clip(q[i - 1], kin.lower[i - 1], kin.upper[i - 1])
                else:
                    q[i - 1] = self._wrap_angle(q[i - 1])

                total_steps += 1
                changed = True
                kin._full_fk(q)

            if not changed:
                break

        return q, total_steps

    # ----- Orientation via DQ swing & twist -----
    def _align_orientation_swing_twist_dq(self, q_init, R_target):
        """
        Swing: align EE z-axis to target z by applying pure-rotation DQ about wrist axes (6->5->4)
        Twist: rotate about joint6 axis to align EE x-axis around z.
        """
        kin = self.kinematics
        q = q_init.copy()
        ez = np.array([0., 0., 1.])
        ex = np.array([1., 0., 0.])

        def ang_between(u, v):
            cu = u / (np.linalg.norm(u) + 1e-15)
            cv = v / (np.linalg.norm(v) + 1e-15)
            c = np.clip(np.dot(cu, cv), -1.0, 1.0)
            return np.arccos(c)

        for it in range(self.ori_max_iter):
            T_cur, _ = kin.forward_kinematics(q)
            R_cur = T_cur[:3, :3]
            p_all = kin.chain_points(q)
            z_cur = R_cur @ ez
            z_tgt = R_target @ ez

            # 1) Swing (6 -> 5 -> 4)
            swing_err = ang_between(z_cur, z_tgt)
            if swing_err > self.ori_tol_rad:
                for idx in (5, 4, 3):
                    nm = kin.joint_names[idx]
                    jt = kin.joint_type.get(nm, 'revolute')
                    if jt not in ('revolute', 'continuous'):
                        continue
                    a = kin.joint_axis_world(q, nm); a = a / (np.linalg.norm(a)+1e-15)
                    p_i = p_all[idx]
                    th = np.arctan2(np.dot(a, np.cross(z_cur, z_tgt)), np.dot(z_cur, z_tgt))
                    if abs(th) < 1e-9:
                        continue
                    th *= self.ori_step
                    dq = DualQuaternion.pure_rotation_about_axis_through_point(a, th, p_i).normalize()
                    # apply as joint update (equivalent)
                    q[idx] = q[idx] + th
                    if jt == 'revolute' and self.is_joint_limits:
                        q[idx] = np.clip(q[idx], kin.lower[idx], kin.upper[idx])
                    else:
                        q[idx] = self._wrap_angle(q[idx])
                    kin._full_fk(q)
                    T_cur, _ = kin.forward_kinematics(q)
                    R_cur = T_cur[:3, :3]
                    z_cur = R_cur @ ez
                    swing_err = ang_between(z_cur, z_tgt)
                    if swing_err < self.ori_tol_rad:
                        break

            # 2) Twist about joint6 axis to align EE x around z
            T_cur, _ = kin.forward_kinematics(q)
            R_cur = T_cur[:3, :3]
            p_all = kin.chain_points(q)
            x_cur = R_cur @ ex
            x_tgt = R_target @ ex
            a6 = kin.joint_axis_world(q, kin.joint_names[5]); a6 = a6 / (np.linalg.norm(a6)+1e-15)
            p6 = p_all[5]
            th_twist = np.arctan2(np.dot(a6, np.cross(x_cur, x_tgt)), np.dot(x_cur, x_tgt))
            if abs(th_twist) > 1e-9:
                th_twist *= self.ori_step
                dq = DualQuaternion.pure_rotation_about_axis_through_point(a6, th_twist, p6).normalize()
                q[5] = q[5] + th_twist
                jt6 = kin.joint_type.get(kin.joint_names[5], 'revolute')
                if jt6 == 'revolute' and self.is_joint_limits:
                    q[5] = np.clip(q[5], kin.lower[5], kin.upper[5])
                else:
                    q[5] = self._wrap_angle(q[5])

            # stop if small orient error
            T_try, _ = kin.forward_kinematics(q)
            R_try = T_try[:3, :3]
            R_err = R_target @ R_try.T
            cosang = max(-1.0, min(1.0, (np.trace(R_err) - 1.0) * 0.5))
            ori_err = np.arccos(cosang)
            if ori_err < self.ori_tol_rad:
                return q, it + 1

        return q, self.ori_max_iter

    # ------------- public solve -------------
    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics

        if q_seed is not None:
            q = np.asarray(q_seed, float).copy()
        elif self.last_q is not None:
            q = self.last_q.copy()
        else:
            q = self.q0.copy()

        Tt = np.asarray(target_pose, float)
        p_ee = Tt[:3, 3]; R_ee = Tt[:3, :3]
        r = kin.r_ee_to_j6_ee if kin.r_ee_to_j6_ee is not None else np.zeros(3)
        p6 = p_ee + R_ee @ r

        tot_fab = 0; tot_align = 0; tot_ori = 0
        ok = False
        q_best = q.copy(); best_score = np.inf

        for _ in range(self.max_rounds):
            # chain points + lengths
            P0 = kin.chain_points(q)
            L = np.linalg.norm(P0[1:] - P0[:-1], axis=1)

            # (1) positions-only FABRIK to get p6 target chain
            P_tar, it_fab = self._fabrik_positions(P0, L, p6, base_lock=True)
            tot_fab += it_fab

            # (2) axis alignment via pure-rotation DQ about each joint axis
            q_pos, steps = self._align_axes_dq(q, P_tar, p6)
            tot_align += steps

            # (3) orientation via DQ swing & twist
            q_new, it_ori = self._align_orientation_swing_twist_dq(q_pos, R_ee)
            tot_ori += it_ori

            # evaluate
            P_fin = kin.chain_points(q_new)
            pos_err_p6 = np.linalg.norm(P_fin[6] - p6)
            pos_err_chain = np.linalg.norm(P_fin[2:7] - P_tar[2:7], axis=1).max()

            T_try, _ = kin.forward_kinematics(q_new)
            R_cur = T_try[:3, :3]
            R_err = R_ee @ R_cur.T
            cosang = max(-1.0, min(1.0, (np.trace(R_err) - 1.0) * 0.5))
            ori_err = np.arccos(cosang)

            score = pos_err_p6 + pos_err_chain + 0.1 * ori_err
            if score < best_score:
                best_score = score
                q_best = q_new.copy()

            ok = (pos_err_p6 < self.tol_fabrik) and (pos_err_chain < self.tol_align) and (ori_err < self.ori_tol_rad)
            q = q_new
            if ok:
                break

            # if little progress, stop
            if steps == 0 and it_fab <= 1 and it_ori <= 1:
                break

        q_fin = q if ok else q_best
        self.last_q = q_fin.copy()
        info = {
            'iters_total': int(max(1, tot_fab + tot_align + tot_ori)),
            'iters_fabrik': int(tot_fab),
            'iters_align': int(tot_align),
            'iters_ori': int(tot_ori),
        }
        return q_fin, bool(ok), info


# ---------------- 2D examples kept as-is ----------------
class CCD2D:
    def __init__(self, link_lengths):
        self.L = np.asarray(link_lengths, float)
        self.max_iter = 200
        self.tol = 1e-3

    def fk(self, q):
        q = np.asarray(q, float)
        pts = [np.zeros(2)]
        angle=0.0; p=np.zeros(2)
        for i, (qi, Li) in enumerate(zip(q, self.L)):
            angle += qi
            d = np.array([np.cos(angle)*Li, np.sin(angle)*Li])
            p = p + d
            pts.append(p.copy())
        return np.asarray(pts)

    def solve(self, target, q_seed):
        q = np.asarray(q_seed, float).copy()
        for it in range(self.max_iter):
            pts = self.fk(q); pe = np.linalg.norm(pts[-1]-target)
            if pe < self.tol: 
                return q, True, {'iters_total': it+1, 'pos_err': pe}
            for i in reversed(range(len(q))):
                pi = pts[i]; pe_pt = pts[-1]
                v1 = pe_pt - pi
                v2 = target - pi
                if np.linalg.norm(v1)<1e-9 or np.linalg.norm(v2)<1e-9:
                    continue
                th = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                q[i] += th
                pts = self.fk(q)
        pts = self.fk(q)
        return q, False, {'iters_total': self.max_iter, 'pos_err': np.linalg.norm(pts[-1]-target)}

class FABRIK2D:
    def __init__(self, link_lengths):
        self.L = np.asarray(link_lengths, float)
        self.max_iter = 200
        self.tol = 1e-3

    def forward_points(self, q):
        q = np.asarray(q, float)
        pts = [np.zeros(2)]
        angle=0.0; p=np.zeros(2)
        for qi, Li in zip(q, self.L):
            angle += qi
            p = p + np.array([np.cos(angle)*Li, np.sin(angle)*Li])
            pts.append(p.copy())
        return np.asarray(pts)

    def solve(self, target, q_seed):
        N = len(self.L)
        q = np.asarray(q_seed, float).copy()
        pts = self.forward_points(q)
        base = pts[0].copy()
        L = self.L.copy()
        for it in range(self.max_iter):
            if np.linalg.norm(pts[-1]-target) < self.tol:
                return q, True, {'iters_total': it+1, 'pos_err': np.linalg.norm(pts[-1]-target)}
            # forward
            pts[-1] = target.copy()
            for i in reversed(range(N)):
                r = np.linalg.norm(pts[i+1]-pts[i])
                pts[i] = pts[i+1] + (pts[i]-pts[i+1]) * (L[i]/r)
            pts[0] = base.copy()
            # backward
            for i in range(N):
                r = np.linalg.norm(pts[i+1]-pts[i])
                pts[i+1] = pts[i] + (pts[i+1]-pts[i]) * (L[i]/r)
            # get angles
            ang = 0.0
            for i in range(N):
                v = pts[i+1]-pts[i]
                th = np.arctan2(v[1], v[0])
                dth = th-ang
                dth = (dth + np.pi)%(2*np.pi) - np.pi
                q[i] += dth
                ang = th
        return q, False, {'iters_total': self.max_iter, 'pos_err': np.linalg.norm(pts[-1]-target)}

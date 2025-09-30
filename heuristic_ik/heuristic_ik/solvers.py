# heuristic_ik/solvers.py
import numpy as np
from ik_common.common.base import IKSolverBase
from ik_common.common.kinematics import KinematicModel

# ---------- 3D FABRIK (URDF/Pinocchio) ----------
# 기존 구현을 옮겨와 약간 정리했습니다.
import numpy as np
class FABRIK3D(IKSolverBase):
    def __init__(self, kinematics: KinematicModel):
        super().__init__(kinematics)
        self.max_iter_fabrik = 80
        self.tol_fabrik = 1e-3
        self.q0 = np.deg2rad(np.array([55.0, 0.0, 205.0, 0.0, 85.0, 0.0], float))
        self.align_passes = 3
        self.tol_align = 2e-3

    @staticmethod
    def _norm(v): 
        n=np.linalg.norm(v); 
        return v if n<1e-12 else v/n

    def _place(self, prev, curr, L):
        d=np.linalg.norm(curr-prev)
        return prev.copy() if d<1e-12 else prev + (curr-prev)*(L/d)

    def _project_plane(self, p, n, p0):
        n = self._norm(n); return p - np.dot(p-p0, n)*n

    def _vert_plane(self, base, through):
        z=np.array([0,0,1.0]); v=through-base; vxy=np.array([v[0],v[1],0.0])
        normal = np.array([1.0,0,0]) if np.linalg.norm(vxy)<1e-9 else np.cross(z, vxy)
        return self._norm(normal), base.copy()

    def _points(self, q): 
        return self.kinematics.chain_points(q)

    def _fabrik(self, pts, L, target):
        pts = pts.copy(); p0=pts[0].copy(); p1=pts[1].copy()
        if np.linalg.norm(pts[6]-target)<self.tol_fabrik: 
            return pts, 0
        it=0
        for _ in range(self.max_iter_fabrik):
            it += 1
            pts[0]=p0; pts[1]=p1
            pts[6]=target.copy()
            pts[5]=self._place(pts[6], pts[5], L[5])
            w_n, w_p = self._vert_plane(p0, pts[5])
            for i in [4,3,2]:
                proj = self._project_plane(pts[i], w_n, w_p)
                pts[i] = self._place(pts[i+1], proj, L[i])
            pts[0]=p0; pts[1]=p1
            v_n, v_p = self._vert_plane(p0, pts[5])
            for i in [1,2,3,4,5]:
                nxt = pts[i+1]
                if i+1 in (2,3,4):
                    nxt = self._project_plane(nxt, v_n, v_p)
                pts[i+1] = self._place(pts[i], nxt, L[i])
            if np.linalg.norm(pts[6]-target)<self.tol_fabrik:
                break
        return pts, it

    def _align(self, q_init, P):
        kin = self.kinematics
        q = kin.clamp(q_init.copy()); eps=1e-10; steps=0
        def best_k(i,a,pts):
            base=pts[i]; best=i+1; bestn=-1.0
            for k in range(i+1,7):
                v=pts[k]-base; v_perp = v - a*np.dot(a,v); n=np.linalg.norm(v_perp)
                if n>bestn: bestn=n; best=k
            return best, bestn
        for _ in range(self.align_passes):
            kin._full_fk(q); changed=False
            for i in range(1,6):
                nm = kin.joint_names[i-1]
                jtype = kin.joint_type.get(nm,'revolute')
                a = kin.joint_axis_world(q, nm); a/=np.linalg.norm(a)+1e-15
                k,spread = best_k(i,a,P)
                if spread<1e-6: continue
                pts_cur = self.kinematics.chain_points(q)
                p_i, p_k = pts_cur[i], pts_cur[k]
                r_cur = p_k-p_i; r_tgt = P[k]-P[i]
                if jtype in ('revolute','continuous'):
                    r_p = r_cur - a*np.dot(a,r_cur)
                    t_p = r_tgt - a*np.dot(a,r_tgt)
                    if np.linalg.norm(r_p)<eps or np.linalg.norm(t_p)<eps: 
                        continue
                    th = np.arctan2(np.dot(a, np.cross(r_p, t_p)), np.dot(r_p, t_p))
                    if abs(th)>1e-6:
                        q[i-1] = np.clip(q[i-1]+th, kin.lower[i-1], kin.upper[i-1]); changed=True; steps+=1
                        kin._full_fk(q)
                elif jtype=='prismatic':
                    p_ip1=pts_cur[i+1]; rseg_cur=p_ip1-p_i; rseg_tgt=P[i+1]-P[i]
                    delta = np.dot(rseg_tgt,a) - np.dot(rseg_cur,a)
                    if abs(delta)>1e-6:
                        q[i-1] = np.clip(q[i-1]+delta, kin.lower[i-1], kin.upper[i-1]); changed=True; steps+=1
                        kin._full_fk(q)
            if not changed: break
        return kin.clamp(q), steps

    def solve(self, target_pose, q_seed=None):
        kin = self.kinematics
        q0 = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed,float).copy())
        P0 = kin.chain_points(q0)
        L = np.linalg.norm(P0[1:] - P0[:-1], axis=1)
        Tt = np.asarray(target_pose, float); p_ee = Tt[:3,3]; R_ee=Tt[:3,:3]
        r = kin.r_ee_to_j6_ee if kin.r_ee_to_j6_ee is not None else np.zeros(3)
        p6 = p_ee + R_ee @ r
        P_sol, it_fab = self._fabrik(P0, L, p6)
        q_fin, steps = self._align(q0, P_sol)
        P_fin = kin.chain_points(q_fin)
        pos_err_chain = np.linalg.norm(P_fin[2:7]-P_sol[2:7], axis=1).max()
        ok = (np.linalg.norm(P_sol[6]-p6)<self.tol_fabrik) and (pos_err_chain<self.tol_align)
        tot = max(1, it_fab+steps)
        return kin.clamp(q_fin), bool(ok), {'iters_total': int(tot), 'iters_fabrik': int(it_fab), 'iters_align': int(steps)}

# ---------- 2D CCD ----------
class CCD2D:
    """Planar CCD for N-link arm (lengths L, joint angles q in rad)"""
    def __init__(self, link_lengths):
        self.L = np.asarray(link_lengths, float)
        self.max_iter = 200
        self.tol = 1e-3

    def fk(self, q):
        q = np.asarray(q, float)
        pts = [np.zeros(2)]
        T = np.eye(2); angle=0.0; p=np.zeros(2)
        for i, (qi, Li) in enumerate(zip(q, self.L)):
            angle += qi
            d = np.array([np.cos(angle)*Li, np.sin(angle)*Li])
            p = p + d
            pts.append(p.copy())
        return np.asarray(pts)  # 0..N

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

# ---------- 2D FABRIK ----------
class FABRIK2D:
    """Planar FABRIK (positions only)"""
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
            # get angles from positions
            ang = 0.0
            for i in range(N):
                v = pts[i+1]-pts[i]
                th = np.arctan2(v[1], v[0])
                dth = th-ang
                # wrap to [-pi, pi]
                dth = (dth + np.pi)%(2*np.pi) - np.pi
                q[i] += dth
                ang = th
        return q, False, {'iters_total': self.max_iter, 'pos_err': np.linalg.norm(pts[-1]-target)}

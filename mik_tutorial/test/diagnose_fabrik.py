#!/usr/bin/env python3
import os, sys, json, csv, math, argparse, importlib, importlib.util, types, traceback
import numpy as np

# ----------------------------- Utilities -----------------------------

def import_by_path_or_name(module_or_path: str):
    """
    Import a module given either a module name (e.g., 'my_pkg.ik')
    or a filesystem path to a .py file. Returns the imported module.
    """
    if os.path.exists(module_or_path) and module_or_path.endswith('.py'):
        spec = importlib.util.spec_from_file_location("user_module", module_or_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["user_module"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    # else treat as module name
    return importlib.import_module(module_or_path)

def se3_error_rotation_only(T_current, T_target):
    # Log map on SO(3): R_err = R_c^T R_t, then axis-angle norm
    Rc = T_current[:3,:3]
    Rt = T_target[:3,:3]
    R = Rc.T @ Rt
    # Robust log for small angles
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    return theta

def homogeneous_from_pos_quat(p, q):
    """ q = (x,y,z,w) """
    x,y,z,w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = np.array(p, dtype=float).reshape(3,)
    return T

# -------------------- Reference FABRIK-R (Instrumented) --------------------

class RefFabrikRIKSolverInstrumented:
    """
    A self-contained, instrumented FABRIK-R (Santos et al., 2021) position-only IK.
    It does not depend on Pinocchio; it expects a 'KinematicModel'-like object
    with forward_kinematics(q), jacobian(q, ref_frame) and joint limits.
    Diagnostics per-iteration are stored into self.logs.
    """
    def __init__(self, kinematics):
        self.kin = kinematics
        self.max_iter = 80
        self.tol = 1e-3
        lo, hi = kinematics.lower, kinematics.upper
        mid = np.where(np.isfinite(lo + hi), 0.5 * (lo + hi), 0.0)
        self.q0 = np.where(np.isfinite(mid), mid, 0.0)
        self.axis_parallel_tol = 0.995  # |dot| >= tol -> nearly parallel
        self.logs = []  # filled by solve()

    # ---------- helpers ----------
    @staticmethod
    def _unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    @staticmethod
    def _any_perp(n):
        n = RefFabrikRIKSolverInstrumented._unit(n)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(n, a); vn = float(np.linalg.norm(v))
        return v / vn if vn > 1e-12 else np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _rodrigues(v, axis, theta):
        a = RefFabrikRIKSolverInstrumented._unit(axis); c, s = math.cos(theta), math.sin(theta)
        return v*c + np.cross(a, v)*s + a*np.dot(a, v)*(1.0 - c)

    def _axes_world(self, q):
        # WORLD-frame joint axes from the WORLD geometric Jacobian (angular rows)
        # ref_frame 2 is pin.ReferenceFrame.WORLD; but to avoid importing pin, accept both int/enum
        try:
            WORLD = 2
            Jw = self.kin.jacobian(q, ref_frame=WORLD)
        except Exception:
            # try enum if available
            import pinocchio as pin  # type: ignore
            Jw = self.kin.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
        A = []
        for j in range(6):
            a = np.array(Jw[3:, j]).reshape(3,)
            n = float(np.linalg.norm(a))
            A.append(a / n if n > 1e-12 else np.array([0., 0., 1.]))
        return A  # [a1..a6], unit

    def _chain_positions(self, q):
        Tee, Ts = self.kin.forward_kinematics(q)
        ps = [T[:3, 3].copy() for T in Ts]  # p1..p6
        if abs(self.kin.d6) > 1e-9:
            a6 = self._axes_world(q)[5]
            ps.append(Tee[:3, 3] - self.kin.d6 * a6)
        else:
            ps.append(Tee[:3, 3].copy())
        return ps  # [p1..p6, p_wc]

    @staticmethod
    def _link_lengths(ps):
        return [float(np.linalg.norm(ps[i+1] - ps[i])) for i in range(6)]

    def _choose_anchor_for_s_forward(self, i, axes):
        for j in range(i-1, -1, -1):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        for j in range(i+1, 6):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        return max(i-1, 0)

    def _choose_anchor_for_s_backward(self, i, axes):
        for j in range(i+1, 6):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        for j in range(i-1, -1, -1):
            if abs(np.dot(axes[i], axes[j])) < self.axis_parallel_tol:
                return j
        return min(i+1, 5)

    @staticmethod
    def _solve_theta_eq6(K1, K2, K3):
        # (K1-K2)cos(2θ) + K3 sin(2θ) = -K2
        A = K1 - K2; B = K3; C = -K2
        R = float(math.hypot(A, B))
        if R < 1e-12:
            return [0.0], (A, B, C)
        rhs = float(np.clip(C / R, -1.0, 1.0))
        phi = float(math.atan2(B, A))
        alpha = float(math.acos(rhs))
        return [0.5 * (phi + alpha), 0.5 * (phi - alpha)], (A, B, C)

    def _define_planes_and_rotate_with_diag(self, p_prev, p_i, p_next, a_prev, d_prev):
        """
        Same as the paper's step, but returns rich diagnostics.
        """
        El = self._unit(a_prev)

        # Ev: p_prev -> p_i projected onto Π_prev
        v = p_i - p_prev
        v_perp = v - El * np.dot(El, v)
        if float(np.linalg.norm(v_perp)) < 1e-9:
            v_perp = self._any_perp(El)
        Ev = self._unit(v_perp)
        p_hat = p_prev + d_prev * Ev

        # s = p_next - p_prev
        s = p_next - p_prev
        ns = float(np.linalg.norm(s))
        if ns < 1e-12:
            diag = dict(kind="degenerate_s", v_dot_a=float(abs(np.dot(self._unit(p_hat-p_prev), El))))
            return p_hat, Ev, 0.0, diag

        s_u = s / ns
        t = np.cross(El, Ev)

        # K1, K2, K3
        K1 = float(np.dot(s_u, Ev))
        K2 = float(np.dot(El, Ev) * np.dot(s_u, El))
        K3 = float(np.dot(s_u, t))

        # Solve eq.(6)
        thetas, (A, B, C) = self._solve_theta_eq6(K1, K2, K3)
        best = None
        for th in thetas:
            u_rot = self._rodrigues(Ev, El, th)
            p_rot = p_prev + d_prev * u_rot
            cost = float(np.linalg.norm(p_rot - p_i))
            En = self._unit(
                math.cos(2*th) * Ev + (1 - math.cos(2*th)) * np.dot(El, Ev) * El + math.sin(2*th) * np.cross(El, Ev)
            )
            En_dot_s = float(abs(np.dot(En, s_u)))  # should be ~0
            v_dot_a = float(abs(np.dot(self._unit(p_rot - p_prev), El)))  # Π_prev orthogonality
            r_eq6 = float(A*math.cos(2*th) + B*math.sin(2*th) - C)  # should be ~0
            cand = dict(theta=float(th), cost=cost, En_dot_s=En_dot_s, v_dot_a=v_dot_a, A=float(A), B=float(B), C=float(C), r_eq6=r_eq6)
            if best is None or cost < best["cost"]:
                best = cand
                best_p, best_u = p_rot, u_rot

        assert best is not None
        diag = best
        return best_p, best_u, best["theta"], diag

    # ---------- FABRIK-R passes (instrumented) ----------
    def _forward_pass(self, p, axes, L, target_wc, trial_log):
        p[6] = target_wc.copy()
        for i in range(5, -1, -1):
            a_prev = axes[i]
            idx_for_s = self._choose_anchor_for_s_forward(i, axes)
            p_new, u_new, theta, diag = self._define_planes_and_rotate_with_diag(
                p_prev=p[i+1], p_i=p[i], p_next=p[idx_for_s], a_prev=a_prev, d_prev=L[i]
            )
            rec = dict(phase="F", i=i, center_idx=i+1, s_anchor=idx_for_s)
            rec.update(diag)               # diag 안에 theta 포함
            trial_log["steps"].append(rec)
            p[i] = p_new

    def _backward_pass(self, p, axes, L, base, trial_log):
        p[0] = base.copy()
        for i in range(0, 6):
            a_prev = axes[i]
            idx_for_s = self._choose_anchor_for_s_backward(i, axes)
            p_new, u_new, theta, diag = self._define_planes_and_rotate_with_diag(
                p_prev=p[i], p_i=p[i+1], p_next=p[idx_for_s], a_prev=a_prev, d_prev=L[i]
            )
            rec = dict(phase="B", i=i, center_idx=i, s_anchor=idx_for_s)
            rec.update(diag)               # diag 안에 theta 포함
            trial_log["steps"].append(rec)
            p[i+1] = p_new

    # ---------- q recovery ----------
    @staticmethod
    def _signed_angle_around_axis(v_from, v_to, axis):
        a = axis / (np.linalg.norm(axis) + 1e-12)
        vf = v_from - a * np.dot(a, v_from)
        vt = v_to   - a * np.dot(a, v_to)
        nf = float(np.linalg.norm(vf)); nt = float(np.linalg.norm(vt))
        if nf < 1e-12 or nt < 1e-12:
            return 0.0
        vf /= nf; vt /= nt
        s = float(np.dot(a, np.cross(vf, vt)))
        c = float(np.clip(np.dot(vf, vt), -1.0, 1.0))
        return float(math.atan2(s, c))

    def _positions_to_q(self, q_in, p_des):
        kin = self.kin
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
            q[i] = float(np.clip(q[i] + 0.8 * dtheta, kin.lower[i], kin.upper[i]))
            axes_world = self._axes_world(q)
        return q

    # ---------- main ----------
    def solve(self, target_pose: np.ndarray, q_seed=None):
        self.logs = []
        kin = self.kin
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())

        p = self._chain_positions(q)
        L = self._link_lengths(p)
        base = p[0].copy()

        p_t = target_pose[:3, 3]
        R_t = target_pose[:3, :3]
        target_wc = p_t - kin.d6 * R_t[:, 2] if abs(kin.d6) > 1e-6 else p_t

        if np.linalg.norm(target_wc - base) > (np.sum(L) + 1e-6):
            return kin.clamp(q), False

        ok = False
        for it in range(self.max_iter):
            axes = self._axes_world(q)
            p = self._chain_positions(q)

            trial_log = dict(iteration=it, steps=[])
            self._forward_pass(p, axes, L, target_wc, trial_log)
            self._backward_pass(p, axes, L, base, trial_log)
            self.logs.append(trial_log)

            q = self._positions_to_q(q, p)

            Tee, _ = kin.forward_kinematics(q)
            axes_now = self._axes_world(q)
            wc_now = Tee[:3, 3] - kin.d6 * axes_now[5] if abs(kin.d6) > 1e-9 else Tee[:3, 3]
            if float(np.linalg.norm(wc_now - target_wc)) < self.tol:
                ok = True
                break

        return kin.clamp(q), ok

# ----------------------------- Test Harness -----------------------------

def make_targets(kin, kind="random", n=30, seed=0):
    rng = np.random.default_rng(seed)
    # rough workspace estimate: sum of link lengths at q0
    q0 = np.zeros(6, dtype=float)
    T0, Ts0 = kin.forward_kinematics(q0)
    P = [T[:3,3] for T in Ts0] + [T0[:3,3]]
    Ls = [np.linalg.norm(P[i+1]-P[i]) for i in range(6)]
    R = float(sum(Ls))
    base = P[0]
    targets = []
    if kind == "random":
        for _ in range(n):
            # sample in a ball of radius 0.8 R centered somewhere reachable
            dirv = rng.normal(size=3); dirn = float(np.linalg.norm(dirv)); dirv = dirv / dirn if dirn>1e-9 else np.array([1,0,0],float)
            r = 0.2*R + 0.6*R * float(rng.random())
            p = base + r*dirv
            T = np.eye(4); T[:3,3] = p
            targets.append(T)
    elif kind == "circle":
        # circle in a plane offset from base
        center = base + np.array([0.3*R, 0.2*R, 0.1*R])
        normal = np.array([0,0,1],float)
        radius = 0.4*R
        for k in range(n):
            ang = 2*math.pi*k/max(1,n)
            # pick orthonormal basis
            if abs(normal[0])<0.9: a = np.array([1,0,0],float)
            else: a = np.array([0,1,0],float)
            u = np.cross(normal/np.linalg.norm(normal), a); u /= np.linalg.norm(u)
            v = np.cross(normal/np.linalg.norm(normal), u)
            p = center + radius*(math.cos(ang)*u + math.sin(ang)*v)
            T = np.eye(4); T[:3,3] = p
            targets.append(T)
    elif kind == "line":
        start = base + np.array([0.2*R, 0, 0])
        end   = base + np.array([0.8*R, 0.2*R, -0.1*R])
        for k in range(n):
            t = k/max(1,n-1)
            p = (1-t)*start + t*end
            T = np.eye(4); T[:3,3] = p
            targets.append(T)
    else:
        raise ValueError("unknown target kind")
    return targets

def evaluate_solver_pair(kin, user_solver_cls, ref_solver_cls, target_T, q_seed=None):
    # helpers
    def wrist_center(q):
        Tee, _ = kin.forward_kinematics(q)
        try:
            WORLD=2
            axes = kin.jacobian(q, ref_frame=WORLD)
        except Exception:
            import pinocchio as pin
            axes = kin.jacobian(q, ref_frame=pin.ReferenceFrame.WORLD)
        a6 = np.array(axes[3:,5]).reshape(3,)
        if abs(kin.d6) > 1e-9:
            return Tee[:3,3] - kin.d6 * (a6/np.linalg.norm(a6))
        return Tee[:3,3]

    # target wrist center
    p_t = target_T[:3,3]; R_t = target_T[:3,:3]
    target_wc = p_t - kin.d6 * R_t[:,2] if abs(kin.d6)>1e-9 else p_t

    # run user solver
    user_out = {"ok": False, "wc_err": None, "q": None, "exception": None}
    try:
        user_solver = user_solver_cls(kin)
        q_u, ok_u = user_solver.solve(target_T, q_seed=q_seed)
        wc_u = wrist_center(q_u)
        user_out.update(ok=bool(ok_u), wc_err=float(np.linalg.norm(wc_u - target_wc)), q=[float(x) for x in q_u])
    except Exception as e:
        user_out["exception"] = repr(e) + "\n" + traceback.format_exc()

    # run reference solver
    ref_out = {"ok": False, "wc_err": None, "q": None, "logs": None, "exception": None}
    try:
        ref_solver = ref_solver_cls(kin)
        q_r, ok_r = ref_solver.solve(target_T, q_seed=q_seed)
        wc_r = wrist_center(q_r)
        ref_out.update(ok=bool(ok_r), wc_err=float(np.linalg.norm(wc_r - target_wc)), q=[float(x) for x in q_r], logs=ref_solver.logs)
    except Exception as e:
        ref_out["exception"] = repr(e) + "\n" + traceback.format_exc()

    return user_out, ref_out

def run_diagnostics(args):
    # 1) import user's module and pick classes
    user_mod = import_by_path_or_name(args.user_module) if args.user_module else None
    if user_mod is None:
        print("ERROR: --user-module is required")
        sys.exit(2)

    # Find KinematicModel, JacobianIKSolver, and user's Fabrik class
    def find_class(mod, name):
        if hasattr(mod, name):
            return getattr(mod, name)
        # fallback: search by name in module attributes
        for k,v in mod.__dict__.items():
            if k.lower() == name.lower() and isinstance(v, type):
                return v
        raise AttributeError(f"Class {name} not found in module {mod.__name__}")

    KinCls = find_class(user_mod, args.kin_class)
    JacoCls = find_class(user_mod, args.jacobian_class)
    UserFabrikCls = find_class(user_mod, args.fabrik_class)

    # 2) build kinematics
    kin = KinCls()

    # 3) generate targets
    targets = []
    for kind in args.kinds:
        targets += make_targets(kin, kind=kind, n=args.trials_per_kind, seed=args.seed)

    # 4) loop trials
    results = []
    for idx, T in enumerate(targets):
        # optional: use jacobian to get a decent seed
        if args.use_jacobian_seed:
            try:
                jac = JacoCls(kin)
                qj, okj = jac.solve(T)
                q_seed = qj
            except Exception:
                q_seed = None
        else:
            q_seed = None

        user_out, ref_out = evaluate_solver_pair(kin, UserFabrikCls, RefFabrikRIKSolverInstrumented, T, q_seed=q_seed)

        results.append(dict(
            idx=idx,
            target=list(map(float, T[:3,3])),
            user=user_out,
            ref=ref_out,
        ))
        print(f"[{idx+1}/{len(targets)}] user_ok={user_out['ok']} err={user_out['wc_err']}, ref_ok={ref_out['ok']} err={ref_out['wc_err']}")

    # 5) write JSON + CSV summary
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True) if os.path.dirname(args.out_json) else None
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(dict(
            meta=dict(kinds=args.kinds, trials_per_kind=args.trials_per_kind, seed=args.seed, user_module=user_mod.__name__),
            results=results
        ), f, ensure_ascii=False, indent=2)
    print("Wrote JSON:", args.out_json)

    # Flat CSV with key metrics (one row per trial)
    csv_path = args.out_csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","tgt_x","tgt_y","tgt_z","user_ok","user_wc_err","ref_ok","ref_wc_err"])
        for r in results:
            t = r["target"]
            w.writerow([r["idx"], t[0], t[1], t[2], r["user"]["ok"], r["user"]["wc_err"], r["ref"]["ok"], r["ref"]["wc_err"]])
    print("Wrote CSV:", csv_path)

    # Also split out a lightweight per-iteration diagnostics file for the ref solver only
    if args.out_diag:
        diag = []
        for r in results:
            if r["ref"].get("logs"):
                diag.append(dict(idx=r["idx"], target=r["target"], logs=r["ref"]["logs"]))
        with open(args.out_diag, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
        print("Wrote per-iteration diagnostics:", args.out_diag)

def main():
    ap = argparse.ArgumentParser(description="Diagnose user's FABRIK solver vs reference FABRIK-R and Jacobian baseline.")
    ap.add_argument("--user-module", required=True, help="Module name or path to .py containing KinematicModel, JacobianIKSolver, FabrikRIKSolver")
    ap.add_argument("--kin-class", default="KinematicModel")
    ap.add_argument("--jacobian-class", default="JacobianIKSolver")
    ap.add_argument("--fabrik-class", default="FabrikRIKSolver")
    ap.add_argument("--kinds", nargs="+", default=["random","circle","line"], help="Target kinds to test")
    ap.add_argument("--trials-per-kind", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-jacobian-seed", action="store_true", help="Use Jacobian IK to obtain a seed q for each target")
    ap.add_argument("--out-json", default="fabrik_diag.json")
    ap.add_argument("--out-csv", default="fabrik_diag_summary.csv")
    ap.add_argument("--out-diag", default="fabrik_diag_per_iter.json")
    args = ap.parse_args()
    run_diagnostics(args)

if __name__ == "__main__":
    main()

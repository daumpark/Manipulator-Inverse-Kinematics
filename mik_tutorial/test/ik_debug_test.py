#!/usr/bin/env python3
# ik_debug_test.py

import os
import sys
import math
import numpy as np

try:
    from mik_tutorial.ik_solvers import (
        KinematicModel, AnalyticalIKSolver, JacobianIKSolver
    )
except Exception as e:
    print("[ERR] mik_tutorial.ik_solvers import 실패:", e)
    print(" -> build/설정 확인: colcon build && source install/setup.bash")
    sys.exit(1)

np.set_printoptions(precision=5, suppress=True)

def rad2deg(x): return np.rad2deg(x)
def deg2rad(x): return np.deg2rad(x)

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def angdiff(a, b):
    return wrap_angle(a - b)

def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return np.rad2deg(np.arccos(c))

def pose_err(Ta, Tb):
    pe = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])
    re = rot_err_deg(Ta[:3, :3], Tb[:3, :3])
    return pe, re

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def check_model_basics(kin: KinematicModel):
    print_header("Model / Frames / Constants")
    print(f"- URDF file         : {kin.urdf_file}")
    print(f"- #frames           : {kin.model.nframes}")
    print(f"- ee_frame_id       : {kin.ee_frame_id}")
    print(f"- joint_names (6)   : {kin.joint_names}")
    T0, Ts0 = kin.forward_kinematics(np.zeros(6))
    print(f"- FK(q=0) pos [m]   : {np.round(T0[:3,3], 4)}")
    print(f"- FK(q=0) R row0    : {np.round(T0[0,:3], 4)}")
    for name in ["d1", "L2", "L3", "L6"]:
        v = getattr(kin, name, None)
        print(f"- const {name:>2}       : {v:.6f}" if v is not None else f"- const {name} : <missing>")

def debug_analytical_arm_step(ana: AnalyticalIKSolver, p_wc):
    q1_raw, q2_raw, q3_raw, ok = ana._arm_3dof_from_wc(p_wc)
    print(f"  [arm_3dof] ok={ok}, q1/q2/q3 raw (deg) = {np.round(rad2deg([q1_raw,q2_raw,q3_raw]),2)}")
    return ok, q1_raw, q2_raw, q3_raw

def test_zero_pose_roundtrip(kin: KinematicModel):
    print_header("Zero Pose Roundtrip (FK(0) -> Analytical IK -> FK)")
    ana = AnalyticalIKSolver(kin)
    T0, _ = kin.forward_kinematics(np.zeros(6))
    q_sol, ok = ana.solve(T0)
    print(f"- analytical ok     : {ok}")
    if q_sol is None:
        print("  -> No solution returned.")
        return
    print(f"- q_sol (deg)       : {np.round(rad2deg(q_sol), 3)}")
    T_chk, _ = kin.forward_kinematics(q_sol)
    pe, re = pose_err(T_chk, T0)
    print(f"- FK error pos [m]  : {pe:.6f}")
    print(f"- FK error rot [deg]: {re:.4f}")

    R0 = T0[:3,:3]; p0 = T0[:3,3]
    p_wc0 = p0 - kin.L6 * R0[:,2]
    debug_analytical_arm_step(ana, p_wc0)

def test_random_joint_samples(kin: KinematicModel, n=10, seed=0):
    print_header(f"Random Joint Samples x{n}  (FK -> IK compare)")
    rng = np.random.default_rng(seed)
    lo, hi = kin.lower, kin.upper

    ana = AnalyticalIKSolver(kin)
    jac = JacobianIKSolver(kin)

    stats = {
        "ana_ok":0, "jac_ok":0,
        "ana_pos_mm": [], "ana_rot_deg": [],
        "jac_pos_mm": [], "jac_rot_deg": [],
        "q_rmse_deg_ana": [], "q_rmse_deg_jac": [],
        "c3_violations":0
    }

    for i in range(n):
        q_ref = rng.uniform(lo, hi)
        T_ref, _ = kin.forward_kinematics(q_ref)

        q_ana, ok_a = ana.solve(T_ref)
        q_jac, ok_j = jac.solve(T_ref)

        if q_ana is not None:
            Ta, _ = kin.forward_kinematics(q_ana)
            pe, re = pose_err(Ta, T_ref)
            stats["ana_pos_mm"].append(pe*1000.0)
            stats["ana_rot_deg"].append(re)
            dq = angdiff(q_ana, q_ref)
            rmse = float(np.sqrt(np.mean(np.square(rad2deg(dq)))))
            stats["q_rmse_deg_ana"].append(rmse)
        if q_jac is not None:
            Tj, _ = kin.forward_kinematics(q_jac)
            pe, re = pose_err(Tj, T_ref)
            stats["jac_pos_mm"].append(pe*1000.0)
            stats["jac_rot_deg"].append(re)
            dq = angdiff(q_jac, q_ref)
            rmse = float(np.sqrt(np.mean(np.square(rad2deg(dq)))))
            stats["q_rmse_deg_jac"].append(rmse)

        stats["ana_ok"] += int(bool(ok_a and (q_ana is not None)))
        stats["jac_ok"] += int(bool(ok_j and (q_jac is not None)))

        # arm 단계 코사인법칙 유효성 체크
        Rr = T_ref[:3,:3]; pr = T_ref[:3,3]
        p_wc = pr - kin.L6 * Rr[:,2]
        q1_raw, q2_raw, q3_raw, ok_arm = ana._arm_3dof_from_wc(p_wc)

        q_tmp = np.zeros(6); q_tmp[0]=q1_raw
        _, Ts = kin.forward_kinematics(q_tmp)
        p2=Ts[1][:3,3]; p3=Ts[2][:3,3]; p4=Ts[3][:3,3]
        L2=np.linalg.norm(p3-p2); L3=np.linalg.norm(p4-p3)
        u=(p3-p2)/np.linalg.norm(p3-p2)
        eps=1e-4
        q_eps=q_tmp.copy(); q_eps[1]=eps
        _,Ts_eps=kin.forward_kinematics(q_eps)
        p3e=Ts_eps[2][:3,3]; t=p3e-p3
        if np.linalg.norm(t)<1e-9:
            n=np.cross(u, np.array([0,0,1.0]))
            if np.linalg.norm(n)<1e-9: n=np.array([0,1.0,0])
            n=n/np.linalg.norm(n)
        else:
            t=t/np.linalg.norm(t); n=np.cross(u,t); n=n/np.linalg.norm(n)
        v=np.cross(n,u)
        w = p_wc - p2
        w_p = w - np.dot(w,n)*n
        r = np.linalg.norm(w_p)
        c3 = (r*r - L2*L2 - L3*L3) / (2.0*L2*L3)
        if not (-1.0-1e-6 <= c3 <= 1.0+1e-6):
            stats["c3_violations"] += 1
            print(f"[{i:02d}] c3 out of range: {c3:.4f} | r={r:.4f}, L2={L2:.4f}, L3={L3:.4f}")

        # 안전한 문자열 조합 (중첩 f-string 회피)
        ana_pair = "NA" if q_ana is None else f"{stats['ana_pos_mm'][-1]:.2f}/{stats['ana_rot_deg'][-1]:.2f}"
        jac_pair = "NA" if q_jac is None else f"{stats['jac_pos_mm'][-1]:.2f}/{stats['jac_rot_deg'][-1]:.2f}"
        print(f"[{i:02d}] ANA ok={ok_a}  JAC ok={ok_j}  "
              f"ANA pos(mm)/rot(deg)={ana_pair}  JAC pos(mm)/rot(deg)={jac_pair}")

    def smean(x): return float(np.mean(x)) if len(x)>0 else float('nan')
    def smax(x):  return float(np.max(x))  if len(x)>0 else float('nan')

    print_header("Summary")
    print(f"- Analytical success   : {stats['ana_ok']}/{n}")
    print(f"- Jacobian success     : {stats['jac_ok']}/{n}")
    print(f"- ANA pos err  mean/max (mm): {smean(stats['ana_pos_mm']):.3f} / {smax(stats['ana_pos_mm']):.3f}")
    print(f"- ANA rot err  mean/max (deg): {smean(stats['ana_rot_deg']):.3f} / {smax(stats['ana_rot_deg']):.3f}")
    print(f"- JAC pos err  mean/max (mm): {smean(stats['jac_pos_mm']):.3f} / {smax(stats['jac_pos_mm']):.3f}")
    print(f"- JAC rot err  mean/max (deg): {smean(stats['jac_rot_deg']):.3f} / {smax(stats['jac_rot_deg']):.3f}")
    print(f"- ANA joint RMSE mean/max (deg): {smean(stats['q_rmse_deg_ana']):.3f} / {smax(stats['q_rmse_deg_ana']):.3f}")
    print(f"- JAC joint RMSE mean/max (deg): {smean(stats['q_rmse_deg_jac']):.3f} / {smax(stats['q_rmse_deg_jac']):.3f}")
    print(f"- Cosine-law violations (c3 out of [-1,1]): {stats['c3_violations']}")

def main():
    print_header("IK Debug Tester")
    print(f"Python {sys.version.split()[0]}")
    try:
        kin = KinematicModel()
    except Exception as e:
        print("[ERR] KinematicModel 초기화 실패:", e)
        sys.exit(1)

    check_model_basics(kin)
    test_zero_pose_roundtrip(kin)
    test_random_joint_samples(kin, n=12, seed=42)

if __name__ == "__main__":
    main()

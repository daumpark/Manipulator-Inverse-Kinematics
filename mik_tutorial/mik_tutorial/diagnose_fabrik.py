#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import os
import time
import math
import json
import csv
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

# Make sure we can import the user's solver module (this file is placed next to ik_solvers.py)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from ik_solvers import KinematicModel, JacobianIKSolver, FABRIKSolver

@dataclass
class SolveSummary:
    solver: str
    success: bool
    iters: int
    pos_err: float
    ori_err: float
    time_ms: float

def se3_pos_ori_error(T_cur: np.ndarray, T_tar: np.ndarray) -> Tuple[float,float]:
    dp = T_cur[:3, 3] - T_tar[:3, 3]
    R_err = T_cur[:3, :3] @ T_tar[:3, :3].T
    tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(tr))
    return float(np.linalg.norm(dp)), ang

def sample_reachable_pose(kin: KinematicModel, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    # sample q uniformly in limits (finite only), fallback to [-pi, pi]
    lo, hi = kin.lower.copy(), kin.upper.copy()
    lo[~np.isfinite(lo)] = -math.pi
    hi[~np.isfinite(hi)] =  math.pi
    q = np.array([rng.uniform(float(a), float(b)) for a,b in zip(lo, hi)], dtype=float)
    T, _ = kin.forward_kinematics(q)
    return q, T

def check_wrist_offset_invariance(kin: KinematicModel, rng: random.Random, trials: int = 50) -> float:
    # Recompute r_ee_to_j6_ee across random q and measure max deviation from stored value
    base_val = kin.r_ee_to_j6_ee.copy()
    diffs = []
    for _ in range(trials):
        q,_ = sample_reachable_pose(kin, rng)
        kin._full_fk(q)
        j6_name = kin.joint_names[-1]
        j6_id = kin.model.getJointId(j6_name)
        Tj6 = kin.data.oMi[j6_id]
        Tee = kin.data.oMf[kin.ee_frame_id]
        r_world = Tj6.translation - Tee.translation
        R_ee = Tee.rotation
        r_ee = R_ee.T @ r_world
        diffs.append(np.linalg.norm(r_ee - base_val))
    return float(np.max(diffs))

def run_solver(kin: KinematicModel, solver_name: str, T_target: np.ndarray, q_seed: np.ndarray, params: dict) -> Tuple[SolveSummary, dict, np.ndarray]:
    if solver_name == "jacobian":
        solver = JacobianIKSolver(kin)
        # optional overrides
        for k,v in params.items():
            if hasattr(solver, k):
                setattr(solver, k, v)
    else:
        solver = FABRIKSolver(kin)
        for k,v in params.items():
            if hasattr(solver, k):
                setattr(solver, k, v)

    t0 = time.time()
    q_sol, ok = solver.solve(T_target, q_seed=q_seed)
    ms = 1000.0 * (time.time() - t0)

    T_sol, _ = kin.forward_kinematics(q_sol)
    pe, oe = se3_pos_ori_error(T_sol, T_target)

    dbg = getattr(solver, "debug", {}) or {}
    iters = dbg.get("iters", 0) or len(dbg.get("pos_errs", []))
    summary = SolveSummary(
        solver = solver.__class__.__name__,
        success = bool(ok),
        iters = int(iters),
        pos_err = float(pe),
        ori_err = float(oe),
        time_ms = float(ms),
    )
    return summary, dbg, q_sol

def deep_dive_report(dbg: dict) -> List[dict]:
    # Flatten per-iteration traces into a table for CSV
    rows = []
    pos = dbg.get("pos_errs", [])
    ori  = dbg.get("ori_errs", [])
    wrist = dbg.get("wrist_errs", [])
    gates = dbg.get("gates", [])
    n = max(len(pos), len(ori), len(wrist), len(gates))
    for i in range(n):
        rows.append({
            "iter": i+1,
            "pos_err": float(pos[i]) if i < len(pos) else float("nan"),
            "ori_err": float(ori[i]) if i < len(ori) else float("nan"),
            "wrist_err": float(wrist[i]) if i < len(wrist) else float("nan"),
            "gate_on": int(gates[i]) if i < len(gates) else 0,
        })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Diagnose FABRIK vs Jacobian IK on PiPER")
    ap.add_argument("--N", type=int, default=20, help="number of random targets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fabrik_pos_only", action="store_true", help="disable orientation gating (position-only test)")
    ap.add_argument("--all_hinges", action="store_true", help="override joint roles as all hinges for FABRIK")
    ap.add_argument("--out", type=str, default="diagnose_results")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    kin = KinematicModel()

    print(f"[Info] Using EE frame id: {kin.ee_frame_id} (name wanted: 'link6'), joint6 name: {kin.joint_names[-1]}")
    # quick consistency check on wrist offset
    worst = check_wrist_offset_invariance(kin, rng, trials=50)
    print(f"[Check] r_ee_to_j6_ee invariance max deviation across 50 q's: {worst*1000:.3f} mm")

    if args.all_hinges:
        kin.set_joint_roles(["hinge"]*6)
        print("[Info] Overriding joint roles to all hinges for FABRIK.")

    # parameter tweaks for FABRIK in pos-only mode
    fabrik_overrides = {}
    if args.fabrik_pos_only:
        fabrik_overrides["orient_gate_mul"] = 1e9  # keep orientation gate off
        fabrik_overrides["smooth_q"] = 0.0
        fabrik_overrides["relax_pos"] = 0.0
        fabrik_overrides["q_reg"] = 0.0

    summaries: List[SolveSummary] = []
    failing_cases: List[int] = []

    for ti in range(args.N):
        q0, Ttar = sample_reachable_pose(kin, rng)

        # Run both solvers from the same seed
        sum_j, dbg_j, qj = run_solver(kin, "jacobian", Ttar, q_seed=q0, params={})
        sum_f, dbg_f, qf = run_solver(kin, "fabrik", Ttar, q_seed=q0, params=fabrik_overrides)

        summaries.extend([sum_j, sum_f])

        print(f"[{ti:03d}] JAC pe={sum_j.pos_err*1000:.2f}mm, oe={math.degrees(sum_j.ori_err):.2f}deg, ok={sum_j.success}, it={sum_j.iters}")
        print(f"      FAB pe={sum_f.pos_err*1000:.2f}mm, oe={math.degrees(sum_f.ori_err):.2f}deg, ok={sum_f.success}, it={sum_f.iters}")

        # If FABRIK clearly worse, store deep dive CSV
        if sum_f.pos_err > sum_j.pos_err * 1.5 or not sum_f.success:
            failing_cases.append(ti)
            csv_path = os.path.join(args.out, f"case_{ti:03d}_fabrik_trace.csv")
            rows = deep_dive_report(dbg_f)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["iter","pos_err","ori_err","wrist_err","gate_on"])
                w.writeheader()
                w.writerows(rows)
            # also dump final chain geometry (last iteration)
            geom_json = os.path.join(args.out, f"case_{ti:03d}_chain.json")
            chain = dbg_f.get("final_chain_p", None)
            if chain is not None:
                data = {"p": [list(map(float, list(p))) for p in chain]}
                with open(geom_json, "w") as f:
                    json.dump(data, f, indent=2)

    # Save overall summary CSV
    sum_csv = os.path.join(args.out, "summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SolveSummary.__annotations__.keys()))
        w.writeheader()
        for s in summaries:
            w.writerow(asdict(s))

    print(f"\n[Done] Saved summary to: {sum_csv}")
    if failing_cases:
        print(f"[Hint] Wrote deep-dive traces for failing cases: {sorted(failing_cases)} in folder '{args.out}'")

if __name__ == "__main__":
    main()

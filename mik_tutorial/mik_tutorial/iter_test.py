import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# ik_solvers.py 파일이 같은 폴더에 있다고 가정합니다.
from ik_solvers import KinematicModel, FABRIKSolver

# 정체 탈출 로직이 없는, chain_history만 기록하는 솔버
class ModifiedFABRIKSolver(FABRIKSolver):
    def solve(self, target_pose: np.ndarray, q_seed=None):
        kin = self.kinematics
        q = kin.clamp(self.q0.copy() if q_seed is None else np.asarray(q_seed, float).copy())
        p, x, y, z, L, axes, is_pris = self._frame_arrays(q)
        base_p = p[0].copy()
        _, Ts0 = kin.forward_kinematics(q)
        base_R = Ts0[0][:3, :3].copy()
        p_t = target_pose[:3, 3].copy()
        R_t = target_pose[:3, :3].copy()
        r_ee_to_j6 = getattr(kin, "r_ee_to_j6_ee", np.zeros(3))
        target_wc = p_t + R_t @ r_ee_to_j6

        # Reachability Clamp
        L_arr = np.array(L, dtype=float)
        r_max = float(np.sum(L_arr)) - 1e-9
        v_wc = target_wc - base_p
        d_wc = float(np.linalg.norm(v_wc))
        if d_wc > r_max:
            target_wc = base_p + v_wc * (r_max / (d_wc + 1e-12))

        # 디버그용 변수
        ee_positions, pos_errs, ori_errs, wrist_errs, gates = [], [], [], [], []
        chain_history = [np.array([pi.copy() for pi in p])]

        def eval_all(qv, consider_ori=True):
            Tee, _ = kin.forward_kinematics(qv)
            ee_pos = Tee[:3, 3]
            dp = float(np.linalg.norm(ee_pos - p_t))
            R_err = Tee[:3, :3] @ R_t.T
            tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
            ang = float(np.arccos(tr))
            wc_cur = ee_pos + Tee[:3, :3] @ r_ee_to_j6
            werr = float(np.linalg.norm(wc_cur - target_wc))
            cost = dp + (ang if consider_ori else 0.0)
            return ee_pos.copy(), dp, ang, cost, werr

        ee_pos, pe, oe, _, werr = eval_all(q, consider_ori=False)
        fix_orientation = (werr < self.orient_gate_mul * self.tol_pos)
        _, _, _, cost_prev, _ = eval_all(q, consider_ori=fix_orientation)
        ee_positions.append(ee_pos); pos_errs.append(pe); ori_errs.append(oe)

        ok = False
        for _it in range(self.max_iter):
            q_prev_iter = q.copy()
            fix_orientation = (werr < self.orient_gate_mul * self.tol_pos)
            types = kin.joint_roles
            
            self._forward_stage(p, x, y, z, L, types, target_wc, R_t if fix_orientation else None, fix_orientation)
            chain_history.append(np.array([pi.copy() for pi in p]))
            
            self._backward_stage(p, x, y, z, L, types, base_p, base_R)
            chain_history.append(np.array([pi.copy() for pi in p]))

            # q 업데이트
            q_prop = self._positions_to_q(q_prev_iter, p, gain=self.q_gain)
            dq = q_prop - q_prev_iter
            dq = np.clip(dq, -np.deg2rad(self.max_step_deg), np.deg2rad(self.max_step_deg))
            q_step = kin.clamp(q_prev_iter + dq)
            if self.smooth_q > 0.0:
                 q_try = kin.clamp((1.0 - self.smooth_q) * q_step + self.smooth_q * q_prev_iter)
            else:
                 q_try = q_step

            # 평가 및 적용
            ee_pos_try, pe_try, oe_try, cost_now, werr_try = eval_all(q_try, consider_ori=fix_orientation)
            if cost_now < cost_prev:
                q = q_try
                cost_prev = cost_now
                werr = werr_try
                pe = pe_try
                oe = oe_try
            
            # 디버그 정보 기록
            ee_pos, _, _, _, _ = eval_all(q, consider_ori=fix_orientation)
            ee_positions.append(ee_pos)
            pos_errs.append(pe)
            ori_errs.append(oe)

            # 수렴 체크
            if (not fix_orientation and pe < self.tol_pos) or (fix_orientation and pe < self.tol_pos and oe < self.tol_rot):
                ok = True
                break
            
            p, x, y, z, L, axes, is_pris = self._frame_arrays(q)

        self.debug = {"iters": len(ee_positions), "ee_positions": ee_positions, "pos_errs": pos_errs, "ori_errs": ori_errs, "chain_history": chain_history, "final_chain_p": p}
        return kin.clamp(q), ok

def animate_fabrik_process_with_error(solver: ModifiedFABRIKSolver, target_pose: np.ndarray):
    """
    FABRIK 반복 과정을 애니메이션으로 시각화하며, 매 Backward 단계마다 오차를 표시합니다.
    """
    if not hasattr(solver, 'debug') or solver.debug is None or "chain_history" not in solver.debug:
        print("애니메이션을 위한 디버그 정보(chain_history)가 없습니다.")
        return

    chain_history = solver.debug["chain_history"]
    pos_errs = solver.debug["pos_errs"]
    ori_errs = solver.debug["ori_errs"]
    target_pos = target_pose[:3, 3]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 축 범위 설정
    all_points = np.vstack(chain_history)
    all_points = np.vstack([all_points, target_pos.reshape(1, -1)])
    max_range = np.ptp(all_points, axis=0).max() / 2.0 * 1.1
    mid_point = np.mean(all_points, axis=0)
    ax.set_xlim(mid_point[0] - max_range, mid_point[0] + max_range)
    ax.set_ylim(mid_point[1] - max_range, mid_point[1] + max_range)
    ax.set_zlim(mid_point[2] - max_range, mid_point[2] + max_range)
    
    # 고정 요소들
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='red', marker='*', s=250, label='Target', depthshade=False)
    line, = ax.plot([], [], [], 'o-', color='blue', markersize=6, lw=2, label='Robot Arm')
    text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    def init():
        line.set_data_3d([], [], [])
        text.set_text('')
        return line, text

    def update(frame):
        chain_p = chain_history[frame]
        line.set_data_3d(chain_p[:, 0], chain_p[:, 1], chain_p[:, 2])
        
        iteration = (frame + 1) // 2
        stage_text = ""
        error_text = ""

        if frame == 0:
            stage_text = "Start"
            p_err = pos_errs[0]
            o_err_deg = np.rad2deg(ori_errs[0])
            error_text = f"\nPos Err: {p_err:.4f} m\nOri Err: {o_err_deg:.2f} deg"
        elif frame % 2 != 0:
            stage_text = f"Iter: {iteration} / Forward"
        else: # Backward
            stage_text = f"Iter: {iteration} / Backward"
            if iteration < len(pos_errs):
                p_err = pos_errs[iteration]
                o_err_deg = np.rad2deg(ori_errs[iteration])
                error_text = f"\nPos Err: {p_err:.4f} m\nOri Err: {o_err_deg:.2f} deg"

        text.set_text(stage_text + error_text)
        return line, text

    ani = FuncAnimation(fig, update, frames=len(chain_history),
                        init_func=init, blit=True, interval=300, repeat=True)

    ax.set_title('FABRIK Iteration Process with Error Display')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("KinematicModel 로딩 중...")
    try:
        kinematics = KinematicModel()
        print("PiPER 로봇 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"KinematicModel 초기화 중 오류 발생: {e}")
        exit()

    fabrik_solver = ModifiedFABRIKSolver(kinematics)
    target_pose = pin.SE3(np.eye(3), np.array([0.1, -0.1, 0.15])).homogeneous

    print("\nFABRIK IK 계산 시작...")
    q_solution, success = fabrik_solver.solve(target_pose, q_seed=fabrik_solver.q_mid)

    print(f"\nIK 계산 완료. 성공: {success}")
    if fabrik_solver.debug:
        final_pos_error = fabrik_solver.debug.get('pos_errs', [-1])[-1]
        print(f"  - 최종 위치 오차 (m): {final_pos_error:.6f}")

    print("\n애니메이션 시각화 함수 호출...")
    animate_fabrik_process_with_error(fabrik_solver, target_pose)
import numpy as np
import pinocchio as pin
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math
import re

# ================================================================
#  Kinematic Model (URDF Loader using Pinocchio)
# ================================================================
class KinematicModel:
    def __init__(self, urdf_filename):
        if not os.path.exists(urdf_filename):
            raise FileNotFoundError(f"URDF not found: {urdf_filename}")

        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_filename)
        self.model = self.robot.model
        self.data = self.robot.data
        print(f"✅ URDF '{os.path.basename(urdf_filename)}' loaded successfully.")
        print(f"Model Name: {self.model.name}, Movable Joints: {self.model.njoints - 1}")


# ================================================================
#  Interactive Constrained FABRIK Solver (Revolute-only limits)
# ================================================================
class InteractiveConstrainedFABRIKSolver:
    def __init__(self, kin_model):
        self.kin_model = kin_model
        self._extract_chain_from_model()  # sets: initial_positions, joint_axes_world, kinds, ids, + hinge bases/limits

        print(self.joint_kinds)

        self.base_position = np.zeros(3, dtype=float)
        self.joint_positions = np.copy(self.initial_positions)
        self.num_joints = len(self.joint_positions)

        # link lengths (base->j0, j0->j1, ..., j{n-2}->j{n-1})
        self.link_lengths = []
        self.link_lengths.append(np.linalg.norm(self.initial_positions[0] - self.base_position))
        for i in range(self.num_joints - 1):
            self.link_lengths.append(np.linalg.norm(self.initial_positions[i + 1] - self.initial_positions[i]))

        # neutral parent directions for axis transport
        parent_dir0 = []
        parent_dir0.append(self._safe_normalize(self.initial_positions[0] - self.base_position))
        for i in range(self.num_joints - 1):
            d = self.initial_positions[i + 1] - self.initial_positions[i]
            parent_dir0.append(self._safe_normalize(d))
        self.parent_dir0 = np.stack(parent_dir0, axis=0).astype(float)

        self.total_length = float(np.sum(self.link_lengths))
        self.history = []
        self.current_step = 0
        self.last_yaw_angle = 0.0

        print("\n🤖 Interactive Constrained FABRIK Solver Initialized.")
        print(f"Total Arm Length: {self.total_length:.3f} m")

    # --------------------------- Utilities ---------------------------
    @staticmethod
    def _norm(v):
        return float(np.linalg.norm(v))

    @staticmethod
    def _safe_normalize(v, fallback=None):
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
        axis = InteractiveConstrainedFABRIKSolver._safe_normalize(axis)
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(axis, ref)
        return InteractiveConstrainedFABRIKSolver._safe_normalize(perp)

    @staticmethod
    def _rot_between(u0, u):
        a = InteractiveConstrainedFABRIKSolver._safe_normalize(u0).astype(float)
        b = InteractiveConstrainedFABRIKSolver._safe_normalize(u).astype(float)
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = float(np.dot(a, b))
        if s < 1e-12:
            if c > 0:
                return np.eye(3)
            axis = InteractiveConstrainedFABRIKSolver._any_perp(a)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=float)
            return np.eye(3) + 2 * K @ K
        axis = v / s
        theta = math.atan2(s, c)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=float)
        return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    @staticmethod
    def _transport_axis_by_parent_dir(axis_neutral_world, parent_dir0, parent_dir_now):
        R = InteractiveConstrainedFABRIKSolver._rot_between(
            np.asarray(parent_dir0, dtype=float).reshape(3,),
            np.asarray(parent_dir_now, dtype=float).reshape(3,)
        )
        ax = R @ np.asarray(axis_neutral_world, dtype=float).reshape(3,)
        n = np.linalg.norm(ax)
        return ax / (n if n > 1e-12 else 1.0)

    # --------------------------- Chain extraction & limits ---------------------------
    def _extract_chain_from_model(self):
        model, data = self.kin_model.model, self.kin_model.data
        q_neutral = pin.neutral(model)
        pin.forwardKinematics(model, data, q_neutral)

        positions, joint_axes_world, joint_kinds, joint_model_ids, unsupported = [], [], [], [], []

        for jid in range(1, model.njoints):
            j = model.joints[jid]
            if j.nq == 0:
                continue  # fixed
            pos_w = data.oMi[jid].translation.copy()
            R_w = data.oMi[jid].rotation.copy()

            short = j.shortname().upper()
            kind, axis_local = None, None

            # ---- revolute only ----
            if ('REVOLUTE' in short) or ('RX' in short) or ('RY' in short) or ('RZ' in short):
                kind = 'hinge'
                got_axis = False
                try:
                    a = np.array(j.axis, dtype=float).reshape(3,)
                    if np.linalg.norm(a) > 1e-12:
                        axis_local, got_axis = a / np.linalg.norm(a), True
                except Exception:
                    pass
                if not got_axis:
                    if 'RX' in short:
                        axis_local = np.array([1.0, 0.0, 0.0])
                    elif 'RY' in short:
                        axis_local = np.array([0.0, 1.0, 0.0])
                    elif 'RZ' in short:
                        axis_local = np.array([0.0, 0.0, 1.0])
                    else:
                        kind = 'unknown'
            elif 'SPHERICAL' in short or 'PRISMATIC' in short or 'PX' in short or 'PY' in short or 'PZ' in short:
                kind = 'unsupported'
                unsupported.append(j.name)
            else:
                kind = 'unsupported'
                unsupported.append(j.name)

            positions.append(pos_w)

            if kind == 'hinge' and axis_local is not None:
                axis_world = R_w @ axis_local
                n = np.linalg.norm(axis_world)
                axis_world = axis_world / (n if n > 1e-12 else 1.0)
            else:
                axis_world = None

            joint_axes_world.append(axis_world)
            joint_kinds.append(kind)
            joint_model_ids.append(jid)

        self.initial_positions = np.array(positions, dtype=float)
        # keep dtype=object to allow None's, but cast to float whenever using it
        self.joint_axes_world = np.array(joint_axes_world, dtype=object)
        self.joint_kinds = joint_kinds
        self.joint_model_ids = joint_model_ids

        if unsupported:
            print(f"⚠️ Non-revolute joints detected (ignored/treated as fixed in limits): {unsupported}")

        # ---- store per-hinge joint limits and neutral plane bases (u0, v0) ----
        nj = len(self.initial_positions)
        self.hinge_limits = [None] * nj
        self.hinge_u0     = [None] * nj
        self.hinge_v0     = [None] * nj

        for jidx in range(nj):
            # limits: 그대로 유지
            if self.joint_kinds[jidx] == 'hinge' and self.joint_axes_world[jidx] is not None:
                jid   = self.joint_model_ids[jidx]
                jmodel = model.joints[jid]
                if getattr(jmodel, 'nq', 0) == 1:
                    iq = jmodel.idx_q
                    lo = float(model.lowerPositionLimit[iq])
                    hi = float(model.upperPositionLimit[iq])
                    self.hinge_limits[jidx] = None if ((not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi)) else (lo, hi)

            # >>> (FIX) plane basis u0, v0는 "현재→자식"으로 계산
            if (self.joint_kinds[jidx] == 'hinge' and
                self.joint_axes_world[jidx] is not None and
                jidx < nj - 1):                       # 마지막 관절은 자식 없음
                axis = np.asarray(self.joint_axes_world[jidx], dtype=float).reshape(3,)
                p_parent = self.initial_positions[jidx]
                p_child  = self.initial_positions[jidx + 1]
                d0 = p_child - p_parent               # 부모(=해당 힌지 위치) → 자식 관절
                d0_plane = d0 - np.dot(d0, axis) * axis
                if np.linalg.norm(d0_plane) < 1e-12:
                    u0 = self._any_perp(axis)
                else:
                    u0 = self._safe_normalize(d0_plane)
                u0 = np.asarray(u0, dtype=float).reshape(3,)
                v0 = self._safe_normalize(np.cross(axis, u0))
                self.hinge_u0[jidx] = u0
                self.hinge_v0[jidx] = v0
            else:
                self.hinge_u0[jidx] = None
                self.hinge_v0[jidx] = None

    # --------------------------- FABRIK helpers ---------------------------
    def _backward_global(self, chain, target):
        out = np.copy(chain)
        out[-1] = target
        for k in range(len(out) - 2, -1, -1):
            v = out[k] - out[k + 1]
            dir_v = self._safe_normalize(v)
            out[k] = out[k + 1] + self.link_lengths[k + 1] * dir_v
        return out

    def _fabrik_subchain(self, chain, anchor_idx, target):
        out = np.copy(chain)
        out[-1] = target
        for k in range(len(out) - 2, anchor_idx, -1):
            v = out[k] - out[k + 1]
            dir_v = self._safe_normalize(v)
            out[k] = out[k + 1] + self.link_lengths[k + 1] * dir_v
        for k in range(anchor_idx, len(out) - 1):
            v = out[k + 1] - out[k]
            dir_v = self._safe_normalize(v)
            out[k + 1] = out[k] + self.link_lengths[k + 1] * dir_v
        return out

    def _get_current_dynamic_axes(self, joint_positions):
        dynamic_axes = []
        for j in range(self.num_joints):
            axis_neutral_world = self.joint_axes_world[j]
            if self.joint_kinds[j] != 'hinge' or axis_neutral_world is None:
                dynamic_axes.append(None)
                continue
            parent_pos = self.base_position if j == 0 else joint_positions[j - 1]
            parent_dir_now = joint_positions[j] - parent_pos
            parent_dir0_seg = self.parent_dir0[j]
            axis_dyn = self._transport_axis_by_parent_dir(
                axis_neutral_world,
                parent_dir0_seg,
                parent_dir_now
            )
            dynamic_axes.append(axis_dyn)
        return dynamic_axes

    def _angwrap(self, a):
        return (a + math.pi) % (2*math.pi) - math.pi

    def _angdist(self, a, b):
        return abs(self._angwrap(a - b))

    def _clamp_angle_to_limits(self, phi, lohi):
        if lohi is None:
            return phi
        lo, hi = float(lohi[0]), float(lohi[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or (hi - lo >= 2*math.pi - 1e-9):
            return phi
        # 이미 범위 안이면 그대로
        if lo <= phi <= hi:
            return phi
        # 바깥이면 원둘레 기준으로 더 가까운 경계 선택
        return lo if self._angdist(phi, lo) <= self._angdist(phi, hi) else hi

    # --------------------------- (NEW) plane basis transport & projection with limits ---------------------------
    def _transport_plane_basis(self, jidx, parent_dir_now):
        """
        Rotate neutral (u0, v0, axis) of hinge jidx from neutral parent_dir0 -> current parent_dir_now.
        """
        axis0 = self.joint_axes_world[jidx]
        u0 = self.hinge_u0[jidx]
        v0 = self.hinge_v0[jidx]
        if axis0 is None or u0 is None or v0 is None:
            return None, None, None

        axis0 = np.asarray(axis0, dtype=float).reshape(3,)
        u0 = np.asarray(u0, dtype=float).reshape(3,)
        # v0는 사용하진 않지만 형식상 캐스팅
        _ = np.asarray(v0, dtype=float).reshape(3,)

        R = self._rot_between(self.parent_dir0[jidx], parent_dir_now)
        axis_now = self._safe_normalize(R @ axis0)
        u_now = self._safe_normalize(R @ u0)
        v_now = self._safe_normalize(np.cross(axis_now, u_now))  # re-orthogonalize
        return u_now, v_now, axis_now

    def _project_to_feasible_with_limits(self, jidx, parent_pos, desired_pos, link_len, parent_dir_now, prev_dir=None):
        """
        Revolute hinge: project target onto hinge plane, compute angle φ in (u_now,v_now),
        clamp φ to [lo, hi], reconstruct direction and return new joint position.
        Non-hinge: simple normalize along desired_pos-parent_pos.
        """
        if (self.joint_kinds[jidx] != 'hinge') or (self.joint_axes_world[jidx] is None):
            v = np.asarray(desired_pos - parent_pos, dtype=float)
            dir_vec = self._safe_normalize(v, prev_dir)
            return np.asarray(parent_pos, dtype=float) + link_len * dir_vec

        u_now, v_now, axis_now = self._transport_plane_basis(jidx, parent_dir_now)
        if u_now is None or v_now is None or axis_now is None:
            v = np.asarray(desired_pos - parent_pos, dtype=float)
            dir_vec = self._safe_normalize(v, prev_dir)
            return np.asarray(parent_pos, dtype=float) + link_len * dir_vec

        v = np.asarray(desired_pos - parent_pos, dtype=float)
        # project onto hinge plane
        v_plane = v - np.dot(v, axis_now) * axis_now
        if self._norm(v_plane) < 1e-12:
            if prev_dir is not None and self._norm(prev_dir) > 1e-12:
                prev_dir = np.asarray(prev_dir, dtype=float).reshape(3,)
                v_plane = prev_dir - np.dot(prev_dir, axis_now) * axis_now
        if self._norm(v_plane) < 1e-12:
            v_plane = u_now
        dir_on_plane = self._safe_normalize(v_plane)

        # angle in current plane basis
        x = float(np.dot(dir_on_plane, u_now))
        y = float(np.dot(dir_on_plane, v_now))
        phi = math.atan2(y, x)

        # clamp
        phi_c = self._clamp_angle_to_limits(phi, self.hinge_limits[jidx])

        dir_clamped = self._safe_normalize(u_now * math.cos(phi_c) + v_now * math.sin(phi_c))
        return np.asarray(parent_pos, dtype=float) + link_len * dir_clamped

    # --------------------------- Main solve ---------------------------
    def solve(self, target_position, tolerance=0.01, max_iterations=20):
        self.target = np.array(target_position, dtype=float)
        self.joint_positions = np.copy(self.initial_positions)
        self.history = []

        # initial axes (dynamic)
        start_axes = self._get_current_dynamic_axes(self.joint_positions)
        self.history.append({
            'title': f'Iter 1 (Start)',
            'positions': np.copy(self.joint_positions),
            'axes': start_axes,
            'error': self._norm(self.joint_positions[-1] - self.target)
        })

        for it in range(1, max_iterations + 1):
            if self.history[-1]['error'] <= tolerance:
                if it > 1:
                    print(f"\n✅ Target reached in {it - 1} iterations.")
                break

            # backward (global) step
            back = self._backward_global(self.joint_positions, self.target)
            self.history.append({
                'title': f'Iter {it} (Backward)',
                'positions': np.copy(back),
                'axes': [None] * self.num_joints,
                'error': self._norm(back[-1] - self.target)
            })

            # forward with constraints & limits
            cur = np.copy(back)
            cur[0] = self.initial_positions[0].copy()

            current_dynamic_axes = [None] * self.num_joints
            current_dynamic_axes[0] = self._get_current_dynamic_axes(cur)[0]  # first axis (after fixing j0 pos)

            for j in range(1, self.num_joints):
                parent = cur[j - 1]
                ideal = back[j]
                link_len = self.link_lengths[j]

                # compute current parent link direction for transport
                grandparent_pos = self.base_position if (j - 1) == 0 else cur[j - 2]
                link_to_parent_dir_now = parent - grandparent_pos
                link_to_parent_dir0 = self.parent_dir0[j - 1]

                # dynamic axis for visualization (keep original function)
                axis_neutral_world = self.joint_axes_world[j - 1]
                axis_dyn = None
                if (self.joint_kinds[j - 1] == 'hinge') and (axis_neutral_world is not None):
                    axis_dyn = self._transport_axis_by_parent_dir(
                        axis_neutral_world,
                        link_to_parent_dir0,
                        link_to_parent_dir_now
                    )
                current_dynamic_axes[j - 1] = axis_dyn

                # ==== projection with joint limits (parent joint j-1 controls link j) ====
                projected_pos = self._project_to_feasible_with_limits(
                    jidx=j - 1,
                    parent_pos=parent,
                    desired_pos=ideal,
                    link_len=link_len,
                    parent_dir_now=link_to_parent_dir_now,
                    prev_dir=cur[j] - parent
                )
                cur[j] = projected_pos

                self.history.append({
                    'title': f'Iter {it} (Project j={j + 1})',
                    'positions': np.copy(cur),
                    'axes': list(current_dynamic_axes),
                    'error': self._norm(cur[-1] - self.target)
                })

                if j < self.num_joints - 1:
                    # subchain refinement then recompute axes
                    cur = self._fabrik_subchain(cur, j, self.target)
                    subchain_axes = self._get_current_dynamic_axes(cur)
                    self.history.append({
                        'title': f'Iter {it} (Sub-chain from j={j + 1})',
                        'positions': np.copy(cur),
                        'axes': subchain_axes,
                        'error': self._norm(cur[-1] - self.target)
                    })
                    current_dynamic_axes = subchain_axes

            self.joint_positions = cur
        else:
            print(f"\n⚠️ Max iterations ({max_iterations}) reached.")

        final_axes = self._get_current_dynamic_axes(self.joint_positions)
        self.history.append({
            'title': 'Final State',
            'positions': np.copy(self.joint_positions),
            'axes': final_axes,
            'error': self._norm(self.joint_positions[-1] - self.target)
        })
        print(f"Pre-computation complete. {len(self.history)} steps recorded.")

    def solve_for_real_world(self, target_position, tolerance=0.01, max_iterations=20):
        print("\n--- Solving with Problem Decomposition ---")
        target = np.asarray(target_position)
        yaw_angle = np.arctan2(target[1], target[0])
        print(f"Target Yaw Angle: {np.rad2deg(yaw_angle):.2f} degrees")

        c, s = math.cos(-yaw_angle), math.sin(-yaw_angle)
        Rz_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        virtual_target = Rz_inv @ target
        print(f"Virtual Target (in robot's base frame): {virtual_target}")

        self.solve(virtual_target, tolerance, max_iterations)

        self.last_yaw_angle = yaw_angle
        self.target = target
        print("--- Base IK solved, final rotation will be applied in visualization ---")

    # --------------------------- Visualization ---------------------------
    def _draw_constraint_circle(self, ax, center, normal, radius):
        u = self._any_perp(normal)
        v = np.cross(normal, u)
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = center[:, np.newaxis] + radius * (
            u[:, np.newaxis] * np.cos(theta) + v[:, np.newaxis] * np.sin(theta)
        )
        ax.plot(circle_points[0, :], circle_points[1, :], circle_points[2, :],
                color='g', linestyle='--', label='Feasible Range')

    def run_interactive_viewer(self):
        if not self.history:
            print("Please run the 'solve' method first to compute the steps.")
            return
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("\n--- Interactive Mode ---\nPress '→' to advance, '←' to go back.\nClose the plot window to exit.")
        self.current_step = 0
        self.redraw()
        plt.show()

    def on_key_press(self, event):
        if event.key == 'right':
            if self.current_step < len(self.history) - 1:
                self.current_step += 1
        elif event.key == 'left':
            if self.current_step > 0:
                self.current_step -= 1
        self.redraw()

    def redraw(self):
        self.ax.clear()
        state = self.history[self.current_step]
        # saved positions/axes are in "virtual" frame
        unrotated_positions = state['positions']
        unrotated_axes = state['axes']
        title, error = state['title'], state['error']

        # apply final base yaw
        yaw = self.last_yaw_angle
        c, s = math.cos(yaw), math.sin(yaw)
        Rz_fwd = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        positions = (Rz_fwd @ unrotated_positions.T).T

        dynamic_axes = []
        if unrotated_axes:
            for axis in unrotated_axes:
                if axis is not None:
                    dynamic_axes.append(Rz_fwd @ np.asarray(axis, dtype=float).reshape(3,))
                else:
                    dynamic_axes.append(None)

        plot_points = np.vstack([self.base_position, positions])
        self.ax.plot(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                    'c-', marker='o', markersize=8, markerfacecolor='b', label='Robot Arm')

        axis_starts, axis_vectors = [], []
        for j in range(self.num_joints):
            axis = dynamic_axes[j]
            if axis is not None:
                axis_starts.append(positions[j])
                axis_vectors.append(axis)

        if axis_starts:
            starts = np.array(axis_starts)
            vectors = np.array(axis_vectors)
            self.ax.quiver(starts[:, 0], starts[:, 1], starts[:, 2],
                           vectors[:, 0], vectors[:, 1], vectors[:, 2],
                           length=0.1, color='m', label='Joint Axis')

        # visualize feasible circle during Project step (parent joint)
        match = re.search(r'Project j=(\d+)', title)
        if match:
            locked_idx = int(match.group(1)) - 1
            parent_idx = locked_idx - 1
            if 0 <= parent_idx < len(dynamic_axes):
                parent_axis = dynamic_axes[parent_idx]
                if parent_axis is not None:
                    center = positions[parent_idx]
                    radius = self.link_lengths[locked_idx]
                    self._draw_constraint_circle(self.ax, center, parent_axis, radius)

        self.ax.scatter(self.base_position[0], self.base_position[1], self.base_position[2],
                        c='black', s=100, marker='s', label='Base')
        self.ax.scatter(self.target[0], self.target[1], self.target[2],
                        c='r', s=150, marker='*', label='Target')

        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        max_reach = self.total_length * 1.1
        self.ax.set_xlim([-max_reach, max_reach])
        self.ax.set_ylim([-max_reach, max_reach])
        self.ax.set_zlim([-max_reach, max_reach])

        final_error = self._norm(positions[-1] - self.target)
        self.ax.set_title(f"Step {self.current_step + 1}/{len(self.history)}: {title}\nPosition Error: {final_error:.6f}")

        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.fig.canvas.draw()


# ================================================================
#  Main Execution
# ================================================================
if __name__ == "__main__":
    urdf_path = "piper_no_gripper_description.urdf"
    # urdf_path = "panda.urdf"
    kin_model = KinematicModel(urdf_filename=urdf_path)
    solver = InteractiveConstrainedFABRIKSolver(kin_model)

    target = [0.4, 0.1, 0.1]

    solver.solve_for_real_world(target, tolerance=0.01, max_iterations=15)

    solver.run_interactive_viewer()

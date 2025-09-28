import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ===============================
# Utility helpers
# ===============================
def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float).reshape(3,)
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _any_perp(a):
    a = _unit(a)
    ref = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
    return _unit(np.cross(a, ref))


def _proj_on_plane(v, n):
    n = _unit(n)
    return v - np.dot(v, n) * n


def _rodrigues(v, axis, theta):
    axis = _unit(axis)
    v = np.asarray(v, dtype=float).reshape(3,)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    I = np.eye(3)
    return (I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)) @ v


# ===============================
# FABRIK-R Standalone Solver
# ===============================
class FabrikRSolver:
    """
    Standalone FABRIK-R (hinge/pivot 1-DOF chain) with interactive visualization.

    Inputs:
      - joint_positions: (N,3) initial joint positions p1..pN (base p0 is separate)
      - joint_axes:      (N,3) per-joint axis vector (hinge and pivot use same 'axis' carrier)
      - joint_types:     list of 'hinge' or 'pivot' of length N
      - base_position:   3-vector p0 (fixed)

    Behavior:
      - Stores per-step states in `self.history` for visualization:
        dict(title, positions, axes, error)
      - Visualization exposes: run_interactive_viewer(), on_key_press(), redraw()
    """

    def __init__(self, joint_positions, joint_axes, joint_types, base_position=None):
        self.p0 = np.zeros(3, dtype=float) if base_position is None else np.asarray(base_position, float).reshape(3,)
        self.p = np.array(joint_positions, dtype=float).reshape(-1, 3)
        self.axes = np.array(joint_axes, dtype=float).reshape(-1, 3)
        self.kinds = list(joint_types)
        assert len(self.p) == len(self.axes) == len(self.kinds), "Mismatched chain sizes."
        self.N = len(self.p)

        # link lengths: d0 = |p1 - p0|, d1 = |p2 - p1|, ..., d_{N-1}
        self.d = np.zeros(self.N, dtype=float)
        self.d[0] = np.linalg.norm(self.p[0] - self.p0)
        for i in range(1, self.N):
            self.d[i] = np.linalg.norm(self.p[i] - self.p[i - 1])
        self.total_len = float(np.sum(self.d))

        self.history = []
        # viz fields
        self.fig = None
        self.ax = None
        self.current_step = 0
        self.target = None

    # -------------- Algorithm 2 helpers (Find concurrent & theta) --------------
    def _find_concurrent_index(self, i, ev_init, forward=True):
        step = -1 if forward else 1
        start = i - 1 if forward else i + 1
        for j in range(start, -1 if forward else self.N, step):
            if j < 0 or j >= self.N:
                break
            ax = self.axes[j]
            if np.linalg.norm(np.cross(_unit(ev_init), _unit(ax))) > 1e-6:
                return j
        return 0 if forward else (self.N - 1)

    def _solve_theta_candidates(self, El, Ev, alpha_beta_gamma):
        El = _unit(El); Ev = _unit(Ev)
        alpha, beta, gamma = alpha_beta_gamma
        t = np.cross(El, Ev)

        K1 = alpha*Ev[0] + beta*Ev[1] + gamma*Ev[2]
        K2 = np.dot(El, Ev) * (alpha*El[0] + beta*El[1] + gamma*El[2])
        K3 = alpha*t[0] + beta*t[1] + gamma*t[2]

        A, B, C = (K1 - K2), K3, K2
        R = math.hypot(A, B)
        if R < 1e-12:
            return []

        val = -C / R
        if val < -1.0 - 1e-12 or val > 1.0 + 1e-12:
            return []
        val = min(1.0, max(-1.0, val))

        delta = math.atan2(B, A)
        phi = math.acos(val)
        thetas = []
        for sgn in (+1, -1):
            base = (delta + sgn * phi) / 2.0
            thetas.append(base)
            thetas.append(base + math.pi)
        return [((t + 4*math.pi) % (2*math.pi)) for t in thetas]

    def _define_plane_i(self, i, p_prev, forward=True):
        prev_idx = i + 1 if forward else (i - 1)
        El = self.axes[prev_idx]

        Ev_init = El.copy()
        j = self._find_concurrent_index(i, Ev_init, forward=forward)

        pj = self.p[j]
        alpha_beta_gamma = (p_prev - pj)

        cand = _proj_on_plane((pj - p_prev), El)
        Ev = _any_perp(El) if (np.linalg.norm(cand) < 1e-9) else _unit(cand)

        thetas = self._solve_theta_candidates(El, Ev, alpha_beta_gamma)

        best = None
        best_norm = -1.0
        for th in thetas:
            En = (math.cos(2*th) * Ev
                  + (1 - math.cos(2*th)) * (np.dot(El, Ev)) * El
                  + math.sin(2*th) * np.cross(El, Ev))
            En = _unit(En)
            vproj = _proj_on_plane((pj - p_prev), En)
            nv = np.linalg.norm(vproj)
            if nv > best_norm:
                best_norm = nv
                best = En
        if best is None:
            best = _unit(_any_perp(El))
        return best

    # -------------- FABRIK forward/backward steps --------------
    def _step_forward(self, target, record=True, iter_idx=0):
        p = self.p.copy()
        p[-1] = target
        if record:
            self._record_state("Iter %d (Tip to Target)" % iter_idx, p)

        for i in range(self.N - 2, -1, -1):
            p_prev = p[i + 1]
            axis_prev = self.axes[i + 1]
            if self.kinds[i + 1] == 'hinge':
                En_prev = _unit(axis_prev)
            else:
                hint = _proj_on_plane((self.p[i] - p_prev), axis_prev)
                En_prev = _unit(np.cross(axis_prev, _unit(hint) if np.linalg.norm(hint) > 1e-9 else _any_perp(axis_prev)))

            dir_hint = _proj_on_plane((self.p[i] - p_prev), En_prev)
            if np.linalg.norm(dir_hint) < 1e-9:
                hint2 = (self.p[i - 1] - p_prev) if i - 1 >= 0 else (self.p0 - p_prev)
                dir_hint = _proj_on_plane(hint2, En_prev)
                if np.linalg.norm(dir_hint) < 1e-9:
                    dir_hint = _any_perp(En_prev)
            dir_hat = _unit(dir_hint)

            p_hat_i = p_prev + self.d[i + 1] * dir_hat

            En_i = self._define_plane_i(i, p_prev=p_prev, forward=True)

            v_to_next = ((self.p[i - 1] if i - 1 >= 0 else self.p0) - p_prev)
            dir_plane = _proj_on_plane(v_to_next, En_i)
            if np.linalg.norm(dir_plane) < 1e-9:
                dir_plane = _proj_on_plane((p_hat_i - p_prev), En_i)
            dir_plane = _unit(dir_plane)
            p[i] = p_prev + self.d[i + 1] * dir_plane

        if record:
            self._record_state("Iter %d (After Forward)" % iter_idx, p)
        self.p = p

    def _step_backward(self, record=True, iter_idx=0):
        p = self.p.copy()
        p[0] = self.p0
        if record:
            self._record_state("Iter %d (Base Fixed)" % iter_idx, p)

        for i in range(1, self.N):
            p_prev = p[i - 1]
            axis_prev = self.axes[i - 1]
            if self.kinds[i - 1] == 'hinge':
                En_prev = _unit(axis_prev)
            else:
                hint = _proj_on_plane((self.p[i] - p_prev), axis_prev)
                En_prev = _unit(np.cross(axis_prev, _unit(hint) if np.linalg.norm(hint) > 1e-9 else _any_perp(axis_prev)))

            dir_hint = _proj_on_plane((self.p[i] - p_prev), En_prev)
            if np.linalg.norm(dir_hint) < 1e-9:
                hint2 = (self.p[i + 1] - p_prev) if i + 1 < self.N else (self.p[-1] - p_prev)
                dir_hint = _proj_on_plane(hint2, En_prev)
                if np.linalg.norm(dir_hint) < 1e-9:
                    dir_hint = _any_perp(En_prev)
            dir_hat = _unit(dir_hint)
            p_hat_i = p_prev + self.d[i] * dir_hat

            En_i = self._define_plane_i(i, p_prev=p_prev, forward=False)

            v_to_next = ((self.p[i + 1] if i + 1 < self.N else self.p[-1]) - p_prev)
            dir_plane = _proj_on_plane(v_to_next, En_i)
            if np.linalg.norm(dir_plane) < 1e-9:
                dir_plane = _proj_on_plane((p_hat_i - p_prev), En_i)
            dir_plane = _unit(dir_plane)
            p[i] = p_prev + self.d[i] * dir_plane

        if record:
            self._record_state("Iter %d (After Backward)" % iter_idx, p)
        self.p = p

    # -------------- Public API --------------
    def solve(self, target, tol=1e-3, max_iters=32, record_history=True):
        self.target = np.asarray(target, float).reshape(3,)
        self.history = []
        self._record_state("Initial", self.p)

        if np.linalg.norm(self.target - self.p0) > self.total_len + 1e-9:
            dir0 = _unit(self.target - self.p0)
            p = np.zeros_like(self.p)
            p[0] = self.p0 + dir0 * self.d[0]
            for i in range(1, self.N):
                p[i] = p[i - 1] + dir0 * self.d[i]
            self.p = p
            self._record_state("Unreachable — straightened", self.p)
            return self.p

        for it in range(1, max_iters + 1):
            self._step_forward(self.target, record=record_history, iter_idx=it)
            self._step_backward(record=record_history, iter_idx=it)
            if record_history:
                self._record_state("Iter %d (End)" % it, self.p)
            if np.linalg.norm(self.p[-1] - self.target) <= tol:
                break
        self._record_state("Final", self.p)
        return self.p

    # -------------- History helpers --------------
    def _record_state(self, title, positions):
        positions = np.asarray(positions, float).reshape(-1, 3)
        # For visualization, expose joint axes as-is (no transport relative to parent here).
        axes = [self.axes[i].copy() if (self.kinds[i] == 'hinge' and self.axes[i] is not None) else None
                for i in range(self.N)]
        error = float(np.linalg.norm(positions[-1] - (self.target if self.target is not None else positions[-1])))
        self.history.append({
            'title': str(title),
            'positions': positions.copy(),
            'axes': axes,
            'error': error
        })

    # ===============================
    # Visualization (cfabrik.py-style)
    # ===============================
    def _draw_constraint_circle(self, ax, center, normal, radius):
        """Draws a circle (feasible locus for child link end) in plane with 'normal'."""
        normal = _unit(normal)
        u = _any_perp(normal)
        v = np.cross(normal, u)
        theta = np.linspace(0, 2*np.pi, 120)
        circle_points = center[:, None] + radius * (u[:, None]*np.cos(theta) + v[:, None]*np.sin(theta))
        ax.plot(circle_points[0, :], circle_points[1, :], circle_points[2, :])

    def run_interactive_viewer(self):
        if not self.history:
            print("\nRun solve() first to populate history.")
            return
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("\n--- Interactive Mode ---\nPress Right Arrow to advance, Left Arrow to go back.\nClose the window to exit.")
        self.current_step = 0
        self.redraw()
        plt.show()

    def on_key_press(self, event):
        if event.key == 'right':
            if self.current_step < len(self.history) - 1:
                self.current_step += 1
                self.redraw()
        elif event.key == 'left':
            if self.current_step > 0:
                self.current_step -= 1
                self.redraw()

    def redraw(self):
        self.ax.clear()
        state = self.history[self.current_step]
        positions = state['positions']
        axes = state['axes']
        title = state['title']
        error = state['error']

        # Draw links (including base as first point)
        plot_points = np.vstack([self.p0, positions])
        self.ax.plot(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], marker='o')

        # Draw per-joint axis arrows and feasible circles for hinges
        axis_starts = []
        axis_vecs = []
        for j in range(self.N):
            axj = axes[j]
            if axj is not None:
                axis_starts.append(positions[j])
                axis_vecs.append(_unit(axj))

                # show feasible circle for child if exists
                parent = positions[j]
                if j + 1 < self.N:
                    radius = float(np.linalg.norm(positions[j + 1] - positions[j]))
                else:
                    # last joint — show radius of its incoming link
                    radius = float(np.linalg.norm(positions[j] - (positions[j - 1] if j - 1 >= 0 else self.p0)))
                self._draw_constraint_circle(self.ax, parent, axj, radius)

        if axis_starts:
            axis_starts = np.asarray(axis_starts)
            axis_vecs = np.asarray(axis_vecs)
            self.ax.quiver(axis_starts[:, 0], axis_starts[:, 1], axis_starts[:, 2],
                           axis_vecs[:, 0], axis_vecs[:, 1], axis_vecs[:, 2],
                           length=0.15, normalize=True)

        # Base & target
        self.ax.scatter(self.p0[0], self.p0[1], self.p0[2], marker='s', s=60)
        if self.target is not None:
            self.ax.scatter(self.target[0], self.target[1], self.target[2], marker='*', s=120)

        # Bounds & labels
        reach = self.total_len * 1.25 if self.total_len > 0 else 1.0
        self.ax.set_xlim([-reach, reach])
        self.ax.set_ylim([-reach, reach])
        self.ax.set_zlim([-reach, reach])
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f"Step {self.current_step + 1}/{len(self.history)}: {title}\nEnd-effector error: {error:.6g}")
        self.fig.canvas.draw_idle()


# ===============================
# Demo
# ===============================
def _demo():
    # Simple 4-joint chain in XY plane with Z-axes for hinges
    base = np.array([0., 0., 0.])
    p = np.array([[0.2, 0., 0.],
                  [0.4, 0., 0.],
                  [0.6, 0., 0.],
                  [0.8, 0., 0.]], dtype=float)
    axes = np.array([[0., 0., 1.],
                     [0., 1., 0.],
                     [0., 1., 0.],
                     [1., 0., 0.]], dtype=float)
    kinds = ['hinge', 'hinge', 'hinge', 'hinge']

    solver = FabrikRSolver(p, axes, kinds, base_position=base)
    target = np.array([0.3, 0.2, 0.1])
    solver.solve(target, tol=1e-4, max_iters=30, record_history=True)
    solver.run_interactive_viewer()


if __name__ == "__main__":
    _demo()

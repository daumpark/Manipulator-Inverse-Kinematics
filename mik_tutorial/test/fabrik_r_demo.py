#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FABRIK-R IK demo (6-DoF) with real-time visualization (CYLINDER joints/links).

Joint types (from base to tip):
  joint1: pivot
  joint2: hinge
  joint3: hinge
  joint4: pivot
  joint5: hinge
  joint6: pivot
Nodes: p0..p6, where joint i is located at p(i-1) (i=1..6).

Keyboard (figure focused):
  A/D : move target -/+X
  W/S : move target +/ -Y
  R/F : move target +/ -Z
  Z/X : speed down / up
  C   : recenter target
  H   : print help
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------- Configuration ----------------------
np.set_printoptions(precision=3, suppress=True)

# Link lengths (m) between consecutive joints [p0->p1, ..., p5->p6]
L_DEFAULT = np.array([0.30, 0.30, 0.25, 0.20, 0.15, 0.10], dtype=float)

# Joint types
PIVOT, HINGE = 0, 1
JOINT_TYPES = [PIVOT, HINGE, HINGE, PIVOT, HINGE, PIVOT]

# Hinge axes (WORLD frame). Use None for pivot joints.
AXES_WORLD = [
    None,                              # j1 pivot
    np.array([0.0, 0.0, 1.0]),         # j2 hinge about +Z
    np.array([0.0, 1.0, 0.0]),         # j3 hinge about +Y
    None,                              # j4 pivot
    np.array([0.0, 0.0, 1.0]),         # j5 hinge about +Z
    None                               # j6 pivot
]

# Visual params
LINK_RADIUS  = 0.015
HINGE_RADIUS = 0.025
PIVOT_RADIUS = 0.028
JOINT_LEN    = 0.06    # length of the little joint cylinder
CYL_N_THETA  = 20      # cylinder resolution

COLOR_LINK  = (0.6, 0.6, 0.6, 0.9)
COLOR_HINGE = (1.0, 0.55, 0.0, 0.9)   # orange
COLOR_PIVOT = (0.2, 0.45, 0.9, 0.9)   # blue
COLOR_TGT   = (0.85, 0.1, 0.1, 1.0)

# ---------------------- Init chain ----------------------
def initial_chain(base=np.zeros(3), L=L_DEFAULT):
    p = [np.array(base, dtype=float)]
    # lay roughly in XZ with a slight bend to avoid singularity
    dir0 = np.array([1.0, 0.0, 0.0])
    x = base.copy()
    for i, li in enumerate(L):
        if i == 2:
            dir0 = np.array([0.9, 0.0, 0.435889894])
            dir0 = dir0 / np.linalg.norm(dir0)
        x = x + li * dir0
        p.append(x.copy())
    return p  # [p0..p6]

# ---------------------- FABRIK-R core (hinge+pivot) ----------------------
class FabrikR:
    def __init__(self, joint_types, axes_world, L, max_iter=80, tol=1e-3):
        assert len(joint_types) == 6 and len(L) == 6 and len(axes_world) == 6
        self.joint_types = joint_types
        # normalize axes for hinge joints
        axes = []
        for a in axes_world:
            if a is None:
                axes.append(None)
            else:
                a = np.array(a, dtype=float)
                na = np.linalg.norm(a)
                axes.append(a/na if na > 1e-12 else np.array([0.,0.,1.]))
        self.axes_world = axes
        self.L = np.asarray(L, dtype=float)
        self.max_iter = max_iter
        self.tol = tol
        # hinge invariants from initial shape
        self.dpar  = np.zeros(6)   # axial component along hinge axis
        self.rperp = np.zeros(6)   # perpendicular radius
        self._p_ref = None

    @staticmethod
    def _any_perp(axis):
        t = np.array([1.0,0.0,0.0]) if abs(axis[0]) < 0.9 else np.array([0.0,1.0,0.0])
        u = np.cross(axis, t); n = np.linalg.norm(u)
        return u/n if n > 1e-12 else np.array([0.0,0.0,1.0])

    def set_initial_shape(self, p_list):
        self._p_ref = [np.array(pi, dtype=float) for pi in p_list]
        for i in range(6):
            if self.joint_types[i] == HINGE:
                a = self.axes_world[i]
                v = self._p_ref[i+1] - self._p_ref[i]
                dp = float(np.dot(v, a))
                vp = v - dp * a
                self.dpar[i]  = dp
                self.rperp[i] = float(np.linalg.norm(vp))
            else:
                self.dpar[i]  = 0.0
                self.rperp[i] = 0.0

    def solve_step(self, p_list, target, iters=1):
        p = [np.array(pi, dtype=float) for pi in p_list]
        base = p[0].copy()

        for _ in range(iters):
            # ---------- Forward pass ----------
            p[6] = np.array(target, dtype=float)
            for i in range(5, -1, -1):
                if self.joint_types[i] == PIVOT:
                    ref = p[i] if self._p_ref is None else self._p_ref[i]
                    v = ref - p[i+1]; n = np.linalg.norm(v)
                    v = np.array([1.0,0.0,0.0]) if n < 1e-12 else v/n
                    p[i] = p[i+1] + self.L[i]*v
                else:
                    a = self.axes_world[i]
                    ref = base - p[i+1]
                    u = ref - np.dot(ref, a)*a
                    nu = np.linalg.norm(u)
                    u = self._any_perp(a) if nu < 1e-12 else u/nu
                    v = self.dpar[i]*a + self.rperp[i]*u
                    p[i] = p[i+1] - v

            # ---------- Backward pass ----------
            p[0] = base.copy()
            for i in range(6):
                if self.joint_types[i] == PIVOT:
                    ref = p[i+1] if self._p_ref is None else self._p_ref[i+1]
                    v = ref - p[i]; n = np.linalg.norm(v)
                    v = np.array([1.0,0.0,0.0]) if n < 1e-12 else v/n
                    p[i+1] = p[i] + self.L[i]*v
                else:
                    a = self.axes_world[i]
                    ref = target - p[i]
                    u = ref - np.dot(ref, a)*a
                    nu = np.linalg.norm(u)
                    u = self._any_perp(a) if nu < 1e-12 else u/nu
                    v = self.dpar[i]*a + self.rperp[i]*u
                    p[i+1] = p[i] + v

            if np.linalg.norm(p[6] - target) < self.tol:
                break

            self._p_ref = [pi.copy() for pi in p]
        return p

# ---------------------- Cylinder utilities ----------------------
def _orthonormal_basis(k):
    """Given unit k, return (i, j) so that [i, j, k] is ONB."""
    k = k / (np.linalg.norm(k) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    i = np.cross(tmp, k); ni = np.linalg.norm(i)
    i = i/ni if ni > 1e-12 else np.array([1.0, 0.0, 0.0])
    j = np.cross(k, i)
    return i, j, k

def make_cylinder_between(p1, p2, radius, n_theta=CYL_N_THETA):
    """Return vertices (faces) for a closed cylinder mesh between p1 and p2."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    v  = p2 - p1; L = np.linalg.norm(v)
    if L < 1e-9:
        v = np.array([0.0, 0.0, 1.0]); L = 1e-9
    k = v / L
    i, j, k = _orthonormal_basis(k)

    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=True)
    circle1 = [p1 + radius*(np.cos(t)*i + np.sin(t)*j) for t in thetas]
    circle2 = [p2 + radius*(np.cos(t)*i + np.sin(t)*j) for t in thetas]

    faces = []
    # side faces (quads as two triangles)
    for t in range(n_theta-1):
        a1, a2 = circle1[t], circle1[t+1]
        b1, b2 = circle2[t], circle2[t+1]
        faces.append([a1, a2, b2])
        faces.append([a1, b2, b1])
    # close last segment
    a1, a2 = circle1[-1], circle1[0]
    b1, b2 = circle2[-1], circle2[0]
    faces.append([a1, a2, b2])
    faces.append([a1, b2, b1])

    # end caps (optional): two fan triangulations
    c1 = np.mean(np.stack(circle1), axis=0)
    c2 = np.mean(np.stack(circle2), axis=0)
    for t in range(n_theta-1):
        faces.append([c1, circle1[t], circle1[t+1]])
        faces.append([c2, circle2[t+1], circle2[t]])
    faces.append([c1, circle1[-1], circle1[0]])
    faces.append([c2, circle2[0], circle2[-1]])
    return faces

def make_centered_cylinder(center, axis_dir, length, radius, n_theta=CYL_N_THETA):
    """Centered cylinder at 'center' oriented along axis_dir (unit), with given length & radius."""
    a = np.asarray(axis_dir, float)
    na = np.linalg.norm(a)
    if na < 1e-9:
        a = np.array([0.0, 0.0, 1.0]); na = 1.0
    a = a / na
    p1 = center - 0.5*length*a
    p2 = center + 0.5*length*a
    return make_cylinder_between(p1, p2, radius, n_theta)

# ---------------------- Visualization ----------------------
class LiveSim:
    def __init__(self):
        self.L = L_DEFAULT.copy()
        self.joint_types = JOINT_TYPES[:]
        self.axes = AXES_WORLD[:]

        # init chain & solver
        self.p = initial_chain(L=self.L)
        self.solver = FabrikR(self.joint_types, self.axes, self.L, max_iter=120, tol=1e-3)
        self.solver.set_initial_shape(self.p)

        # target
        self.target = self.p[-1].copy() + np.array([0.10, 0.10, 0.10])
        self.step = 0.02  # movement step in meters

        # figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect((1,1,1))
        self.ax.set_xlabel('X [m]'); self.ax.set_ylabel('Y [m]'); self.ax.set_zlabel('Z [m]')
        self._set_bounds()

        # artists
        self.surfs = []   # Poly3DCollection list (links + joints)
        self.tgt_artist = None

        # events & animation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ani = FuncAnimation(self.fig, self._update, interval=33, blit=False)

        print(self._help_text())

    def _help_text(self):
        return (
            "Controls:\n"
            "  A/D : target -/+X\n"
            "  W/S : target +/ -Y\n"
            "  R/F : target +/ -Z\n"
            "  Z/X : speed down / up\n"
            "  C   : recenter target\n"
            "  H   : help\n"
        )

    def _set_bounds(self, span=1.2):
        self.ax.set_xlim(-span, span)
        self.ax.set_ylim(-span, span)
        self.ax.set_zlim( 0.0, span*2/1.2)

    def on_key(self, event):
        key = (event.key or '').lower()
        d = self.step
        if   key == 'a': self.target[0] -= d
        elif key == 'd': self.target[0] += d
        elif key == 'w': self.target[1] += d
        elif key == 's': self.target[1] -= d
        elif key == 'r': self.target[2] += d
        elif key == 'f': self.target[2] -= d
        elif key == 'z': self.step = max(0.005, self.step * 0.5); print(f"step={self.step:.3f} m")
        elif key == 'x': self.step = min(0.2,  self.step * 2.0); print(f"step={self.step:.3f} m")
        elif key == 'c': self.target = np.array([0.4, 0.0, 0.4])
        elif key == 'h': print(self._help_text())

    # ---------- drawing helpers ----------
    def _clear_surfaces(self):
        for s in self.surfs:
            try: s.remove()
            except Exception: pass
        self.surfs = []
        if self.tgt_artist is not None:
            try: self.tgt_artist.remove()
            except Exception: pass
            self.tgt_artist = None

    def _add_mesh(self, faces, color):
        poly = Poly3DCollection(faces, facecolors=[color], edgecolors='none')
        self.ax.add_collection3d(poly)
        self.surfs.append(poly)

    def _draw_links(self, P):
        # cylinders between p[i] and p[i+1]
        for i in range(6):
            faces = make_cylinder_between(P[i], P[i+1], LINK_RADIUS)
            self._add_mesh(faces, COLOR_LINK)

    def _draw_joints(self, P):
        # joint i (1..6) is at node p[i-1] => index j_idx = i-1 => 0..5
        for j_idx in range(6):
            center = P[j_idx]
            if JOINT_TYPES[j_idx] == HINGE:
                axis = AXES_WORLD[j_idx]
                faces = make_centered_cylinder(center, axis, JOINT_LEN, HINGE_RADIUS)
                self._add_mesh(faces, COLOR_HINGE)
            else:  # PIVOT
                # choose an orientation: bisector of incoming/outgoing link directions if possible
                if j_idx == 0:
                    out_dir = P[1] - P[0]
                    axis = out_dir / (np.linalg.norm(out_dir) + 1e-12)
                else:
                    in_dir  = P[j_idx]   - P[j_idx-1]
                    out_dir = P[j_idx+1] - P[j_idx]
                    ni = np.linalg.norm(in_dir); no = np.linalg.norm(out_dir)
                    if ni < 1e-9 and no < 1e-9:
                        axis = np.array([0.0,0.0,1.0])
                    elif ni < 1e-9:
                        axis = out_dir / no
                    elif no < 1e-9:
                        axis = in_dir / ni
                    else:
                        b = in_dir/ni + out_dir/no
                        if np.linalg.norm(b) < 1e-9:  # straight line: pick any perp
                            axis = out_dir / no
                        else:
                            axis = b / np.linalg.norm(b)
                faces = make_centered_cylinder(center, axis, JOINT_LEN, PIVOT_RADIUS)
                self._add_mesh(faces, COLOR_PIVOT)

    def _draw_target(self):
        # tiny cylinder “marker” at target (vertical)
        faces = make_centered_cylinder(self.target, np.array([0,0,1.0]), 0.04, 0.01, n_theta=14)
        poly = Poly3DCollection(faces, facecolors=[COLOR_TGT], edgecolors='none')
        self.ax.add_collection3d(poly)
        self.tgt_artist = poly

    # ---------- animation update ----------
    def _update(self, _frame):
        # IK update
        self.p = self.solver.solve_step(self.p, self.target, iters=2)
        P = np.array(self.p)

        # redraw meshes
        self._clear_surfaces()
        self._draw_links(P)
        self._draw_joints(P)
        self._draw_target()

        return tuple(self.surfs) + ((self.tgt_artist,) if self.tgt_artist else ())

    def run(self):
        plt.title("FABRIK-R (6DoF) — cylinders for links & joints")
        plt.show()

def main():
    sim = LiveSim()
    sim.run()

if __name__ == "__main__":
    main()

"""Inverse kinematics solvers for simple 2D planar chains.

Contains:
    - CCD2D: Cyclic Coordinate Descent solver.
    - FABRIK2D: Forward And Backward Reaching IK solver.
"""

import numpy as np


class CCD2D:
    """Simple 2D CCD inverse kinematics solver for a planar chain."""

    def __init__(self, link_lengths):
        """
        Initialize the CCD solver.

        Args:
            link_lengths: Iterable of link lengths for the chain.
        """
        # Store link lengths as a NumPy array of floats.
        self.L = np.asarray(link_lengths, dtype=float)

        # Maximum number of CCD iterations.
        self.max_iter = 200

        # Position tolerance for convergence.
        self.tol = 1e-3

    def fk(self, q):
        """
        Forward kinematics for a 2D chain.

        Args:
            q: Joint angles in radians, shape (N,).

        Returns:
            (N+1, 2) array of joint and end-effector positions.
            The first point is the base at (0, 0).
        """
        q = np.asarray(q, dtype=float)

        # Start with the base at the origin.
        pts = [np.zeros(2, dtype=float)]

        # Accumulated absolute angle along the chain.
        angle = 0.0

        # Current end position of the chain.
        p = np.zeros(2, dtype=float)

        for qi, Li in zip(q, self.L):
            # Accumulate absolute angle for each link.
            angle += qi

            # Direction of current link in the world frame.
            d = np.array(
                [np.cos(angle) * Li, np.sin(angle) * Li],
                dtype=float,
            )

            # Move the end of the chain by this link.
            p = p + d
            pts.append(p.copy())

        return np.asarray(pts, dtype=float)

    def solve(self, target, q_seed):
        """
        Run CCD iterations to reach a 2D target.

        Args:
            target: Desired 2D end-effector position [x, y].
            q_seed: Initial joint angles, used as a starting guess.

        Returns:
            q: Solution joint angles.
            ok: True if converged within tolerance.
            info: Dict with keys:
                - 'iters_total': number of iterations executed.
                - 'pos_err': final position error.
        """
        target = np.asarray(target, dtype=float)

        # Copy the seed so we don't modify the input in-place.
        q = np.asarray(q_seed, dtype=float).copy()

        for it in range(self.max_iter):
            # Current forward kinematics and end-effector position.
            pts = self.fk(q)
            pe = np.linalg.norm(pts[-1] - target)

            # Early exit if we already satisfy the position tolerance.
            if pe < self.tol:
                return q, True, {
                    "iters_total": it + 1,
                    "pos_err": float(pe),
                }

            # Sweep joints in reverse order (from end-effector to base).
            for i in reversed(range(len(q))):
                pi = pts[i]
                pe_pt = pts[-1]

                # Vector from joint i to current end-effector.
                v1 = pe_pt - pi

                # Vector from joint i to desired target.
                v2 = target - pi

                # If either vector is too small, skip this joint.
                if (
                    np.linalg.norm(v1) < 1e-9
                    or np.linalg.norm(v2) < 1e-9
                ):
                    continue

                # Signed angle between v1 and v2.
                # np.cross for 2D vectors returns a scalar "z" component.
                cross = np.cross(v1, v2)
                dot = np.dot(v1, v2)
                dtheta = np.arctan2(cross, dot)

                # Apply joint update.
                q[i] += dtheta

                # Update FK for the new configuration before moving
                # the next joint in the loop.
                pts = self.fk(q)

        # If we are here, we did not converge within max_iter.
        pts = self.fk(q)
        pos_err = float(np.linalg.norm(pts[-1] - target))
        return q, False, {
            "iters_total": self.max_iter,
            "pos_err": pos_err,
        }


class FABRIK2D:
    """2D FABRIK inverse kinematics solver for a planar chain."""

    def __init__(self, link_lengths):
        """
        Initialize the FABRIK solver.

        Args:
            link_lengths: Iterable of link lengths for the chain.
        """
        # Link lengths (fixed).
        self.L = np.asarray(link_lengths, dtype=float)

        # Maximum number of FABRIK iterations.
        self.max_iter = 200

        # Position tolerance for convergence [meters].
        self.tol = 1e-3

    def forward_points(self, q):
        """
        Forward kinematics: return all joint + end-effector positions.

        Args:
            q: Joint angles [q0, ..., q_{N-1}] (radians).

        Returns:
            (N+1, 2) array of points, starting from base (0, 0).
        """
        q = np.asarray(q, dtype=float)

        # Start with the base at the origin.
        pts = [np.zeros(2, dtype=float)]

        # Accumulated absolute angle.
        angle = 0.0

        # Current end position of the chain.
        p = np.zeros(2, dtype=float)

        for qi, Li in zip(q, self.L):
            # Accumulate absolute angle for each link.
            angle += qi

            # World-frame displacement of this link.
            p = p + np.array(
                [np.cos(angle) * Li, np.sin(angle) * Li],
                dtype=float,
            )
            pts.append(p.copy())

        return np.asarray(pts, dtype=float)

    def solve(self, target, q_seed):
        """
        FABRIK iteration in 2D.

        Args:
            target: 2D target position [x, y].
            q_seed: Initial joint angles (N,).

        Returns:
            q: Solution joint angles.
            ok: True if converged within tolerance.
            info: Dict with 'iters_total' and 'pos_err'.
        """
        target = np.asarray(target, dtype=float)
        N = len(self.L)

        # Start from the seed configuration.
        q = np.asarray(q_seed, dtype=float).copy()

        for it in range(self.max_iter):
            # -----------------------------------------------------------------
            # 1) Rebuild the current chain geometry from q
            # -----------------------------------------------------------------
            pts = self.forward_points(q)

            # Save base position in case it is not at the origin.
            base = pts[0].copy()

            # -----------------------------------------------------------------
            # 2) Forward reaching: move from end-effector to base
            # -----------------------------------------------------------------

            # Fix the end-effector to the target.
            pts[-1] = target.copy()

            # Move each intermediate joint while keeping link lengths fixed.
            for i in reversed(range(N)):
                r = np.linalg.norm(pts[i + 1] - pts[i])

                if r < 1e-12:
                    # Points are (almost) coincident; skip to avoid division
                    # by zero. The configuration will be regularized in later
                    # iterations after q is updated.
                    continue

                # Move Pi so that ||Pi - P_{i+1}|| = L[i].
                pts[i] = pts[i + 1] + (pts[i] - pts[i + 1]) * (self.L[i] / r)

            # Restore base position to its original location.
            pts[0] = base.copy()

            # -----------------------------------------------------------------
            # 3) Backward reaching: move from base to end-effector
            # -----------------------------------------------------------------
            for i in range(N):
                r = np.linalg.norm(pts[i + 1] - pts[i])

                if r < 1e-12:
                    # Again, avoid division by zero for degenerate segments.
                    continue

                # Move P_{i+1} so that ||P_{i+1} - P_i|| = L[i].
                pts[i + 1] = (
                    pts[i]
                    + (pts[i + 1] - pts[i]) * (self.L[i] / r)
                )

            # -----------------------------------------------------------------
            # 4) Recover joint angles from updated points
            # -----------------------------------------------------------------
            ang = 0.0

            for i in range(N):
                # Vector of link i in world coordinates.
                v = pts[i + 1] - pts[i]

                # Absolute orientation of link i.
                th = np.arctan2(v[1], v[0])

                # Relative angle at joint i (difference from previous link).
                dth = th - ang

                # Wrap to [-pi, pi] to keep angles bounded.
                dth = (dth + np.pi) % (2.0 * np.pi) - np.pi

                # Overwrite joint angle (do not accumulate).
                q[i] = dth

                # Update accumulated absolute angle.
                ang = th

            # -----------------------------------------------------------------
            # 5) Convergence check based on FK from q
            # -----------------------------------------------------------------
            pts_fk = self.forward_points(q)
            pos_err = float(np.linalg.norm(pts_fk[-1] - target))

            if pos_err < self.tol:
                return q, True, {
                    "iters_total": it + 1,
                    "pos_err": pos_err,
                }

        # ---------------------------------------------------------------------
        # If we are here, we didn't converge within max_iter
        # ---------------------------------------------------------------------
        pts_fk = self.forward_points(q)
        pos_err = float(np.linalg.norm(pts_fk[-1] - target))

        return q, False, {
            "iters_total": self.max_iter,
            "pos_err": pos_err,
        }

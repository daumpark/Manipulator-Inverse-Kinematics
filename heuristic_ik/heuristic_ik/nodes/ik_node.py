"""2D heuristic IK node (CCD2D / FABRIK2D) with RViz visualization."""

import time
from typing import Optional

import numpy as np
import rclpy
from interactive_markers.interactive_marker_server import (
    InteractiveMarkerServer,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

from heuristic_ik.solvers import CCD2D, FABRIK2D


class HeuristicIK2DNode(Node):
    """
    ROS 2 node exposing 2D heuristic IK (CCD2D / FABRIK2D).

    Features:
        - Planar N-link arm in the XY plane (base at (0, 0)).
        - RViz interactive marker to move a 2D target on the XY plane.
        - On mouse release, run CCD2D or FABRIK2D and publish JointState.
    """

    def __init__(self) -> None:
        """Initialize the IK node and set up parameters and ROS interfaces."""
        super().__init__("heuristic_ik_2d_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        # Which solver to use: "ccd" or "fabrik".
        self.declare_parameter("solver", "ccd")

        # Link lengths [m]. Can be list or comma-separated string.
        self.declare_parameter("link_lengths", [0.35, 0.25, 0.20])

        # IK hyperparameters (common for CCD and FABRIK).
        self.declare_parameter("max_iter", 200)
        self.declare_parameter("tol", 1e-3)

        # Read and parse parameters from the parameter server.
        solver_name = (
            str(self.get_parameter("solver").value)
            .strip()
            .lower()
        )

        link_lengths = self.get_parameter("link_lengths").value
        max_iter = int(self.get_parameter("max_iter").value)
        tol = float(self.get_parameter("tol").value)

        # Select solver implementation.
        if solver_name == "fabrik":
            self.ik = FABRIK2D(link_lengths)
        else:
            # Default to CCD2D if the name does not match.
            self.ik = CCD2D(link_lengths)

        # Apply hyperparameters to solver.
        self.ik.max_iter = max_iter
        self.ik.tol = tol

        # Number of joints is inferred from number of links.
        self.n_joints = len(link_lengths)

        # Current configuration (seed for the next solver).
        self.q_seed = np.zeros(self.n_joints, dtype=float)

        # Base frame used for JointState and marker.
        self.frame: str = "base_link"

        # Target position on XY plane (start somewhere reachable).
        total_length = float(sum(link_lengths))
        self.target_xy = np.array(
            [0.8 * total_length, 0.0],
            dtype=float,
        )

        # Internal drag state for the interactive marker.
        # True while the user is dragging the marker.
        self._drag: bool = False

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        # Publisher for the joint state of the planar arm.
        self.joint_pub = self.create_publisher(
            JointState,
            "joint_states",
            10,
        )

        # Interactive marker server for RViz.
        self.im_server = InteractiveMarkerServer(
            self,
            "heuristic_2d_marker",
        )

        # Create the interactive marker in RViz.
        self._create_marker()

        self.get_logger().info(
            f"Heuristic 2D IK node ready "
            f"({self.ik.__class__.__name__}, "
            f"links={link_lengths}, "
            f"max_iters={self.ik.max_iter}, tol={self.ik.tol})",
        )

    # ----------------------------------------------------------------------
    # Interactive marker
    # ----------------------------------------------------------------------
    def _create_marker(self) -> None:
        """Create interactive marker for 2D target (XY only)."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame
        int_marker.name = "target_2d_heuristic"
        int_marker.description = "2D target (Heuristic IK)"

        # Initial pose of the marker (on XY plane).
        int_marker.pose.position.x = float(self.target_xy[0])
        int_marker.pose.position.y = float(self.target_xy[1])
        int_marker.pose.position.z = 0.0

        # Orientation is irrelevant in 2D; keep identity.
        int_marker.pose.orientation.w = 1.0
        int_marker.pose.orientation.x = 0.0
        int_marker.pose.orientation.y = 0.0
        int_marker.pose.orientation.z = 0.0

        # Visual marker (small cylinder on the plane).
        marker = Marker()
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.01
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Basic control that shows the marker geometry.
        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(marker)
        int_marker.controls.append(ctrl)

        # Move along X.
        ctrl = InteractiveMarkerControl()
        ctrl.orientation.w = 1.0
        ctrl.orientation.x = 1.0
        ctrl.orientation.y = 0.0
        ctrl.orientation.z = 0.0
        ctrl.name = "move_x"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        # Move along Y.
        ctrl = InteractiveMarkerControl()
        ctrl.orientation.w = 1.0
        ctrl.orientation.x = 0.0
        ctrl.orientation.y = 0.0
        ctrl.orientation.z = 1.0
        ctrl.name = "move_y"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        # Register feedback callback with the marker server.
        self.im_server.insert(
            int_marker,
            feedback_callback=self._feedback,
        )
        self.im_server.applyChanges()

    @staticmethod
    def _pose_to_xy(pose) -> np.ndarray:
        """Extract (x, y) position from an InteractiveMarker pose."""
        return np.array(
            [pose.position.x, pose.position.y],
            dtype=float,
        )

    def _feedback(self, fb: InteractiveMarkerFeedback) -> None:
        """
        Interactive marker feedback callback.

        Behavior:
            - While dragging: update target_xy.
            - On mouse release: run IK solver once.
        """
        # Update target position from the marker pose.
        self.target_xy = self._pose_to_xy(fb.pose)
        self.im_server.applyChanges()

        # Mouse pressed: start dragging.
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._drag = True
            return

        # Mouse released: if we were dragging, perform one IK solve.
        if (
            fb.event_type == InteractiveMarkerFeedback.MOUSE_UP
            and self._drag
        ):
            self._drag = False
            self._solve_once()
            return

    # ----------------------------------------------------------------------
    # IK solve and publishing
    # ----------------------------------------------------------------------
    def _solve_once(self) -> None:
        """Solve IK once for the current 2D target and publish JointState."""
        solver_name = self.ik.__class__.__name__
        self.get_logger().info(f"[{solver_name}] solve started")

        # Measure time spent in the IK solver.
        t0 = time.perf_counter()
        q, ok, info = self.ik.solve(self.target_xy, self.q_seed)
        dt = (time.perf_counter() - t0) * 1000.0  # milliseconds

        # Update seed for the next solve (warm start).
        self.g_seed = np.asarray(q, dtype=float).copy()

        # Forward kinematics for error calculation.
        if hasattr(self.ik, "fk"):
            # CCD2D uses 'fk'.
            pts = self.ik.fk(q)
        else:
            # FABRIK2D uses 'forward_points'.
            pts = self.ik.forward_points(q)

        pos_err = float(np.linalg.norm(pts[-1] - self.target_xy))
        iters = int(info.get("iters_total", -1))

        self.get_logger().info(
            f"[{solver_name}] t={dt:.1f} ms, "
            f"iters={iters}, "
            f"pos_err={pos_err * 1000.0:.2f} mm, "
            f"ok={ok}",
        )

        # Publish JointState so RViz can visualize the planar robot.
        js = JointState()
        js.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.frame,
        )

        # Simple joint naming: joint1, joint2, ...
        js.name = [f"joint{i + 1}" for i in range(self.n_joints)]

        # Joint angles as a plain Python list of floats.
        js.position = [float(x) for x in q]

        self.joint_pub.publish(js)


def main(args: Optional[list] = None) -> None:
    """Entry point for the 2D heuristic IK node."""
    rclpy.init(args=args)
    node = HeuristicIK2DNode()

    try:
        # Spin the node until Ctrl+C or shutdown.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C in the terminal.
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
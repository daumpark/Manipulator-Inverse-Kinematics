"""Real-time redundant IK ROS 2 node using CLIK-style solvers.

This node:
    - Uses one of the redundant IK solvers from redundant_ik.solvers.
    - Runs a real-time control loop via a timer, calling solver.step().
    - Provides an interactive marker in RViz to move the EE target position.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)
from interactive_markers.interactive_marker_server import (
    InteractiveMarkerServer,
)

from ik_common.common.kinematics import KinematicModel
from redundant_ik.solvers import (
    CTP_SVF_SD,
    NullspacePositionOnly,
    WeightedCLIK,
)


_SOLVER_MAP = {
    "nullspace": NullspacePositionOnly,
    "wclik": WeightedCLIK,
    "ctp_svf_sd": CTP_SVF_SD,
}


def _vec_to_pose44(p: np.ndarray) -> np.ndarray:
    """Convert a 3D position vector into a 4x4 homogeneous pose.

    Orientation is set to identity.

    Args:
        p: 3D position vector (3,).

    Returns:
        T: 4x4 homogeneous transform with translation p.
    """
    T = np.eye(4)
    T[:3, 3] = np.asarray(p, float)
    return T


class RedundantIKNode(Node):
    """Redundant IK node with real-time CLIK-style control loop.

    Key features:
        - Reads solver name and tuning parameters from ROS 2 parameters.
        - Uses an interactive marker to specify EE target position.
        - Runs a periodic timer that calls solver.step() once per tick.
        - Publishes JointState for visualization (e.g., in RViz).
    """

    def __init__(self) -> None:
        """Initialize node, solver, ROS interfaces, and control loop."""
        super().__init__("redundant_ik_node")

        # ------------------------------------------------------------------ #
        # Parameters                                                         #
        # ------------------------------------------------------------------ #
        self.declare_parameter("solver", "nullspace")  # nullspace|wclik|ctp_svf_sd
        self.declare_parameter("frame", "base_link")
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("lam", 1.0e-2)
        self.declare_parameter("k_ns", 0.2)
        self.declare_parameter("Kp", 1.0)
        self.declare_parameter("gamma_max", 0.2)
        self.declare_parameter("nu", 10.0)
        self.declare_parameter("sigma0", 5e-3)

        self.frame = str(self.get_parameter("frame").value)

        # Kinematic model (URDF + Pinocchio).
        self.kin = KinematicModel()

        # Select solver implementation.
        key = str(self.get_parameter("solver").value).lower()
        solver_cls = _SOLVER_MAP.get(key, NullspacePositionOnly)
        self.ik = solver_cls(self.kin)

        # Inject common tuning parameters into the solver (if available).
        dt = float(self.get_parameter("dt").value)
        if hasattr(self.ik, "dt"):
            self.ik.dt = dt
        if hasattr(self.ik, "lam"):
            self.ik.lam = float(self.get_parameter("lam").value)
        if hasattr(self.ik, "k_ns"):
            self.ik.k_ns = float(self.get_parameter("k_ns").value)
        if hasattr(self.ik, "Kp"):
            self.ik.Kp = float(self.get_parameter("Kp").value)
        if hasattr(self.ik, "gamma_max"):
            self.ik.gamma_max = float(
                self.get_parameter("gamma_max").value,
            )
        if hasattr(self.ik, "nu"):
            self.ik.nu = float(self.get_parameter("nu").value)
        if hasattr(self.ik, "sigma0"):
            self.ik.sigma0 = float(
                self.get_parameter("sigma0").value,
            )

        self.get_logger().info(
            f"Using Redundant IK solver: {self.ik.__class__.__name__}",
        )

        # ------------------------------------------------------------------ #
        # ROS I/O                                                            #
        # ------------------------------------------------------------------ #
        # JointState publisher.
        self.joint_pub = self.create_publisher(
            JointState,
            "joint_states",
            10,
        )

        # Interactive marker server for RViz.
        self.im_server = InteractiveMarkerServer(
            self,
            "ik_controls_redundant",
        )

        # ------------------------------------------------------------------ #
        # Internal state                                                     #
        # ------------------------------------------------------------------ #
        # Current configuration (initialized from solver.q0).
        self.current_q = self.kin.clamp(self.ik.q0.copy())

        # Target EE pose; orientation fixed to identity, only position moves.
        self.target_pose = np.eye(4)

        # Create and register the interactive marker in RViz.
        self._create_interactive_marker()

        # Publish an initial joint state so RViz shows the robot.
        self._publish_joint_state(self.current_q)

        # Real-time control loop timer: calls _control_loop() periodically.
        self.control_period = dt
        self.timer = self.create_timer(
            self.control_period,
            self._control_loop,
        )

        self.get_logger().info(
            "Redundant IK node ready (real-time). "
            f"solver={self.ik.__class__.__name__}, "
            f"dt={self.control_period:.3f} s",
        )

    # ---------------------------------------------------------------------- #
    # JointState publishing                                                  #
    # ---------------------------------------------------------------------- #
    def _publish_joint_state(self, q: np.ndarray) -> None:
        """Publish a JointState message for the current configuration."""
        js = JointState()
        js.header = Header()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = self.frame

        js.name = self.kin.joint_names

        # Flatten q to a 1D list of floats.
        js.position = [float(x) for x in q.flatten().tolist()]

        self.joint_pub.publish(js)

    # ---------------------------------------------------------------------- #
    # Interactive marker creation                                            #
    # ---------------------------------------------------------------------- #
    def _create_interactive_marker(self) -> None:
        """Create and register the EE position target interactive marker."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame
        int_marker.name = "target_ee"
        int_marker.description = "EE target (position-only)"

        # Initial position: current end-effector position from q0.
        q0 = self.ik.q0 if hasattr(self.ik, "q0") else np.zeros(6)
        T0, _ = self.kin.forward_kinematics(self.kin.clamp(q0))
        p0 = T0[:3, 3]

        # Place the marker slightly above the EE in z, at least 5 cm.
        int_marker.pose.position.x = float(p0[0])
        int_marker.pose.position.y = float(p0[1])
        int_marker.pose.position.z = float(max(0.05, p0[2]))

        # Internal target pose uses the same position; orientation is identity.
        self.target_pose = _vec_to_pose44(p0)

        # Visual marker: small colored sphere.
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(marker)
        int_marker.controls.append(ctrl)

        # 3-axis translation handles: X, Y, Z.
        ctrl = InteractiveMarkerControl()
        ctrl.orientation.w = 1.0
        ctrl.orientation.x = 1.0
        ctrl.orientation.y = 0.0
        ctrl.orientation.z = 0.0
        ctrl.name = "move_x"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.orientation.w = 1.0
        ctrl.orientation.x = 0.0
        ctrl.orientation.y = 1.0
        ctrl.orientation.z = 0.0
        ctrl.name = "move_y"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        ctrl = InteractiveMarkerControl()
        ctrl.orientation.w = 1.0
        ctrl.orientation.x = 0.0
        ctrl.orientation.y = 0.0
        ctrl.orientation.z = 1.0
        ctrl.name = "move_z"
        ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ctrl)

        # Register feedback callback and apply changes.
        self.im_server.insert(
            int_marker,
            feedback_callback=self._fb,
        )
        self.im_server.applyChanges()

    def _fb(self, fb: InteractiveMarkerFeedback) -> None:
        """Interactive marker feedback callback.

        In this real-time control setup, we simply update target_pose whenever
        the marker pose changes. The control loop then continuously drives the
        robot towards the latest target.
        """
        p = np.array(
            [
                fb.pose.position.x,
                fb.pose.position.y,
                fb.pose.position.z,
            ],
            float,
        )
        self.target_pose = _vec_to_pose44(p)
        self.im_server.applyChanges()

    # ---------------------------------------------------------------------- #
    # Control loop                                                           #
    # ---------------------------------------------------------------------- #
    def _control_loop(self) -> None:
        """Periodic real-time IK loop.

        At each timer tick:
            - Call solver.step() for one CLIK-style update.
            - Publish the resulting JointState.
        """
        q_next, reached, info = self.ik.step(
            self.current_q,
            self.target_pose,
        )
        self.current_q = q_next
        self._publish_joint_state(self.current_q)

        # Optionally log when the target is reached. Use debug to avoid spam.
        if reached:
            pos_err = info.get("pos_err", 0.0)
            self.get_logger().debug(
                f"[{self.ik.__class__.__name__}] Target reached "
                f"(pos_err={pos_err * 1000.0:.2f} mm)",
            )


def main(args: list[str] | None = None) -> None:
    """Entry point for the redundant IK node."""
    rclpy.init(args=args)
    node = RedundantIKNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C.
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

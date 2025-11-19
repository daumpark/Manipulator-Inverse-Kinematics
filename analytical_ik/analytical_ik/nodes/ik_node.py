"""ROS 2 node exposing 2-DOF planar analytical IK via interactive markers."""

import time
from typing import Optional

import numpy as np
import rclpy
from analytical_ik.solvers import Planar2DAnalyticalIK, Planar2DParams
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)


class Analytical2DNode(Node):
    """
    ROS 2 node for 2-DOF planar analytical IK.

    Features:
        - Uses interactive markers to drag a 2D target on the XY plane.
        - Solves analytical IK when the mouse is released.
        - Publishes JointState messages for joint1 and joint2.
    """

    def __init__(self) -> None:
        super().__init__("analytical_2d_node")

        # Declare parameters for link lengths and the base frame.
        self.declare_parameter("L1", 0.35)
        self.declare_parameter("L2", 0.25)
        self.declare_parameter("frame", "base_link")

        # Read parameter values.
        L1 = float(self.get_parameter("L1").value)
        L2 = float(self.get_parameter("L2").value)

        # Create IK solver with given link lengths.
        self.ik = Planar2DAnalyticalIK(Planar2DParams(L1=L1, L2=L2))

        # Seed configuration used to select the closest IK solution.
        self.q_seed = np.zeros(2, dtype=float)

        # Base frame used for marker and joint states.
        self.frame: str = str(self.get_parameter("frame").value)

        # Interactive marker server.
        self.im_server = InteractiveMarkerServer(self, "analytical_2d_marker")

        # Publisher for joint states.
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)

        # Internal state: track whether the marker is being dragged.
        self._drag: bool = False

        # Initial target position slightly inside the maximum reach.
        self.target_xy = np.array([L1 + L2 - 0.05, 0.0], dtype=float)

        # Create the interactive marker in RViz.
        self._create_marker()

        self.get_logger().info(
            "2D Analytical IK node ready. "
            "(drag the marker on XY plane → solve IK on mouse release)",
        )

    # -------------------------------------------------------------------------
    # Interactive marker creation and callback
    # -------------------------------------------------------------------------
    def _create_marker(self) -> None:
        """Create and insert an interactive marker for the 2D target."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame
        int_marker.name = "target_2d"
        int_marker.description = "2D target (XY only)"

        int_marker.pose.position.x = float(self.target_xy[0])
        int_marker.pose.position.y = float(self.target_xy[1])
        int_marker.pose.position.z = 0.0

        # Visual marker: small cylinder.
        marker = Marker()
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.01
        marker.color.r = 0.2
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Control that holds the visual marker.
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)

        # Allow movement along Y axis.
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 1.0
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # Allow movement along X axis.
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 1.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # Register callback and apply changes.
        self.im_server.insert(int_marker, feedback_callback=self._fb)
        self.im_server.applyChanges()

    def _fb(self, fb: InteractiveMarkerFeedback) -> None:
        """
        Interactive marker feedback callback.

        Updates the target position while the marker is being dragged and
        triggers IK solve on mouse button release.
        """
        # Update target (x, y) position from marker pose.
        self.target_xy = np.array(
            [fb.pose.position.x, fb.pose.position.y],
            dtype=float,
        )
        self.im_server.applyChanges()

        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            # Start drag.
            self._drag = True
            return

        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self._drag:
            # End drag, solve IK once.
            self._drag = False
            self._solve_once()
            return

    # -------------------------------------------------------------------------
    # IK solve and publishing
    # -------------------------------------------------------------------------
    def _solve_once(self) -> None:
        """Solve IK once for the current target and publish joint states."""
        t0 = time.perf_counter()
        q, ok, info = self.ik.solve(self.target_xy, q_seed=self.q_seed)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if ok and q is not None:
            # Update seed with the new solution.
            self.q_seed = q.copy()

            # Publish JointState for joint1 and joint2.
            js = JointState()
            js.header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.frame,
            )
            js.name = ["joint1", "joint2"]
            js.position = [float(q[0]), float(q[1])]
            self.joint_pub.publish(js)

            # Compute position error in meters.
            pts = self.ik.fk_points(q)
            pos_err = float(np.linalg.norm(pts[-1] - self.target_xy))

            # Log timing and position error.
            self.get_logger().info(
                f"[Planar2D] t={dt_ms:.2f} ms, "
                f"pos_err={pos_err * 1000.0:.2f} mm, ok=True",
            )
        else:
            # Log failure reason if any.
            reason = info.get("reason", "")
            self.get_logger().warn(
                f"[Planar2D] t={dt_ms:.2f} ms, solve failed: {reason}",
            )


# -----------------------------------------------------------------------------


def main(args: Optional[list] = None) -> None:
    """Entry point for the analytical 2D IK node."""
    rclpy.init(args=args)

    node = Analytical2DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

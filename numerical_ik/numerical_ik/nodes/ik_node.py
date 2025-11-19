"""ROS 2 node exposing numerical IK via interactive markers."""

from typing import Optional

import numpy as np
import rclpy
from interactive_markers.interactive_marker_server import (
    InteractiveMarkerServer,
)
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

from ik_common.common.kinematics import KinematicModel
from ik_common.common.utils import se3_pos_ori_error
from numerical_ik.solvers import (
    JacobianDLS,
    JacobianPinv,
    JacobianTranspose,
)


_SOLVER_MAP = {
    "jt": JacobianTranspose,
    "jpinv": JacobianPinv,
    "dls": JacobianDLS,
}


class NumericalIKNode(Node):
    """
    Numerical IK node in a real-time control style.

    Design:
        - A timer callback is called periodically (control period = dt).
        - Each cycle performs exactly one call to the IK solver's step().
        - The interactive marker only updates the target_pose; the
          control loop continuously drives the robot towards that pose.
    """

    def __init__(self) -> None:
        """Initialize the numerical IK ROS 2 node."""
        super().__init__("numerical_ik_node")

        # ---------------------------------------------------------------------
        # Parameters & solver selection
        # ---------------------------------------------------------------------
        self.declare_parameter("solver", "dls")  # jt | jpinv | dls
        self.kin = KinematicModel()

        solver_key = str(self.get_parameter("solver").value).lower()
        solver_cls = _SOLVER_MAP.get(solver_key, JacobianDLS)
        self.ik = solver_cls(self.kin)
        self.get_logger().info(
            f"Using numerical IK: {self.ik.__class__.__name__}",
        )

        # Tuning parameters.
        for name, default in [
            ("max_iter", 150),
            ("tol_pos", 1e-3),
            ("tol_rot_deg", 1.0),
            ("alpha", 0.7),
            ("w_pos", 1.0),
            ("w_rot", 0.7),
            ("lmbda", 0.05),
            # In a CLIK structure, position/orientation gains are often
            # separated; here we use a single Kp inside the solvers.
            ("Kp", 10.0),
            ("dt", 0.02),  # Control period [s].
        ]:
            self.declare_parameter(name, default)

        # Apply parameter values to the solver.
        self._apply_params()

        # ---------------------------------------------------------------------
        # ROS I/O
        # ---------------------------------------------------------------------
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls_num")

        # ---------------------------------------------------------------------
        # Internal state
        # ---------------------------------------------------------------------
        self.current_q = self.kin.clamp(self.ik.q0.copy())
        self.target_pose = np.eye(4, dtype=float)
        self._last_reached: bool = False

        # Create interactive marker (initialize target_pose from q0).
        self._create_interactive_marker()

        # Publish initial joint state.
        self._publish_joint_state(self.current_q)

        # Create control-loop timer (real-time style IK).
        self.control_period = float(self.get_parameter("dt").value)
        self.timer = self.create_timer(
            self.control_period,
            self._control_loop,
        )

        self.get_logger().info(
            "Numerical IK node ready (real-time style). "
            f"solver={self.ik.__class__.__name__}, "
            f"dt={self.control_period:.3f} s",
        )

    # -------------------------------------------------------------------------
    # Parameter handling
    # -------------------------------------------------------------------------
    def _apply_params(self) -> None:
        """Read ROS parameters and apply them to the solver instance."""
        # Common parameters.
        self.ik.max_iter = int(self.get_parameter("max_iter").value)
        self.ik.tol_pos = float(self.get_parameter("tol_pos").value)
        self.ik.tol_rot = float(
            np.deg2rad(float(self.get_parameter("tol_rot_deg").value)),
        )

        # Optional fields, applied only if they exist in the solver.
        if hasattr(self.ik, "alpha"):
            self.ik.alpha = float(self.get_parameter("alpha").value)
        if hasattr(self.ik, "w_pos"):
            self.ik.w_pos = float(self.get_parameter("w_pos").value)
        if hasattr(self.ik, "w_rot"):
            self.ik.w_rot = float(self.get_parameter("w_rot").value)
        if hasattr(self.ik, "lmbda"):
            self.ik.lmbda = float(self.get_parameter("lmbda").value)

        # Single gain Kp.
        if hasattr(self.ik, "Kp"):
            self.ik.Kp = float(self.get_parameter("Kp").value)

        # Control period (solver and node must share the same dt).
        dt = float(self.get_parameter("dt").value)
        if hasattr(self.ik, "dt"):
            self.ik.dt = dt

    # -------------------------------------------------------------------------
    # Interactive marker
    # -------------------------------------------------------------------------
    def _create_interactive_marker(self) -> None:
        """Create an interactive marker to control the target pose."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_marker_num"
        int_marker.description = "Target pose (Numerical IK, real-time)"

        # Initial pose: EE pose at q0.
        q_neutral = (
            self.ik.q0
            if hasattr(self.ik, "q0")
            else np.zeros(6, dtype=float)
        )
        initial_pose, _ = self.kin.forward_kinematics(
            self.kin.clamp(q_neutral),
        )

        int_marker.pose.position.x = float(initial_pose[0, 3])
        int_marker.pose.position.y = float(initial_pose[1, 3])
        int_marker.pose.position.z = float(initial_pose[2, 3])

        quat = R.from_matrix(initial_pose[:3, :3]).as_quat()
        int_marker.pose.orientation.x = float(quat[0])
        int_marker.pose.orientation.y = float(quat[1])
        int_marker.pose.orientation.z = float(quat[2])
        int_marker.pose.orientation.w = float(quat[3])

        # Internal target_pose is initialized to the same transform.
        self.target_pose = np.eye(4, dtype=float)
        self.target_pose[:3, :3] = initial_pose[:3, :3]
        self.target_pose[:3, 3] = initial_pose[:3, 3]

        # Visualization marker (sphere at the target).
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(marker)
        int_marker.controls.append(ctrl)

        # 3-axis translation controls.
        for name, ox, oy, oz in [
            ("move_x", 1.0, 0.0, 0.0),
            ("move_y", 0.0, 1.0, 0.0),
            ("move_z", 0.0, 0.0, 1.0),
        ]:
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.x = ox
            c.orientation.y = oy
            c.orientation.z = oz
            c.name = name
            c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(c)

        # 3-axis rotation controls.
        for name, ox, oy, oz in [
            ("rotate_x", 1.0, 0.0, 0.0),
            ("rotate_y", 0.0, 1.0, 0.0),
            ("rotate_z", 0.0, 0.0, 1.0),
        ]:
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.x = ox
            c.orientation.y = oy
            c.orientation.z = oz
            c.name = name
            c.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(c)

        # Register feedback callback.
        self.im_server.insert(
            int_marker,
            feedback_callback=self._process_feedback,
        )
        self.im_server.applyChanges()

    @staticmethod
    def _pose_to_mat(pose_msg) -> np.ndarray:
        """
        Convert a geometry_msgs/Pose into a 4x4 homogeneous transform.

        Args:
            pose_msg: Pose message coming from InteractiveMarker feedback.

        Returns:
            4x4 numpy array representing the same pose.
        """
        T = np.eye(4, dtype=float)
        T[:3, 3] = np.asarray(
            [
                pose_msg.position.x,
                pose_msg.position.y,
                pose_msg.position.z,
            ],
            dtype=float,
        )
        Rm = R.from_quat(
            [
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ],
        ).as_matrix()
        T[:3, :3] = Rm
        return T

    def _process_feedback(self, fb: InteractiveMarkerFeedback) -> None:
        """
        Interactive marker feedback callback.

        In the real-time control style used here, we only need to update
        the target_pose on each drag event. The IK itself is continuously
        running in the timer-based control loop.
        """
        self.target_pose = self._pose_to_mat(fb.pose)
        self.im_server.applyChanges()

    # -------------------------------------------------------------------------
    # Control loop (real-time IK)
    # -------------------------------------------------------------------------
    def _publish_joint_state(self, q: np.ndarray) -> None:
        """
        Publish a JointState message for the current joint configuration.

        Args:
            q: Joint configuration array.
        """
        js = JointState()
        js.header = Header()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = "base_link"
        js.name = list(self.kin.joint_names)
        js.position = [float(x) for x in q]
        self.joint_pub.publish(js)

    def _control_loop(self) -> None:
        """
        Periodic control loop callback.

        Behavior:
            - Perform one IK step from current_q towards target_pose.
            - Publish the new joint state.
            - When the target is reached for the first time, log the
              final position and orientation errors.
        """
        q_next, reached, info = self.ik.step(self.current_q, self.target_pose)
        self.current_q = q_next
        self._publish_joint_state(self.current_q)

        # Log once when we enter the "reached" region.
        if reached and not self._last_reached:
            T_cur, _ = self.kin.forward_kinematics(self.current_q)
            pos_err, ori_err = se3_pos_ori_error(T_cur, self.target_pose)
            self.get_logger().info(
                f"[{self.ik.__class__.__name__}] target reached: "
                f"pos_err={pos_err * 1000.0:.2f} mm, "
                f"ori_err={np.rad2deg(ori_err):.2f} deg",
            )

        self._last_reached = reached


def main(args: Optional[list] = None) -> None:
    """Entry point for the numerical IK node."""
    rclpy.init(args=args)
    node = NumericalIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

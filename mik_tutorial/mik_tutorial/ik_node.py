import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

# IK solvers
from .ik_solvers import KinematicModel, JacobianIKSolver, FabrikPureIKSolver

class IKTestNode(Node):
    def __init__(self):
        super().__init__('ik_test_node')

        # Parameters
        self.declare_parameter('solver', 'jacobian')  # 'jacobian' or 'fabrik'
        self.declare_parameter('publish_joint_rate_hz', 30)
        solver_name = self.get_parameter('solver').get_parameter_value().string_value
        pub_rate = self.get_parameter('publish_joint_rate_hz').get_parameter_value().integer_value

        # Kinematics + solver
        kinematics = KinematicModel()
        if solver_name.lower() == 'fabrik':
            self.ik_solver = FabrikPureIKSolver(kinematics)
        else:
            self.ik_solver = JacobianIKSolver(kinematics)

        self.get_logger().info(f"Using IK Solver: {self.ik_solver.__class__.__name__}")

        # Comms
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls")

        # Keep last successful q as seed
        self.q_seed = None

        # Publish throttle
        self._last_pub_time = self.get_clock().now()
        self._min_pub_period = rclpy.duration.Duration(seconds=1.0/float(pub_rate if pub_rate > 0 else 30))

        self.create_interactive_marker()
        self.get_logger().info("IK Test Node with Interactive Marker is running.")

    # ---------- Interactive marker setup ----------
    def create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_marker"
        int_marker.description = "Target Pose for IK"

        # seed pose from a neutral joint vector
        initial_pose, _ = self.ik_solver.kinematics.forward_kinematics(np.array([0, 0.5, 0.5, 0, 0, 0], dtype=float))

        int_marker.pose.position.x = float(initial_pose[0,3])
        int_marker.pose.position.y = float(initial_pose[1,3])
        int_marker.pose.position.z = float(initial_pose[2,3])
        quat = R.from_matrix(initial_pose[:3,:3]).as_quat()
        int_marker.pose.orientation.x = float(quat[0])
        int_marker.pose.orientation.y = float(quat[1])
        int_marker.pose.orientation.z = float(quat[2])
        int_marker.pose.orientation.w = float(quat[3])

        # visual marker
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.05
        box_marker.scale.y = 0.05
        box_marker.scale.z = 0.05
        box_marker.color.r = 0.0
        box_marker.color.g = 0.5
        box_marker.color.b = 0.5
        box_marker.color.a = 1.0

        # add controls
        button_control = InteractiveMarkerControl()
        button_control.always_visible = True
        button_control.markers.append(box_marker)
        int_marker.controls.append(button_control)

        # 6-DOF controls
        # translate
        for name, ox, oy, oz in [
            ("move_x", 1.0, 0.0, 0.0),
            ("move_y", 0.0, 1.0, 0.0),
            ("move_z", 0.0, 0.0, 1.0),
        ]:
            ctrl = InteractiveMarkerControl()
            ctrl.orientation.w = 1.0
            ctrl.orientation.x = ox
            ctrl.orientation.y = oy
            ctrl.orientation.z = oz
            ctrl.name = name
            ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(ctrl)

        # rotate
        for name, ox, oy, oz in [
            ("rotate_x", 1.0, 0.0, 0.0),
            ("rotate_y", 0.0, 1.0, 0.0),
            ("rotate_z", 0.0, 0.0, 1.0),
        ]:
            ctrl = InteractiveMarkerControl()
            ctrl.orientation.w = 1.0
            ctrl.orientation.x = ox
            ctrl.orientation.y = oy
            ctrl.orientation.z = oz
            ctrl.name = name
            ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(ctrl)

        # register callback
        self.im_server.insert(int_marker, feedback_callback=self.process_feedback)
        self.im_server.applyChanges()

    # ---------- Marker callback ----------
    def process_feedback(self, feedback: InteractiveMarkerFeedback):
        # throttle publishes
        now = self.get_clock().now()
        if (now - self._last_pub_time) < self._min_pub_period:
            return

        # build target pose 4x4
        p = feedback.pose.position
        q = feedback.pose.orientation

        target_pose = np.identity(4, dtype=float)
        target_pose[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        target_pose[:3, 3] = np.array([p.x, p.y, p.z], dtype=float)

        # solve IK (pass seed if available)
        joint_angles, is_reachable = self.ik_solver.solve(target_pose, q_seed=self.q_seed)

        marker = self.im_server.get("target_pose_marker")
        control = marker.controls[0]  # box marker
        if is_reachable:
            control.markers[0].color.r = 0.0
            control.markers[0].color.g = 1.0
            control.markers[0].color.b = 0.0
            self.q_seed = joint_angles.copy()
            # publish JointState
            js_msg = JointState()
            js_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
            js_msg.name = self.ik_solver.joint_names
            js_msg.position = list(joint_angles)
            self.joint_pub.publish(js_msg)
            self._last_pub_time = now
        else:
            control.markers[0].color.r = 1.0
            control.markers[0].color.g = 0.0
            control.markers[0].color.b = 0.0

        self.im_server.insert(marker)
        self.im_server.applyChanges()

def main(args=None):
    rclpy.init(args=args)
    node = IKTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

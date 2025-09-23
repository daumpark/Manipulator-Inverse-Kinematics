#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import JointState
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    InteractiveMarkerFeedback, Marker)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import math
import numpy as np # numpy 추가

def clamp(value, min_value, max_value):
    """Clamps a value between a minimum and maximum."""
    return max(min_value, min(value, max_value))

def euler_to_quaternion(roll, pitch, yaw):
    """ Euler angles to Quaternion """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

class PlanarIKNode(Node):
    """
    Solves IK for a 2-DoF planar robot using an Interactive Marker as the target.
    """
    def __init__(self):
        super().__init__('planar_ik_node')

        # Declare and get parameters
        self.declare_parameter('L1', 0.25)
        self.declare_parameter('L2', 0.20)
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('toggle_elbow_every_sec', 1.5)

        self.L1 = self.get_parameter('L1').get_parameter_value().double_value
        self.L2 = self.get_parameter('L2').get_parameter_value().double_value
        rate = self.get_parameter('rate_hz').get_parameter_value().double_value
        toggle_period = self.get_parameter('toggle_elbow_every_sec').get_parameter_value().double_value

        self.get_logger().info(f'IK Node started with L1={self.L1}, L2={self.L2}')

        # Publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Internal state
        self.target_point = Point(x=self.L1, y=0.0, z=0.05) # Initial position
        self.elbow_up = True
        
        # Set up the Interactive Marker Server
        self.server = InteractiveMarkerServer(self, 'planar_ik_interactive_marker')
        self.create_interactive_marker()
        
        # Timers
        self.ik_timer = self.create_timer(1.0 / rate, self.solve_and_publish)
        self.elbow_toggle_timer = self.create_timer(toggle_period, self.toggle_elbow)

    def create_interactive_marker(self):
        """Creates a 2D-movable interactive marker."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_point_marker"
        int_marker.description = "Drag in XY plane to set IK target"
        int_marker.pose.position = self.target_point
        int_marker.scale = 0.3

        visual_marker = Marker()
        visual_marker.type = Marker.SPHERE
        visual_marker.scale.x = 0.05
        visual_marker.scale.y = 0.05
        visual_marker.scale.z = 0.05
        visual_marker.color.r = 0.2
        visual_marker.color.g = 1.0
        visual_marker.color.b = 0.2
        visual_marker.color.a = 0.9

        visual_control = InteractiveMarkerControl()
        visual_control.always_visible = True
        visual_control.markers.append(visual_marker)
        int_marker.controls.append(visual_control)

        # Create a control to move the marker in the XY plane
        move_control = InteractiveMarkerControl()
        move_control.name = "move_xy"

        move_control.orientation = euler_to_quaternion(0, np.pi/2, 0)

        move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        int_marker.controls.append(move_control)

        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.server.applyChanges()

    def process_feedback(self, feedback: InteractiveMarkerFeedback):
        """Callback to update the target position when the marker is moved."""
        self.target_point = feedback.pose.position
        
    def toggle_elbow(self):
        self.elbow_up = not self.elbow_up
        self.get_logger().info(f"Switching to elbow {'UP' if self.elbow_up else 'DOWN'} solution.")

    def solve_and_publish(self):
        """ Core IK solver function, called by a timer. """
        x = self.target_point.x
        y = self.target_point.y
        
        r_sq = x*x + y*y
        r = math.sqrt(r_sq)

        eps = 1e-6
        is_reachable = (abs(self.L1 - self.L2) - eps) <= r <= (self.L1 + self.L2 + eps)
        
        if not is_reachable:
            self.get_logger().warn(f'Target ({x:.2f}, {y:.2f}) is unreachable!', throttle_duration_sec=1.0)
            return

        cos_q2 = (r_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_q2 = clamp(cos_q2, -1.0, 1.0)
        
        sin_q2_abs = math.sqrt(1 - cos_q2**2)
        sin_q2 = sin_q2_abs if self.elbow_up else -sin_q2_abs

        q2 = math.atan2(sin_q2, cos_q2)

        k1 = self.L1 + self.L2 * cos_q2
        k2 = self.L2 * sin_q2
        q1 = math.atan2(y, x) - math.atan2(k2, k1)

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = ['joint1', 'joint2']
        joint_state_msg.position = [q1, q2]
        self.joint_state_pub.publish(joint_state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlanarIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
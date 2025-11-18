# analytical_ik/nodes/ik_node_2d.py
import time, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from std_msgs.msg import Header
from analytical_ik.solvers import Planar2DAnalyticalIK, Planar2DParams

class Analytical2DNode(Node):
    def __init__(self):
        super().__init__('analytical_2d_node')
        self.declare_parameter('L1', 0.35)
        self.declare_parameter('L2', 0.25)
        self.declare_parameter('frame', 'base_link')
        L1 = float(self.get_parameter('L1').value)
        L2 = float(self.get_parameter('L2').value)
        self.ik = Planar2DAnalyticalIK(Planar2DParams(L1=L1, L2=L2))
        self.q_seed = np.zeros(2)

        self.frame = self.get_parameter('frame').value
        self.im_server = InteractiveMarkerServer(self, "analytical_2d_marker")
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        self._drag = False
        self.target_xy = np.array([L1+L2-0.05, 0.0], float)
        self._create_marker()
        self.get_logger().info("2D Analytical IK node ready. (drag on XY plane → solve)")

    def _create_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame
        int_marker.name = "target_2d"
        int_marker.description = "2D target (XY only)"
        int_marker.pose.position.x = float(self.target_xy[0])
        int_marker.pose.position.y = float(self.target_xy[1])
        int_marker.pose.position.z = 0.0

        m = Marker(); m.type=Marker.CYLINDER; m.scale.x=m.scale.y=0.03; m.scale.z=0.01
        m.color.r=m.color.g=0.2; m.color.b=1.0; m.color.a=1.0
        c = InteractiveMarkerControl(); c.always_visible=True; c.markers.append(m); int_marker.controls.append(c)

        c = InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=0.; c.orientation.y=0.; c.orientation.z=1.
        c.name="move_y"; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)
        c = InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=1.; c.orientation.y=0.; c.orientation.z=0.
        c.name="move_x"; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)

        self.im_server.insert(int_marker, feedback_callback=self._fb); self.im_server.applyChanges()

    def _fb(self, fb: InteractiveMarkerFeedback):
        self.target_xy = np.array([fb.pose.position.x, fb.pose.position.y], float)
        self.im_server.applyChanges()
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._drag=True; return
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self._drag:
            self._drag=False; self._solve_once(); return

    def _solve_once(self):
        t0=time.perf_counter()
        q, ok, info = self.ik.solve(self.target_xy, q_seed=self.q_seed)
        dt = (time.perf_counter()-t0)*1000.0
        if ok:
            self.q_seed = q.copy()
            # publish
            js = JointState(); js.header=Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
            js.name=['joint1','joint2']; js.position=[float(q[0]), float(q[1])]
            self.joint_pub.publish(js)
            # error
            pts = self.ik.fk_points(q); pe = float(np.linalg.norm(pts[-1]-self.target_xy))
            self.get_logger().info(f"[Planar2D] t={dt:.2f} ms, pos_err={pe*1000:.2f} mm, ok=True")
        else:
            self.get_logger().warn(f"[Planar2D] t={dt:.2f} ms, solve failed: {info.get('reason','')}")
            
def main(args=None):
    rclpy.init(args=args)
    n=Analytical2DNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()

# redundant_ik/nodes/ik_node.py
import time, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R

from ik_common.common.kinematics import KinematicModel
from redundant_ik.solvers import (
    NullspacePositionOnly, WeightedCLIK, CTP_SVF_SD
)

_SOLVER_MAP = {
    'nullspace': NullspacePositionOnly,
    'wclik':     WeightedCLIK,
    'ctp_svf':   CTP_SVF_SD,
}

def _vec_to_pose44(p):
    T = np.eye(4); T[:3,3] = np.asarray(p, float)
    return T

class RedundantIKNode(Node):
    def __init__(self):
        super().__init__('redundant_ik_node')
        # 파라미터
        self.declare_parameter('solver', 'nullspace')  # nullspace|priority|wclik|ctp_svf
        self.declare_parameter('frame',  'base_link')
        self.declare_parameter('dt',     0.02)
        self.declare_parameter('lam',    1.0e-2)
        self.declare_parameter('k_ns',   0.2)
        self.declare_parameter('Kp',     1.0)
        self.declare_parameter('gamma_max', 0.2)
        self.declare_parameter('nu',        10.0)
        self.declare_parameter('sigma0',    5e-3)

        self.frame = self.get_parameter('frame').value
        self.kin = KinematicModel()
        key = self.get_parameter('solver').value.lower()
        solver_cls = _SOLVER_MAP.get(key, NullspacePositionOnly)
        self.ik = solver_cls(self.kin)
        # 공통 튜닝 주입
        if hasattr(self.ik, 'dt'):         self.ik.dt = float(self.get_parameter('dt').value)
        if hasattr(self.ik, 'lam'):        self.ik.lam = float(self.get_parameter('lam').value)
        if hasattr(self.ik, 'k_ns'):       self.ik.k_ns = float(self.get_parameter('k_ns').value)
        if hasattr(self.ik, 'Kp'):         self.ik.Kp = float(self.get_parameter('Kp').value)
        if hasattr(self.ik, 'gamma_max'):  self.ik.gamma_max = float(self.get_parameter('gamma_max').value)
        if hasattr(self.ik, 'nu'):         self.ik.nu = float(self.get_parameter('nu').value)
        if hasattr(self.ik, 'sigma0'):     self.ik.sigma0 = float(self.get_parameter('sigma0').value)

        self.get_logger().info(f"Using Redundant IK: {self.ik.__class__.__name__}")

        # ROS I/O
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls_redundant")
        self.q_seed = None
        self.latest_target_pose = np.eye(4)
        self._drag=False; self._pose_before=None
        self._create_interactive_marker()
        self.get_logger().info("Redundant IK node ready (drag → solve once).")

    # -------- Interactive Marker --------
    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame
        int_marker.name = "target_ee"
        int_marker.description = "EE target (position-only)"
        # 초기 위치: EE 현재 위치 기준
        q0 = self.ik.q0 if hasattr(self.ik, 'q0') else np.zeros(6)
        T0, _ = self.kin.forward_kinematics(self.kin.clamp(q0))
        p0 = T0[:3,3]
        int_marker.pose.position.x = float(p0[0])
        int_marker.pose.position.y = float(p0[1])
        int_marker.pose.position.z = float(max(0.05, p0[2]))

        # 구체 마커
        m = Marker(); m.type=Marker.SPHERE; m.scale.x=m.scale.y=m.scale.z=0.03
        m.color.r=0.2; m.color.g=0.8; m.color.b=1.0; m.color.a=1.0
        c = InteractiveMarkerControl(); c.always_visible=True; c.markers.append(m); int_marker.controls.append(c)

        # 3축 이동 핸들
        c = InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=1.; c.orientation.y=0.; c.orientation.z=0.
        c.name="move_x"; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)
        c = InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=0.; c.orientation.y=1.; c.orientation.z=0.
        c.name="move_y"; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)
        c = InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=0.; c.orientation.y=0.; c.orientation.z=1.
        c.name="move_z"; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)

        self.im_server.insert(int_marker, feedback_callback=self._fb); self.im_server.applyChanges()

    def _fb(self, fb: InteractiveMarkerFeedback):
        # Pose 추적
        p = np.array([fb.pose.position.x, fb.pose.position.y, fb.pose.position.z], float)
        self.latest_target_pose = _vec_to_pose44(p)
        if fb.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            return
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            if not self._drag:
                self._drag=True; self._pose_before=self.latest_target_pose.copy(); return
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self._drag:
            self._drag=False; self._solve_once(); return

    # -------- Solve-once --------
    def _solve_once(self):
        t0 = time.perf_counter()
        q, ok, info = self.ik.solve(self.latest_target_pose, q_seed=self.q_seed)
        ms = (time.perf_counter()-t0)*1000.0
        if ok:
            self.q_seed = q.copy()
            js = JointState()
            js.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
            js.name = self.kin.joint_names
            js.position = [float(x) for x in q.flatten().tolist()]
            self.joint_pub.publish(js)
            self.get_logger().info(f"Solved [{self.ik.__class__.__name__}] in {ms:.1f} ms; iters={info.get('iters_total','-')}")
        else:
            self.get_logger().warn(f"IK did not converge (iters={info.get('iters_total','-')}).")

def main(args=None):
    rclpy.init(args=args)
    node = RedundantIKNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

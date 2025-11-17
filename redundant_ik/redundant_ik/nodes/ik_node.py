# redundant_ik/nodes/ik_node.py
import rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from std_msgs.msg import Header

from ik_common.common.kinematics import KinematicModel
from redundant_ik.solvers import (
    NullspacePositionOnly, WeightedCLIK, SVF
)

_SOLVER_MAP = {
    'nullspace': NullspacePositionOnly,
    'wclik':     WeightedCLIK,
    'svf':       SVF,
}

def _vec_to_pose44(p):
    T = np.eye(4); T[:3,3] = np.asarray(p, float)
    return T

class RedundantIKNode(Node):
    """
    Redundant IK node (실시간 CLIK 스타일)
    - 타이머 콜백에서 각 solver의 step()을 호출해 한 스텝만 진행
    - Interactive marker는 목표 EE 위치만 계속 갱신
    """
    def __init__(self):
        super().__init__('redundant_ik_node')
        # 파라미터
        self.declare_parameter('solver', 'nullspace')  # nullspace|wclik|svf
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
        dt = float(self.get_parameter('dt').value)
        if hasattr(self.ik, 'dt'):         self.ik.dt = dt
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

        # 내부 상태
        self.current_q = self.kin.clamp(self.ik.q0.copy())
        # 목표 EE pose (position-only, orientation = identity)
        self.target_pose = np.eye(4)

        # Interactive Marker 설정
        self._create_interactive_marker()

        # 초기 joint state publish
        self._publish_joint_state(self.current_q)

        # 제어 루프 타이머 (실시간 IK)
        self.control_period = dt
        self.timer = self.create_timer(self.control_period, self._control_loop)

        self.get_logger().info(
            f"Redundant IK node ready (real-time). "
            f"solver={self.ik.__class__.__name__}, dt={self.control_period:.3f} s"
        )

    # -------- JointState publish --------
    def _publish_joint_state(self, q):
        js = JointState()
        js.header = Header()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = self.frame
        js.name = self.kin.joint_names
        js.position = [float(x) for x in q.flatten().tolist()]
        self.joint_pub.publish(js)

    # -------- Interactive Marker --------
    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame
        int_marker.name = "target_ee"
        int_marker.description = "EE target (position-only)"

        # 초기 위치: EE 현재 위치 기준
        q0 = self.ik.q0 if hasattr(self.ik, 'q0') else np.zeros(6)
        T0, _ = self.kin.forward_kinematics(self.kin.clamp(q0))
        p0 = T0[:3, 3]

        int_marker.pose.position.x = float(p0[0])
        int_marker.pose.position.y = float(p0[1])
        int_marker.pose.position.z = float(max(0.05, p0[2]))

        # 내부 target_pose도 동일 위치로 세팅 (orientation = identity)
        self.target_pose = _vec_to_pose44(p0)

        # 구체 마커
        m = Marker()
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.color.r = 0.2
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 1.0
        c = InteractiveMarkerControl()
        c.always_visible = True
        c.markers.append(m)
        int_marker.controls.append(c)

        # 3축 이동 핸들
        c = InteractiveMarkerControl()
        c.orientation.w = 1.0; c.orientation.x = 1.; c.orientation.y = 0.; c.orientation.z = 0.
        c.name = "move_x"; c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(c)

        c = InteractiveMarkerControl()
        c.orientation.w = 1.0; c.orientation.x = 0.; c.orientation.y = 1.; c.orientation.z = 0.
        c.name = "move_y"; c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(c)

        c = InteractiveMarkerControl()
        c.orientation.w = 1.0; c.orientation.x = 0.; c.orientation.y = 0.; c.orientation.z = 1.
        c.name = "move_z"; c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(c)

        self.im_server.insert(int_marker, feedback_callback=self._fb)
        self.im_server.applyChanges()

    def _fb(self, fb: InteractiveMarkerFeedback):
        """
        실시간 제어에서는 드래그 이벤트와 상관 없이,
        pose가 바뀔 때마다 target_pose만 계속 갱신해주면 된다.
        """
        p = np.array(
            [fb.pose.position.x, fb.pose.position.y, fb.pose.position.z],
            float
        )
        self.target_pose = _vec_to_pose44(p)
        self.im_server.applyChanges()

    # -------- 제어 루프 --------
    def _control_loop(self):
        """
        주기적으로 호출되는 실시간 IK 루프.
        - 현재 q에서 solver.step() 한 번 수행
        - q_next를 joint_states로 publish
        """
        q_next, reached, info = self.ik.step(self.current_q, self.target_pose)
        self.current_q = q_next
        self._publish_joint_state(self.current_q)

        # 필요하면 pos_err 로그 찍기 (너무 자주 찍으면 지저분하니 간단히)
        if reached:
            self.get_logger().debug(
                f"[{self.ik.__class__.__name__}] target reached "
                f"(pos_err={info.get('pos_err', 0.0)*1000:.2f} mm)"
            )

def main(args=None):
    rclpy.init(args=args)
    node = RedundantIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

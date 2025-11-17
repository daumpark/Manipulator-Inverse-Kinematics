# numerical_ik/nodes/ik_node.py
import rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

from ik_common.common.kinematics import KinematicModel
from ik_common.common.utils import se3_pos_ori_error
from numerical_ik.solvers import JacobianTranspose, JacobianPinv, JacobianDLS

_SOLVER_MAP = {
    'jt':    JacobianTranspose,
    'jpinv': JacobianPinv,
    'dls':   JacobianDLS,
}

class NumericalIKNode(Node):
    """
    실시간 제어 스타일의 Numerical IK 노드.
    - 타이머 콜백이 주기적으로 호출됨 (제어 주기 = dt)
    - 각 주기마다 IK solver의 step()을 한 번만 호출
    - Interactive marker는 target_pose만 갱신
    """
    def __init__(self):
        super().__init__('numerical_ik_node')

        # 파라미터 & IK 선택
        self.declare_parameter('solver', 'dls')   # jt | jpinv | dls
        self.kin = KinematicModel()

        solver_key = self.get_parameter('solver').value.lower()
        solver_cls = _SOLVER_MAP.get(solver_key, JacobianDLS)
        self.ik = solver_cls(self.kin)
        self.get_logger().info(f"Using Numerical IK: {self.ik.__class__.__name__}")

        # 튜닝 파라미터
        for p, v in [
            ('max_iter', 150), ('tol_pos', 1e-3), ('tol_rot_deg', 1.0),
            ('alpha', 0.7), ('w_pos', 1.0), ('w_rot', 0.7),
            ('lmbda', 0.05),
            # CLIK 구조를 가정할 때 보통 pos/rot gain을 나누지만,
            # 여기서는 단일 Kp를 solvers 안에서 사용 중.
            ('Kp', 10.0),
            ('dt', 0.02),  # 제어 주기 [s]
        ]:
            self.declare_parameter(p, v)

        self._apply_params()

        # ROS I/O
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls_num")

        # 내부 상태
        self.current_q = self.kin.clamp(self.ik.q0.copy())
        self.target_pose = np.eye(4)
        self._last_reached = False

        # Interactive marker 생성 (초기 target_pose 세팅)
        self._create_interactive_marker()

        # 초기 자세 publish
        self._publish_joint_state(self.current_q)

        # 제어 루프 타이머 생성 (실시간 IK)
        self.control_period = float(self.get_parameter('dt').value)
        self.timer = self.create_timer(self.control_period, self._control_loop)

        self.get_logger().info(
            f"Numerical IK node ready (real-time style). "
            f"solver={self.ik.__class__.__name__}, dt={self.control_period:.3f} s"
        )

    # ---------------- 파라미터 적용 ----------------
    def _apply_params(self):
        # 공통
        self.ik.max_iter = int(self.get_parameter('max_iter').value)
        self.ik.tol_pos  = float(self.get_parameter('tol_pos').value)
        self.ik.tol_rot  = np.deg2rad(float(self.get_parameter('tol_rot_deg').value))

        # 선택적인 필드들
        if hasattr(self.ik, 'alpha'):
            self.ik.alpha = float(self.get_parameter('alpha').value)
        if hasattr(self.ik, 'w_pos'):
            self.ik.w_pos = float(self.get_parameter('w_pos').value)
        if hasattr(self.ik, 'w_rot'):
            self.ik.w_rot = float(self.get_parameter('w_rot').value)
        if hasattr(self.ik, 'lmbda'):
            self.ik.lmbda = float(self.get_parameter('lmbda').value)

        # Kp는 단일 gain으로 사용
        if hasattr(self.ik, 'Kp'):
            self.ik.Kp = float(self.get_parameter('Kp').value)

        # 제어 주기 (solver와 node 모두 같은 dt 사용)
        dt = float(self.get_parameter('dt').value)
        if hasattr(self.ik, 'dt'):
            self.ik.dt = dt

    # ---------------- Interactive Marker ----------------
    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_marker_num"
        int_marker.description = "Target Pose (Numerical IK, real-time)"

        # 초기 pose: q0에서의 EE pose
        q_neutral = self.ik.q0 if hasattr(self.ik, 'q0') else np.zeros(6)
        initial_pose, _ = self.kin.forward_kinematics(self.kin.clamp(q_neutral))

        int_marker.pose.position.x = float(initial_pose[0, 3])
        int_marker.pose.position.y = float(initial_pose[1, 3])
        int_marker.pose.position.z = float(initial_pose[2, 3])

        quat = R.from_matrix(initial_pose[:3, :3]).as_quat()
        int_marker.pose.orientation.x = float(quat[0])
        int_marker.pose.orientation.y = float(quat[1])
        int_marker.pose.orientation.z = float(quat[2])
        int_marker.pose.orientation.w = float(quat[3])

        # 내부 target_pose도 동일하게 세팅
        self.target_pose = np.eye(4)
        self.target_pose[:3, :3] = initial_pose[:3, :3]
        self.target_pose[:3, 3]  = initial_pose[:3, 3]

        # 시각화용 마커
        m = Marker()
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.color.r = 0.2
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 1.0
        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(m)
        int_marker.controls.append(ctrl)

        # 3축 이동
        for name, ox, oy, oz in [
            ("move_x", 1., 0., 0.),
            ("move_y", 0., 1., 0.),
            ("move_z", 0., 0., 1.),
        ]:
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.x = ox
            c.orientation.y = oy
            c.orientation.z = oz
            c.name = name
            c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(c)

        # 3축 회전
        for name, ox, oy, oz in [
            ("rotate_x", 1., 0., 0.),
            ("rotate_y", 0., 1., 0.),
            ("rotate_z", 0., 0., 1.),
        ]:
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.x = ox
            c.orientation.y = oy
            c.orientation.z = oz
            c.name = name
            c.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(c)

        # 콜백 등록
        self.im_server.insert(int_marker, feedback_callback=self._process_feedback)
        self.im_server.applyChanges()

    @staticmethod
    def _pose_to_mat(pose_msg):
        T = np.identity(4, float)
        T[:3, 3] = np.array(
            [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
            float
        )
        Rm = R.from_quat([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]).as_matrix()
        T[:3, :3] = Rm
        return T

    def _process_feedback(self, fb: InteractiveMarkerFeedback):
        """
        실시간 제어에서는 마우스 드래그 이벤트마다
        target_pose만 갱신해주면 된다.
        IK 자체는 타이머 루프에서 계속 동작.
        """
        self.target_pose = self._pose_to_mat(fb.pose)
        self.im_server.applyChanges()

    # ---------------- 제어 루프 (실시간 IK) ----------------
    def _publish_joint_state(self, q):
        js = JointState()
        js.header = Header()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = "base_link"
        js.name = self.kin.joint_names
        js.position = [float(x) for x in q]
        self.joint_pub.publish(js)

    def _control_loop(self):
        """
        주기적으로 호출되는 제어 루프.
        - 현재 q에서 IK 한 스텝(step) 수행
        - 결과 q_next를 joint_states로 publish
        """
        q_next, reached, info = self.ik.step(self.current_q, self.target_pose)
        self.current_q = q_next
        self._publish_joint_state(self.current_q)

        # 처음으로 tolerance 안에 들어왔을 때만 로그 한 번 찍기
        if reached and not self._last_reached:
            T_cur, _ = self.kin.forward_kinematics(self.current_q)
            pos_err, ori_err = se3_pos_ori_error(T_cur, self.target_pose)
            self.get_logger().info(
                f"[{self.ik.__class__.__name__}] target reached: "
                f"pos_err={pos_err*1000:.2f} mm, "
                f"ori_err={np.rad2deg(ori_err):.2f} deg"
            )
        self._last_reached = reached


def main(args=None):
    rclpy.init(args=args)
    node = NumericalIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

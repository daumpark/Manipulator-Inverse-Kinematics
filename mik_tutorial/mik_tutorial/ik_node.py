import time
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

# IK solvers
from .ik_solvers import KinematicModel, JacobianIKSolver, FABRIKSolver


def se3_pos_ori_error(T_cur: np.ndarray, T_tar: np.ndarray):
    dp = T_cur[:3, 3] - T_tar[:3, 3]
    R_err = T_cur[:3, :3] @ T_tar[:3, :3].T
    tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(tr))
    return float(np.linalg.norm(dp)), ang


class IKTestNode(Node):
    def __init__(self):
        super().__init__('ik_test_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('solver', 'jacobian')  # 'jacobian' | 'fabrik'

        # Jacobian
        self.declare_parameter('jacobian_max_iter', 150)
        self.declare_parameter('jacobian_tol_pos', 1e-3)
        self.declare_parameter('jacobian_tol_rot_deg', 1.0)
        self.declare_parameter('jacobian_lambda', 0.05)
        self.declare_parameter('jacobian_alpha', 0.7)
        self.declare_parameter('jacobian_w_pos', 1.0)
        self.declare_parameter('jacobian_w_rot', 0.7)

        # FABRIK
        self.declare_parameter('fabrik_max_iter', 120)
        self.declare_parameter('fabrik_tol_pos', 1e-3)
        self.declare_parameter('fabrik_align_passes', 3)
        self.declare_parameter('fabrik_tol_align', 2e-3)

        # ---------------- Kinematics + solver ----------------
        self.kinematics = KinematicModel()

        solver_name = self.get_parameter('solver').get_parameter_value().string_value
        if solver_name.lower() == 'fabrik':
            self.ik_solver = FABRIKSolver(self.kinematics)
            self.ik_solver.max_iter_fabrik = int(self.get_parameter('fabrik_max_iter').value)
            self.ik_solver.tol_fabrik = float(self.get_parameter('fabrik_tol_pos').value)
            self.ik_solver.align_passes = int(self.get_parameter('fabrik_align_passes').value)
            self.ik_solver.tol_align = float(self.get_parameter('fabrik_tol_align').value)
        else:
            self.ik_solver = JacobianIKSolver(self.kinematics)
            self.ik_solver.max_iter = int(self.get_parameter('jacobian_max_iter').value)
            self.ik_solver.tol_pos = float(self.get_parameter('jacobian_tol_pos').value)
            self.ik_solver.tol_rot = np.deg2rad(float(self.get_parameter('jacobian_tol_rot_deg').value))
            self.ik_solver.lmbda   = float(self.get_parameter('jacobian_lambda').value)
            self.ik_solver.alpha   = float(self.get_parameter('jacobian_alpha').value)
            self.ik_solver.w_pos   = float(self.get_parameter('jacobian_w_pos').value)
            self.ik_solver.w_rot   = float(self.get_parameter('jacobian_w_rot').value)

        self.get_logger().info(f"Using IK Solver: {self.ik_solver.__class__.__name__}")

        # ---------------- Publishers / Servers ----------------
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls")

        # 상태
        self.q_seed = None
        self.latest_target_pose = np.eye(4, dtype=float)

        # 드래그 상태 추적 (mouse_point_valid 토글 기반)
        self._drag_active = False
        self._pose_before_drag = None

        # Interactive Marker
        self._create_interactive_marker()

        self.get_logger().info("IK Test Node (drag-toggle one-shot) is running.")

    # ---------- Interactive marker ----------
    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_marker"
        int_marker.description = "Target Pose for IK"

        # seed pose
        q_neutral = np.array([0, 0.5, 0.5, 0, 0, 0], dtype=float)
        initial_pose, _ = self.ik_solver.kinematics.forward_kinematics(q_neutral)

        int_marker.pose.position.x = float(initial_pose[0, 3])
        int_marker.pose.position.y = float(initial_pose[1, 3])
        int_marker.pose.position.z = float(initial_pose[2, 3])
        quat = R.from_matrix(initial_pose[:3, :3]).as_quat()
        int_marker.pose.orientation.x = float(quat[0])
        int_marker.pose.orientation.y = float(quat[1])
        int_marker.pose.orientation.z = float(quat[2])
        int_marker.pose.orientation.w = float(quat[3])

        # latest target
        self.latest_target_pose[:3, :3] = initial_pose[:3, :3]
        self.latest_target_pose[:3, 3] = initial_pose[:3, 3]

        # visual marker (박스)
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.05
        box_marker.scale.y = 0.05
        box_marker.scale.z = 0.05
        box_marker.color.r = 0.6
        box_marker.color.g = 0.6
        box_marker.color.b = 0.6
        box_marker.color.a = 1.0

        # control (버튼+이동+회전)
        button_control = InteractiveMarkerControl()
        button_control.always_visible = True
        button_control.markers.append(box_marker)
        int_marker.controls.append(button_control)

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

        self.im_server.insert(int_marker, feedback_callback=self._process_feedback)
        self.im_server.applyChanges()  # 반드시 호출

    @staticmethod
    def _pose_to_mat(pose_msg):
        T = np.identity(4, dtype=float)
        T[:3, :3] = R.from_quat([pose_msg.orientation.x,
                                 pose_msg.orientation.y,
                                 pose_msg.orientation.z,
                                 pose_msg.orientation.w]).as_matrix()
        T[:3, 3] = np.array([pose_msg.position.x,
                             pose_msg.position.y,
                             pose_msg.position.z], dtype=float)
        return T

    def _process_feedback(self, fb: InteractiveMarkerFeedback):
        # 최신 타깃 포즈 갱신 (항상 반영)
        self.latest_target_pose = self._pose_to_mat(fb.pose)
        self.im_server.applyChanges()

        # 1) 우선 MOUSE_DOWN/MOUSE_UP 이벤트가 있다면 그걸로 처리
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._drag_active = True
            self._pose_before_drag = self.latest_target_pose.copy()
            self.get_logger().debug("[IM] MOUSE_DOWN → drag start")
            return

        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            # 드래그 종료로 간주
            if self._drag_active:
                self._drag_active = False
                self.get_logger().debug("[IM] MOUSE_UP → drag end → solve once")
                self._solve_once(self.latest_target_pose)
            return

        # 2) 일부 환경에선 event_type이 POSE_UPDATE만 오고,
        #    mouse_point_valid false(잡는 중) → true(놓는 순간)로 한 번 전환됨.
        #    그 전환을 드래그 시작/종료로 간주한다.
        if not self._drag_active and (fb.mouse_point_valid is False):
            # 드래그 시작(처음 false를 만난 시점)
            self._drag_active = True
            self._pose_before_drag = self.latest_target_pose.copy()
            self.get_logger().debug("[IM] mouse_point_valid=false → drag start")
            return

        if self._drag_active and (fb.mouse_point_valid is True):
            # 드래그 종료(처음 true로 돌아오는 시점)
            self._drag_active = False
            self.get_logger().debug("[IM] mouse_point_valid=true → drag end → solve once")
            self._solve_once(self.latest_target_pose)
            return

        # 나머지(계속 드래그 중 POSE_UPDATE 등)는 무시

    # ---------- one-shot solve ----------
    def _solve_once(self, target_pose: np.ndarray):
        solver_name = self.ik_solver.__class__.__name__

        # 시작 로그
        self.get_logger().info(f"[{solver_name}] solve started")

        t0 = time.perf_counter()
        q_sol, ok, info = self.ik_solver.solve(target_pose, q_seed=self.q_seed)
        dt = time.perf_counter() - t0

        # 누적 seed 업데이트
        self.q_seed = q_sol.copy()

        # 최종 EE 에러 계산
        T_try, _ = self.ik_solver.kinematics.forward_kinematics(q_sol)
        pos_err, ori_err = se3_pos_ori_error(T_try, target_pose)

        # 종료 로그
        self.get_logger().info(
            f"[{solver_name}] done: t={dt*1000.0:.1f} ms, "
            f"iters={int(info.get('iters_total', -1))}, "
            f"pos_err={pos_err*1000.0:.2f} mm, "
            f"ori_err={np.rad2deg(ori_err):.2f} deg, "
            f"ok={ok}"
        )

        # JointState 발행(한 번)
        js_msg = JointState()
        js_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
        js_msg.name = self.ik_solver.joint_names
        js_msg.position = list(q_sol)
        self.joint_pub.publish(js_msg)


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

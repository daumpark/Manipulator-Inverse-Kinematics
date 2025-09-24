import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from geometry_msgs.msg import Point

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
        self.declare_parameter('solver', 'jacobian')              # 'jacobian'|'fabrik'
        self.declare_parameter('publish_joint_rate_hz', 30)        # IK 주기(Hz)
        self.declare_parameter('joint_roles_override', '')         # "pivot,hinge,hinge,pivot,hinge,pivot"

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
        self.declare_parameter('fabrik_tol_rot_deg', 1.0)
        self.declare_parameter('fabrik_q_gain', 0.9)
        self.declare_parameter('fabrik_q_reg', 0.02)
        self.declare_parameter('fabrik_smooth_q', 0.30)
        self.declare_parameter('fabrik_max_step_deg', 6.0)
        self.declare_parameter('fabrik_orient_gate_mul', 5.0)

        # Debug viz
        self.declare_parameter('debug_viz', True)                  # 디버그 시각화 on/off
        self.declare_parameter('debug_max_points', 80)             # 라인/점 다운샘플 포인트 수

        # ---------------- Kinematics + solver ----------------
        self.kinematics = KinematicModel()

        roles_str = self.get_parameter('joint_roles_override').get_parameter_value().string_value.strip()
        if roles_str:
            roles = [s.strip().lower() for s in roles_str.split(',')]
            if len(roles) == 6 and all(r in ('pivot', 'hinge', 'pris') for r in roles):
                self.kinematics.set_joint_roles(roles)
                self.get_logger().info(f"Joint roles override: {self.kinematics.describe_joint_roles()}")

        solver_name = self.get_parameter('solver').get_parameter_value().string_value
        if solver_name.lower() == 'fabrik':
            self.ik_solver = FABRIKSolver(self.kinematics)
            self.ik_solver.max_iter = int(self.get_parameter('fabrik_max_iter').value)
            self.ik_solver.tol_pos = float(self.get_parameter('fabrik_tol_pos').value)
            self.ik_solver.tol_rot = np.deg2rad(float(self.get_parameter('fabrik_tol_rot_deg').value))
            self.ik_solver.q_gain  = float(self.get_parameter('fabrik_q_gain').value)
            self.ik_solver.q_reg   = float(self.get_parameter('fabrik_q_reg').value)
            self.ik_solver.smooth_q = float(self.get_parameter('fabrik_smooth_q').value)
            self.ik_solver.max_step_deg = float(self.get_parameter('fabrik_max_step_deg').value)
            self.ik_solver.orient_gate_mul = float(self.get_parameter('fabrik_orient_gate_mul').value)
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
        # 상태(색상) & 디버그 모두 visualization_marker로 송신 (ns/id로 구분)
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls")

        # 내부 상태
        self.q_seed = None
        self.latest_target_pose = np.eye(4, dtype=float)
        self.debug_viz = bool(self.get_parameter('debug_viz').value)
        self.debug_max_points = int(self.get_parameter('debug_max_points').value)

        # Interactive Marker (insert 1회만)
        self._create_interactive_marker()

        # 주기 IK 업데이트 루프
        ik_rate_hz = float(self.get_parameter('publish_joint_rate_hz').get_parameter_value().integer_value)
        period = 1.0 / (ik_rate_hz if ik_rate_hz > 0 else 30.0)
        self.timer = self.create_timer(period, self.update_loop)

        self.get_logger().info("IK Test Node with Interactive Marker is running.")

    # ---------- Interactive marker ----------
    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_marker"
        int_marker.description = "Target Pose for IK"

        # seed pose
        q_neutral = np.array([0, 0.5, 0.5, 0, 0, 0], dtype=float)
        initial_pose, _ = self.ik_solver.kinematics.forward_kinematics(q_neutral)

        int_marker.pose.position.x = float(initial_pose[0,3])
        int_marker.pose.position.y = float(initial_pose[1,3])
        int_marker.pose.position.z = float(initial_pose[2,3])
        quat = R.from_matrix(initial_pose[:3,:3]).as_quat()
        int_marker.pose.orientation.x = float(quat[0])
        int_marker.pose.orientation.y = float(quat[1])
        int_marker.pose.orientation.z = float(quat[2])
        int_marker.pose.orientation.w = float(quat[3])

        # latest target
        self.latest_target_pose[:3,:3] = initial_pose[:3,:3]
        self.latest_target_pose[:3, 3] = initial_pose[:3, 3]

        # visual marker (고정 색)
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.05
        box_marker.scale.y = 0.05
        box_marker.scale.z = 0.05
        box_marker.color.r = 0.6
        box_marker.color.g = 0.6
        box_marker.color.b = 0.6
        box_marker.color.a = 1.0

        # control
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
        self.im_server.applyChanges()

    def _process_feedback(self, feedback: InteractiveMarkerFeedback):
        p = feedback.pose.position
        q = feedback.pose.orientation
        T = np.identity(4, dtype=float)
        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = np.array([p.x, p.y, p.z], dtype=float)
        self.latest_target_pose = T

    # ---------- status overlay (상태 색상만) ----------
    def _publish_status_overlay(self, color_rgb, pose_T):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "ik_status"
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.scale.x = 0.051
        m.scale.y = 0.051
        m.scale.z = 0.051
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pose_T[:3, 3]
        quat = R.from_matrix(pose_T[:3, :3]).as_quat()
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = quat
        m.color.r, m.color.g, m.color.b = color_rgb
        m.color.a = 0.45
        self.marker_pub.publish(m)

    # ---------- debug markers (라인/점/텍스트 + FABRIK 체인) ----------
    def _publish_debug(self, debug_dict, target_pose, solver_name: str):
        if not self.debug_viz or not debug_dict:
            return

        # 샘플링
        pos_list = np.asarray(debug_dict.get("ee_positions", []), dtype=float)
        if pos_list.size == 0:
            return
        N = pos_list.shape[0]
        max_pts = max(5, int(self.debug_max_points))
        stride = int(np.ceil(N / max_pts))
        idx = np.arange(0, N, stride, dtype=int)
        pos_sampled = pos_list[idx]

        # LINE_STRIP
        line = Marker()
        line.header.frame_id = "base_link"
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = "ik_debug"
        line.id = 1000
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.003
        line.color.r, line.color.g, line.color.b, line.color.a = (0.2, 0.8, 1.0, 0.8)
        line.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pos_sampled]
        self.marker_pub.publish(line)

        # SPHERE_LIST
        sp = Marker()
        sp.header.frame_id = "base_link"
        sp.header.stamp = line.header.stamp
        sp.ns = "ik_debug"
        sp.id = 1001
        sp.type = Marker.SPHERE_LIST
        sp.action = Marker.ADD
        sp.scale.x = sp.scale.y = sp.scale.z = 0.008
        sp.color.r, sp.color.g, sp.color.b, sp.color.a = (1.0, 0.6, 0.2, 0.8)
        sp.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pos_sampled]
        self.marker_pub.publish(sp)

        # TEXT (마지막 EE 근처)
        pe = float(debug_dict["pos_errs"][-1]) if "pos_errs" in debug_dict and len(debug_dict["pos_errs"])>0 else 0.0
        oe = float(debug_dict["ori_errs"][-1]) if "ori_errs" in debug_dict and len(debug_dict["ori_errs"])>0 else 0.0
        txt = Marker()
        txt.header.frame_id = "base_link"
        txt.header.stamp = line.header.stamp
        txt.ns = "ik_debug"
        txt.id = 1002
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.scale.z = 0.03
        txt.color.r, txt.color.g, txt.color.b, txt.color.a = (1.0, 1.0, 1.0, 0.9)
        last = pos_list[-1]
        txt.pose.position.x = float(last[0]); txt.pose.position.y = float(last[1]); txt.pose.position.z = float(last[2] + 0.05)
        gate_note = ""
        if "gates" in debug_dict:
            gate_note = f", gate_on={int(bool(debug_dict['gates'][-1]))}"
        txt.text = f"{solver_name} iters={debug_dict.get('iters', len(pos_list))}\npe={pe*1000:.1f} mm, oe={np.rad2deg(oe):.2f} deg{gate_note}"
        self.marker_pub.publish(txt)

        # FABRIK 체인(마지막 반복의 p0..p6) LINE_LIST
        chain = debug_dict.get("final_chain_p", None)
        if chain is not None and len(chain) >= 2:
            ll = Marker()
            ll.header.frame_id = "base_link"
            ll.header.stamp = line.header.stamp
            ll.ns = "ik_debug"
            ll.id = 1003
            ll.type = Marker.LINE_LIST
            ll.action = Marker.ADD
            ll.scale.x = 0.004
            ll.color.r, ll.color.g, ll.color.b, ll.color.a = (0.9, 0.9, 0.1, 0.8)
            pts = []
            for i in range(len(chain)-1):
                a = chain[i]; b = chain[i+1]
                pts.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
                pts.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
            ll.points = pts
            self.marker_pub.publish(ll)

    # ---------- main loop ----------
    def update_loop(self):
        target_pose = self.latest_target_pose.copy()

        # IK solve (누적 seed)
        joint_angles, _ = self.ik_solver.solve(target_pose, q_seed=self.q_seed)
        self.q_seed = joint_angles.copy()

        # 상태 색상 (EE 기준)
        T_try, _ = self.ik_solver.kinematics.forward_kinematics(joint_angles)
        pos_err, ori_err = se3_pos_ori_error(T_try, target_pose)
        tol_pos = getattr(self.ik_solver, 'tol_pos', 1e-3)
        tol_rot = getattr(self.ik_solver, 'tol_rot', np.deg2rad(1.0))

        if pos_err < tol_pos and ori_err < tol_rot:
            color = (0.0, 1.0, 0.0)
        elif pos_err < 5.0 * tol_pos and ori_err < 5.0 * tol_rot:
            color = (1.0, 0.65, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        self._publish_status_overlay(color, target_pose)

        # 디버그 이터레이션 시각화
        solver_name = self.ik_solver.__class__.__name__
        self._publish_debug(getattr(self.ik_solver, "debug", None), target_pose, solver_name)

        # JointState 발행
        js_msg = JointState()
        js_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
        js_msg.name = self.ik_solver.joint_names
        js_msg.position = list(joint_angles)
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

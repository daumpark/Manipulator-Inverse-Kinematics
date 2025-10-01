# numerical_ik/nodes/ik_node.py
import time, rclpy, numpy as np
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
    'jt': JacobianTranspose,
    'jpinv': JacobianPinv,
    'dls': JacobianDLS,
}

class NumericalIKNode(Node):
    def __init__(self):
        super().__init__('numerical_ik_node')
        self.declare_parameter('solver', 'dls')   # jt | jpinv | dls | clik
        self.kin = KinematicModel()

        solver_key = self.get_parameter('solver').value.lower()
        solver_cls = _SOLVER_MAP.get(solver_key, JacobianDLS)
        self.ik = solver_cls(self.kin)
        self.get_logger().info(f"Using Numerical IK: {self.ik.__class__.__name__}")

        # expose tunables
        for p, v in [
            ('max_iter', 150), ('tol_pos', 1e-3), ('tol_rot_deg', 1.0),
            ('alpha', 0.7), ('w_pos', 1.0), ('w_rot', 0.7),
            ('lmbda', 0.05), ('Kp_pos', 2.0), ('Kp_rot', 1.5), ('dt', 0.02),
        ]:
            self.declare_parameter(p, v)
        self._apply_params()

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls_num")
        self.q_seed = None
        self.latest_target_pose = np.eye(4)
        self._drag_active=False
        self._pose_before_drag=None
        self._create_interactive_marker()
        self.get_logger().info("Numerical IK node ready (drag → solve once).")

    def _apply_params(self):
        self.ik.max_iter = int(self.get_parameter('max_iter').value)
        self.ik.tol_pos = float(self.get_parameter('tol_pos').value)
        self.ik.tol_rot = np.deg2rad(float(self.get_parameter('tol_rot_deg').value))
        if hasattr(self.ik, 'alpha'): self.ik.alpha = float(self.get_parameter('alpha').value)
        if hasattr(self.ik, 'w_pos'): self.ik.w_pos = float(self.get_parameter('w_pos').value)
        if hasattr(self.ik, 'w_rot'): self.ik.w_rot = float(self.get_parameter('w_rot').value)
        if hasattr(self.ik, 'lmbda'): self.ik.lmbda = float(self.get_parameter('lmbda').value)
        if hasattr(self.ik, 'Kp_pos'): self.ik.Kp_pos = float(self.get_parameter('Kp_pos').value)
        if hasattr(self.ik, 'Kp_rot'): self.ik.Kp_rot = float(self.get_parameter('Kp_rot').value)
        if hasattr(self.ik, 'dt'): self.ik.dt = float(self.get_parameter('dt').value)

    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_marker_num"
        int_marker.description = "Target Pose (Numerical IK)"
        q_neutral = np.array([0, 0.5, 0.5, 0, 0, 0], float)
        initial_pose, _ = self.kin.forward_kinematics(q_neutral)
        int_marker.pose.position.x, int_marker.pose.position.y, int_marker.pose.position.z = map(float, initial_pose[:3,3])
        quat = R.from_matrix(initial_pose[:3,:3]).as_quat()
        int_marker.pose.orientation.x, int_marker.pose.orientation.y, int_marker.pose.orientation.z, int_marker.pose.orientation.w = map(float, quat)
        self.latest_target_pose[:3,:3] = initial_pose[:3,:3]
        self.latest_target_pose[:3,3] = initial_pose[:3,3]
        box = Marker(); box.type=Marker.CUBE; box.scale.x=box.scale.y=box.scale.z=0.05; box.color.r=box.color.g=box.color.b=0.6; box.color.a=1.0
        ctrl = InteractiveMarkerControl(); ctrl.always_visible=True; ctrl.markers.append(box); int_marker.controls.append(ctrl)
        for name,ox,oy,oz in [("move_x",1.,0.,0.),("move_y",0.,1.,0.),("move_z",0.,0.,1.)]:
            c=InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=ox; c.orientation.y=oy; c.orientation.z=oz; c.name=name; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)
        for name,ox,oy,oz in [("rotate_x",1.,0.,0.),("rotate_y",0.,1.,0.),("rotate_z",0.,0.,1.)]:
            c=InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=ox; c.orientation.y=oy; c.orientation.z=oz; c.name=name; c.interaction_mode=InteractiveMarkerControl.ROTATE_AXIS; int_marker.controls.append(c)
        self.im_server.insert(int_marker, feedback_callback=self._process_feedback)
        self.im_server.applyChanges()

    @staticmethod
    def _pose_to_mat(pose_msg):
        T = np.identity(4, float)
        from scipy.spatial.transform import Rotation as R
        T[:3,:3] = R.from_quat([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]).as_matrix()
        T[:3,3] = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], float)
        return T

    def _process_feedback(self, fb: InteractiveMarkerFeedback):
        self.latest_target_pose = self._pose_to_mat(fb.pose)
        self.im_server.applyChanges()
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._drag_active=True; self._pose_before_drag=self.latest_target_pose.copy(); return
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self._drag_active:
            self._drag_active=False; self._solve_once(self.latest_target_pose); return
        if not self._drag_active and (fb.mouse_point_valid is False):
            self._drag_active=True; self._pose_before_drag=self.latest_target_pose.copy(); return
        if self._drag_active and (fb.mouse_point_valid is True):
            self._drag_active=False; self._solve_once(self.latest_target_pose); return

    def _solve_once(self, target_pose):
        name=self.ik.__class__.__name__
        self.get_logger().info(f"[{name}] solve started")
        t0=time.perf_counter()
        q_sol, ok, info = self.ik.solve(target_pose, q_seed=self.q_seed)
        dt = time.perf_counter()-t0
        self.q_seed = q_sol.copy()
        T_try,_ = self.kin.forward_kinematics(q_sol)
        pos_err,ori_err = se3_pos_ori_error(T_try, target_pose)
        self.get_logger().info(f"[{name}] t={dt*1000:.1f} ms, iters={int(info.get('iters_total',-1))}, pos_err={pos_err*1000:.2f} mm, ori_err={np.rad2deg(ori_err):.2f} deg, ok={ok}")
        js = JointState(); js.header=Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
        js.name=self.ik.joint_names; js.position=list(q_sol); self.joint_pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    node=NumericalIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

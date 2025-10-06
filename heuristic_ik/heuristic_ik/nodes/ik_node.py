# heuristic_ik/nodes/ik_node.py
import time, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

from ik_common.common.kinematics import KinematicModel
from ik_common.common.utils import se3_pos_ori_error
from heuristic_ik.solvers import FABRIK_R, DQ_FABRIK

class HeuristicIKNode(Node):
    def __init__(self):
        super().__init__('heuristic_ik_node')
        self.kin = KinematicModel()

        # parameters
        self.declare_parameter('solver', 'DQ_FABRIK')  # DQ_FABRIK or FABRIK_R
        solver_name = str(self.get_parameter('solver').value).strip().upper()

        for p,v in [('fabrik_max_iter',120), ('fabrik_tol_pos',1e-3),
                    ('fabrik_align_passes',2), ('fabrik_tol_align',2e-3)]:
            self.declare_parameter(p, v)
        self.declare_parameter('ori_max_iter', 30)
        self.declare_parameter('ori_tol_deg', 1.0)
        self.declare_parameter('ori_step', 1.0)

        if solver_name == 'FABRIK_R':
            self.ik = FABRIK_R(self.kin)
        else:
            self.ik = DQ_FABRIK(self.kin)

        # apply params
        self.ik.max_iter_fabrik = int(self.get_parameter('fabrik_max_iter').value)
        self.ik.tol_fabrik      = float(self.get_parameter('fabrik_tol_pos').value)
        self.ik.align_passes    = int(self.get_parameter('fabrik_align_passes').value)
        self.ik.tol_align       = float(self.get_parameter('fabrik_tol_align').value)

        if hasattr(self.ik, 'ori_max_iter'):
            self.ik.ori_max_iter = int(self.get_parameter('ori_max_iter').value)
        if hasattr(self.ik, 'ori_tol_rad'):
            self.ik.ori_tol_rad = np.deg2rad(float(self.get_parameter('ori_tol_deg').value))
        if hasattr(self.ik, 'ori_step'):
            self.ik.ori_step = float(self.get_parameter('ori_step').value)

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "ik_controls_heur")
        self.q_seed=None; self.latest_target_pose=np.eye(4)
        self._drag=False; self._pose_before=None
        self._create_interactive_marker()
        self.get_logger().info(f"Heuristic IK node ready ({self.ik.__class__.__name__}).")

    def _create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id="base_link"
        int_marker.name="target_pose_marker_heur"
        int_marker.description="Target Pose (Heuristic IK)"
        q_neutral=self.ik.q0 if hasattr(self.ik, 'q0') else np.zeros(6)
        T0,_=self.kin.forward_kinematics(q_neutral)
        int_marker.pose.position.x, int_marker.pose.position.y, int_marker.pose.position.z = map(float, T0[:3,3])
        quat=R.from_matrix(T0[:3,:3]).as_quat()
        int_marker.pose.orientation.x, int_marker.pose.orientation.y, int_marker.pose.orientation.z, int_marker.pose.orientation.w = map(float, quat)
        self.latest_target_pose[:3,:3]=T0[:3,:3]; self.latest_target_pose[:3,3]=T0[:3,3]
        m = Marker(); m.type=Marker.SPHERE; m.scale.x=m.scale.y=m.scale.z=0.03
        m.color.r=0.2; m.color.g=0.8; m.color.b=1.0; m.color.a=1.0
        ctrl=InteractiveMarkerControl(); ctrl.always_visible=True; ctrl.markers.append(m); int_marker.controls.append(ctrl)
        for name,ox,oy,oz in [("move_x",1.,0.,0.),("move_y",0.,1.,0.),("move_z",0.,0.,1.)]:
            c=InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=ox; c.orientation.y=oy; c.orientation.z=oz; c.name=name; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)
        for name,ox,oy,oz in [("rotate_x",1.,0.,0.),("rotate_y",0.,1.,0.),("rotate_z",0.,0.,1.)]:
            c=InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=ox; c.orientation.y=oy; c.orientation.z=oz; c.name=name; c.interaction_mode=InteractiveMarkerControl.ROTATE_AXIS; int_marker.controls.append(c)
        self.im_server.insert(int_marker, feedback_callback=self._feedback); self.im_server.applyChanges()

    @staticmethod
    def _pose_to_mat(p):
        from scipy.spatial.transform import Rotation as R
        T=np.identity(4,float)
        T[:3,:3]=R.from_quat([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]).as_matrix()
        T[:3,3]=np.array([p.position.x,p.position.y,p.position.z],float)
        return T

    def _feedback(self, fb: InteractiveMarkerFeedback):
        self.latest_target_pose = self._pose_to_mat(fb.pose)
        self.im_server.applyChanges()
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._drag=True; self._pose_before=self.latest_target_pose.copy(); return
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self._drag:
            self._drag=False; self._solve_once(self.latest_target_pose); return
        if not self._drag and (fb.mouse_point_valid is False):
            self._drag=True; self._pose_before=self.latest_target_pose.copy(); return
        if self._drag and (fb.mouse_point_valid is True):
            self._drag=False; self._solve_once(self.latest_target_pose); return

    def _solve_once(self, target_pose):
        name=self.ik.__class__.__name__
        self.get_logger().info(f"[{name}] solve started")
        t0=time.perf_counter()
        q, ok, info = self.ik.solve(target_pose, q_seed=self.q_seed)
        dt=time.perf_counter()-t0
        self.q_seed=q.copy()
        T_try,_=self.kin.forward_kinematics(q)
        pos_err,ori_err=se3_pos_ori_error(T_try, target_pose)
        self.get_logger().info(f"[{name}] t={dt*1000:.1f} ms, iters={int(info.get('iters_total',-1))}, pos_err={pos_err*1000:.2f} mm, ori_err={np.rad2deg(ori_err):.2f} deg, ok={ok}")
        js=JointState(); js.header=Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
        js.name=self.ik.joint_names; js.position=list(q); self.joint_pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    node=HeuristicIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()

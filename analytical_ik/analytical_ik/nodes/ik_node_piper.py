# analytical_ik/nodes/ik_node_piper.py
import time, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

from analytical_ik.solvers import PiperAnalyticalIK
# 공통 FK/오차 계산을 위해 재사용 (URDF, Pinocchio)
from ik_common.common.kinematics import KinematicModel
from ik_common.common.utils import se3_pos_ori_error

class PiperAnalyticalNode(Node):
    def __init__(self):
        super().__init__('piper_analytical_node')
        self.declare_parameter('dh_type', 'standard')  # 'standard' | 'modified'
        self.kin = KinematicModel()
        self.ik = PiperAnalyticalIK(dh_type=self.get_parameter('dh_type').value.lower())
        self.q_seed = np.zeros(6)

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.im_server = InteractiveMarkerServer(self, "piper_analytical_marker")
        self.latest_target_pose = np.eye(4)
        self._drag=False
        self._create_marker()
        self.get_logger().info("Piper Analytical IK node ready. (drag pose → solve once)")

    def _create_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_pose_piper_analytical"
        int_marker.description = "Target Pose (Analytical IK - PiPER)"

        qn = np.array([0,0.5,0.5,0,0,0], float)
        T0,_ = self.kin.forward_kinematics(qn)
        int_marker.pose.position.x, int_marker.pose.position.y, int_marker.pose.position.z = map(float, T0[:3,3])
        quat = R.from_matrix(T0[:3,:3]).as_quat()
        int_marker.pose.orientation.x, int_marker.pose.orientation.y, int_marker.pose.orientation.z, int_marker.pose.orientation.w = map(float, quat)
        self.latest_target_pose[:3,:3] = T0[:3,:3]; self.latest_target_pose[:3,3] = T0[:3,3]

        box=Marker(); box.type=Marker.CUBE; box.scale.x=box.scale.y=box.scale.z=0.05; box.color.r=box.color.g=0.3; box.color.b=1.0; box.color.a=1.0
        ctrl=InteractiveMarkerControl(); ctrl.always_visible=True; ctrl.markers.append(box); int_marker.controls.append(ctrl)
        for name,ox,oy,oz in [("move_x",1.,0.,0.),("move_y",0.,1.,0.),("move_z",0.,0.,1.)]:
            c=InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=ox; c.orientation.y=oy; c.orientation.z=oz; c.name=name; c.interaction_mode=InteractiveMarkerControl.MOVE_AXIS; int_marker.controls.append(c)
        for name,ox,oy,oz in [("rotate_x",1.,0.,0.),("rotate_y",0.,1.,0.),("rotate_z",0.,0.,1.)]:
            c=InteractiveMarkerControl(); c.orientation.w=1.0; c.orientation.x=ox; c.orientation.y=oy; c.orientation.z=oz; c.name=name; c.interaction_mode=InteractiveMarkerControl.ROTATE_AXIS; int_marker.controls.append(c)
        self.im_server.insert(int_marker, feedback_callback=self._fb); self.im_server.applyChanges()

    @staticmethod
    def _pose_to_mat(p):
        T = np.identity(4, float)
        T[:3,:3] = R.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]).as_matrix()
        T[:3,3] = np.array([p.position.x, p.position.y, p.position.z], float)
        return T

    def _fb(self, fb: InteractiveMarkerFeedback):
        self.latest_target_pose = self._pose_to_mat(fb.pose)
        self.im_server.applyChanges()
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._drag=True; return
        if fb.event_type == InteractiveMarkerFeedback.MOUSE_UP and self._drag:
            self._drag=False; self._solve_once(self.latest_target_pose); return

    def _solve_once(self, T_target):
        t0=time.perf_counter()
        # URDF에서 joint limits 가져와 필터링
        jlimits = list(zip(self.kin.lower.tolist(), self.kin.upper.tolist()))
        all_solutions = self.ik.compute_ik(T_target, joint_limits=jlimits, q_seed=self.q_seed, filter_by_limits=True)
        dt=(time.perf_counter()-t0)*1000.0

        if not all_solutions:
            self.get_logger().warn(f"[PiPER-Analytical] t={dt:.1f} ms, no solution (within limits)")
            return

        q = all_solutions[0]
        self.q_seed = q.copy()

        # 오차/성공 로그
        T_try,_ = self.kin.forward_kinematics(q)
        pe, oe = se3_pos_ori_error(T_try, T_target)
        self.get_logger().info(f"[PiPER-Analytical] t={dt:.1f} ms, pos_err={pe*1000:.2f} mm, ori_err={np.rad2deg(oe):.2f} deg, ok={pe<1e-3 and oe<np.deg2rad(1.0)}")

        # 퍼블리시
        js = JointState(); js.header=Header(stamp=self.get_clock().now().to_msg(), frame_id="base_link")
        js.name = self.kin.joint_names; js.position = list(q); self.joint_pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    n=PiperAnalyticalNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()

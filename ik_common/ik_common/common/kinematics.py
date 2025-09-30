# ik_common/common/kinematics.py
import numpy as np
import pinocchio as pin
import os, xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from rclpy.logging import get_logger

class KinematicModel:
    def __init__(self):
        try:
            desc_share = get_package_share_directory('piper_description')
            urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
        except Exception:
            urdf_file = '/mnt/data/piper_no_gripper_description.urdf'
        if not os.path.exists(urdf_file):
            raise FileNotFoundError(f"PiPER URDF not found: {urdf_file}")

        self.urdf_file = urdf_file
        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_file)
        self.model = self.robot.model
        self.data = self.robot.data
        self.logger = get_logger('ik_common.kinematics')

        # EE frame
        self.ee_joint_name = "joint6"
        try:
            self.ee_frame_id = self.model.getFrameId(self.ee_joint_name)
        except Exception:
            self.ee_frame_id = self.model.nframes - 1

        # joint names
        names = []
        for i in range(1, 7):
            nm = f"joint{i}"
            try:
                if self.model.getJointId(nm) > 0:
                    names.append(nm)
            except Exception:
                pass
        if len(names) != 6:
            names = [j.name for j in self.model.joints if j.nq > 0][:6]
        self.joint_names = names

        # parse URDF
        lower, upper = [], []
        axis_map_local = {}
        type_map = {}
        root = ET.parse(urdf_file).getroot()
        lim_map = {}
        for j in root.findall('joint'):
            nm = j.attrib.get('name', '')
            jtype = j.attrib.get('type', '')
            type_map[nm] = jtype
            lim = j.find('limit')
            if lim is not None and 'lower' in lim.attrib and 'upper' in lim.attrib:
                lim_map[nm] = (float(lim.attrib['lower']), float(lim.attrib['upper']))
            elif jtype == 'continuous':
                lim_map[nm] = (-np.inf, np.inf)
            ax = j.find('axis')
            if ax is not None and 'xyz' in ax.attrib:
                xyz = np.fromstring(ax.attrib['xyz'], sep=' ', dtype=float)
                if np.linalg.norm(xyz) < 1e-12:
                    xyz = np.array([0.0, 0.0, 1.0])
                axis_map_local[nm] = xyz / np.linalg.norm(xyz)
        for nm in self.joint_names:
            lo, hi = lim_map.get(nm, (-np.inf, np.inf))
            lower.append(lo); upper.append(hi)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.joint_axis_local = axis_map_local
        self.joint_type = type_map

        # compute r_ee_to_j6_ee
        try:
            qzero = np.zeros(6, float)
            self._full_fk(qzero)
            j6_name = self.joint_names[-1]
            self.j6_id = self.model.getJointId(j6_name)
            Tj6 = self.data.oMi[self.j6_id]
            Tee = self.data.oMf[self.ee_frame_id]
            r_world = Tj6.translation - Tee.translation
            R_ee = Tee.rotation
            self.r_ee_to_j6_ee = R_ee.T @ r_world
        except Exception:
            self.j6_id = None
            self.r_ee_to_j6_ee = np.zeros(3)

    def _full_fk(self, q):
        q = np.asarray(q, float).flatten()
        if q.size != 6:
            raise ValueError("FK expects 6 joint values.")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def forward_kinematics(self, q):
        self._full_fk(q)
        Ts = []
        for nm in self.joint_names:
            jid = self.model.getJointId(nm)
            Ts.append(self.data.oMi[jid].homogeneous.copy())
        fid = self.ee_frame_id
        if not (0 <= fid < len(self.data.oMf)):
            fid = len(self.data.oMf) - 1
        T_ee = self.data.oMf[fid].homogeneous.copy()
        return T_ee, Ts

    def jacobian(self, q, ref_frame=pin.ReferenceFrame.LOCAL):
        q = np.asarray(q, float).flatten()
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.ee_frame_id
        if not (0 <= fid < self.model.nframes):
            fid = self.model.nframes - 1
        return pin.computeFrameJacobian(self.model, self.data, q, fid, ref_frame)

    def clamp(self, q):
        return np.minimum(np.maximum(q, self.lower), self.upper)

    def joint_axis_world(self, q, joint_name):
        self._full_fk(q)
        jid = self.model.getJointId(joint_name)
        Rw = self.data.oMi[jid].rotation
        a_local = self.joint_axis_local.get(joint_name, np.array([0.0, 0.0, 1.0]))
        a_world = Rw @ a_local
        n = np.linalg.norm(a_world)
        return a_world if n < 1e-12 else a_world / n

    def chain_points(self, q):
        self._full_fk(q)
        pts = [np.zeros(3, float)]
        for nm in self.joint_names:
            jid = self.model.getJointId(nm)
            pts.append(self.data.oMi[jid].translation.copy())
        return np.asarray(pts)

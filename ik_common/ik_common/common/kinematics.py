"""Kinematic model utilities for the PiPER robot."""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory
from rclpy.logging import get_logger


class KinematicModel:
    """
    Wrapper around the Pinocchio robot model for the PiPER robot.

    This class:
      * Loads the robot URDF and builds the Pinocchio model.
      * Extracts joint names, joint limits, and joint axes.
      * Provides forward kinematics and Jacobian computation.
      * Provides convenient helpers such as clamping, chain points, etc.
    """

    def __init__(self) -> None:
        """Load the URDF, build the model, and pre-compute basic info."""
        try:
            # Locate the URDF from the ROS 2 package share directory.
            desc_share = get_package_share_directory("piper_description")
            urdf_file = os.path.join(
                desc_share,
                "urdf",
                "piper_no_gripper_description.urdf",
            )
        except Exception:
            # Fallback path (e.g. when running outside ROS 2 environment).
            urdf_file = "/mnt/data/piper_no_gripper_description.urdf"

        # Ensure that the URDF actually exists.
        if not os.path.exists(urdf_file):
            raise FileNotFoundError(
                f"PiPER URDF not found: {urdf_file}"
            )

        self.urdf_file: str = urdf_file

        # Build the Pinocchio robot wrapper from the URDF.
        self.robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_file)
        self.model: pin.Model = self.robot.model
        self.data: pin.Data = self.robot.data

        # ROS 2 logger for debugging and information messages.
        self.logger = get_logger("ik_common.kinematics")

        # ---------------------------------------------------------------------
        # End-effector (EE) frame handling
        # ---------------------------------------------------------------------

        # Default end-effector joint name used to find the EE frame.
        self.ee_joint_name: str = "joint6"

        try:
            # Try to find the frame ID associated with the EE joint.
            self.ee_frame_id: int = self.model.getFrameId(self.ee_joint_name)
        except Exception:
            # If it fails, fall back to the last frame in the model.
            self.ee_frame_id = self.model.nframes - 1

        # ---------------------------------------------------------------------
        # Joint name discovery
        # ---------------------------------------------------------------------

        # Try to collect joint names "joint1" ~ "joint6".
        names: List[str] = []
        for index in range(1, 7):
            joint_name = f"joint{index}"
            try:
                # getJointId returns 0 if the joint is not found.
                if self.model.getJointId(joint_name) > 0:
                    names.append(joint_name)
            except Exception:
                # Ignore joints that cannot be found.
                pass

        # If something went wrong, fall back to the first 6 actuated joints.
        if len(names) != 6:
            names = [
                joint.name
                for joint in self.model.joints
                if joint.nq > 0
            ][:6]

        self.joint_names: List[str] = names

        # ---------------------------------------------------------------------
        # Parse URDF for joint limits and axes
        # ---------------------------------------------------------------------

        lower: List[float] = []
        upper: List[float] = []
        axis_map_local: Dict[str, np.ndarray] = {}
        type_map: Dict[str, str] = {}

        # Parse the URDF using ElementTree.
        root = ET.parse(urdf_file).getroot()

        # Temporary map from joint name to limits.
        lim_map: Dict[str, Tuple[float, float]] = {}

        for joint in root.findall("joint"):
            name = joint.attrib.get("name", "")
            joint_type = joint.attrib.get("type", "")

            # Store joint type (revolute, prismatic, continuous, etc.).
            type_map[name] = joint_type

            # -------------------------------
            # Joint limit parsing
            # -------------------------------
            limit = joint.find("limit")
            if (
                limit is not None
                and "lower" in limit.attrib
                and "upper" in limit.attrib
            ):
                lim_map[name] = (
                    float(limit.attrib["lower"]),
                    float(limit.attrib["upper"]),
                )
            elif joint_type == "continuous":
                # Continuous joint => no explicit bounds.
                lim_map[name] = (-np.inf, np.inf)

            # -------------------------------
            # Joint axis parsing
            # -------------------------------
            axis = joint.find("axis")
            if axis is not None and "xyz" in axis.attrib:
                xyz = np.fromstring(
                    axis.attrib["xyz"],
                    sep=" ",
                    dtype=float,
                )
                # If axis is zero-length, fall back to z-axis.
                if np.linalg.norm(xyz) < 1e-12:
                    xyz = np.array([0.0, 0.0, 1.0])
                axis_map_local[name] = xyz / np.linalg.norm(xyz)

        # Build joint limit arrays in the order of self.joint_names.
        for name in self.joint_names:
            lo, hi = lim_map.get(name, (-np.inf, np.inf))
            lower.append(lo)
            upper.append(hi)

        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.joint_axis_local: Dict[str, np.ndarray] = axis_map_local
        self.joint_type: Dict[str, str] = type_map

        # ---------------------------------------------------------------------
        # Compute vector from EE to joint6 in EE frame
        # ---------------------------------------------------------------------

        try:
            # Use zero configuration to compute the relative position at rest.
            q_zero = np.zeros(6, dtype=float)
            self._full_fk(q_zero)

            j6_name = self.joint_names[-1]
            self.j6_id: int = self.model.getJointId(j6_name)

            # Homogeneous transform of joint 6 in world frame.
            T_j6 = self.data.oMi[self.j6_id]

            # Homogeneous transform of EE frame in world frame.
            T_ee = self.data.oMf[self.ee_frame_id]

            # Vector from EE to joint 6 in world coordinates.
            r_world = T_j6.translation - T_ee.translation

            # Rotation matrix of EE frame in world coordinates.
            R_ee = T_ee.rotation

            # Express the vector in EE frame coordinates.
            self.r_ee_to_j6_ee = R_ee.T @ r_world
        except Exception:
            # Fallback if anything goes wrong during this computation.
            self.j6_id = None
            self.r_ee_to_j6_ee = np.zeros(3, dtype=float)

    # -------------------------------------------------------------------------
    # Internal helper: full forward kinematics
    # -------------------------------------------------------------------------
    def _full_fk(self, q: np.ndarray) -> None:
        """
        Run full forward kinematics for a given joint configuration.

        Args:
            q: 1D array of 6 joint values.
        """
        q = np.asarray(q, dtype=float).flatten()
        if q.size != 6:
            raise ValueError("FK expects 6 joint values.")

        # Update all joint placements in the Pinocchio model.
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def forward_kinematics(
        self,
        q: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute the end-effector pose and poses of all joints in the chain.

        Args:
            q: 1D array of 6 joint values.

        Returns:
            T_ee: 4x4 homogeneous transform of the end-effector in world frame.
            Ts: List of 4x4 homogeneous transforms, one for each joint in
                `self.joint_names`, in world frame.
        """
        self._full_fk(q)

        # Collect homogeneous transforms of each joint.
        transforms: List[np.ndarray] = []
        for name in self.joint_names:
            joint_id = self.model.getJointId(name)
            transforms.append(self.data.oMi[joint_id].homogeneous.copy())

        # EE frame may be out of bounds; clamp to the last frame if needed.
        frame_id = self.ee_frame_id
        if not (0 <= frame_id < len(self.data.oMf)):
            frame_id = len(self.data.oMf) - 1

        # Homogeneous transform of the end-effector frame.
        T_ee = self.data.oMf[frame_id].homogeneous.copy()
        return T_ee, transforms

    def jacobian(
        self,
        q: np.ndarray,
        ref_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> np.ndarray:
        """
        Compute the end-effector Jacobian for the given configuration.

        Args:
            q: 1D array of 6 joint values.
            ref_frame: Reference frame for the Jacobian (LOCAL, WORLD, etc.).

        Returns:
            6xN Jacobian matrix of the EE frame, where N is the number of
            joints in the model.
        """
        q = np.asarray(q, dtype=float).flatten()

        # Update joint Jacobians and frame placements.
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.ee_frame_id
        if not (0 <= frame_id < self.model.nframes):
            frame_id = self.model.nframes - 1

        # Compute the Jacobian of the EE frame in the requested reference frame.
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            frame_id,
            ref_frame,
        )

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """
        Clamp a joint configuration to the joint limits.

        Args:
            q: 1D array of joint values.

        Returns:
            Clamped joint array where each element is within [lower, upper].
        """
        q = np.asarray(q, dtype=float)
        return np.minimum(np.maximum(q, self.lower), self.upper)

    def joint_axis_world(
        self,
        q: np.ndarray,
        joint_name: str,
    ) -> np.ndarray:
        """
        Get the rotation axis of a joint expressed in world coordinates.

        Args:
            q: 1D array of 6 joint values.
            joint_name: Name of the joint whose axis is requested.

        Returns:
            3D unit vector representing the axis direction in world frame.
        """
        self._full_fk(q)

        joint_id = self.model.getJointId(joint_name)
        # Rotation of the joint frame in world coordinates.
        R_world = self.data.oMi[joint_id].rotation

        # Local joint axis (fallback: z-axis if not found).
        axis_local = self.joint_axis_local.get(
            joint_name,
            np.array([0.0, 0.0, 1.0], dtype=float),
        )

        # Transform local axis into world frame.
        axis_world = R_world @ axis_local
        norm = np.linalg.norm(axis_world)

        # Normalize; if the vector is nearly zero, just return as is.
        if norm < 1e-12:
            return axis_world
        return axis_world / norm

    def chain_points(self, q: np.ndarray) -> np.ndarray:
        """
        Get the 3D positions of each joint in the chain.

        Args:
            q: 1D array of 6 joint values.

        Returns:
            (N+1, 3) array of points in world frame, where:
                - index 0 is the base origin (0, 0, 0),
                - indices 1..N are the positions of each joint in order.
        """
        self._full_fk(q)

        # Start with the base origin.
        points: List[np.ndarray] = [np.zeros(3, dtype=float)]

        # Append the translation of each joint.
        for name in self.joint_names:
            joint_id = self.model.getJointId(name)
            points.append(self.data.oMi[joint_id].translation.copy())

        return np.asarray(points, dtype=float)

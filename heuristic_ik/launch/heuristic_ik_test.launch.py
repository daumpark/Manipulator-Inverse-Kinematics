import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("heuristic_ik")
    urdf_file = os.path.join(pkg_share, "urdf", "planar2d.urdf")

    rviz_cfg_path = os.path.join(pkg_share, "rviz", "planar2d.rviz")
    rviz_cfg = rviz_cfg_path if os.path.exists(rviz_cfg_path) else ""

    with open(urdf_file, "r") as f:
        robot_desc = f.read()

    return LaunchDescription(
        [
            DeclareLaunchArgument("solver", default_value="ccd"),  # "ccd" or "fabrik"
            DeclareLaunchArgument("max_iter", default_value="200"),
            DeclareLaunchArgument("tol", default_value="0.001"),

            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_desc}],
            ),

            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                output="screen",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "map",
                    "base_link",
                ],
            ),

            Node(
                package="heuristic_ik",
                executable="ik_node",
                output="screen",
                parameters=[
                    {"solver": LaunchConfiguration("solver")},
                    {"max_iter": LaunchConfiguration("max_iter")},
                    {"tol": LaunchConfiguration("tol")},
                ],
            ),

            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=(["-d", rviz_cfg] if rviz_cfg else []),
            ),
        ]
    )

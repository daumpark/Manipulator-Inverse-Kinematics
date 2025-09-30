# analytical_ik/launch/analytical_piper_test.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('analytical_ik')
    desc_share = get_package_share_directory('piper_description')
    urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
    rviz_cfg = os.path.join(pkg_share, 'rviz', 'piper.rviz') if os.path.exists(os.path.join(pkg_share,'rviz','piper.rviz')) else ''

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('dh_type', default_value='standard'),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_desc}]),
        # map -> base_link
        Node(package='tf2_ros', executable='static_transform_publisher', output='screen',
             arguments=['0','0','0','0','0','0','map','base_link']),
        Node(package='analytical_ik', executable='ik_node_piper', output='screen',
             parameters=[{'dh_type': LaunchConfiguration('dh_type')}]),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=(['-d', rviz_cfg] if rviz_cfg else [])),
    ])

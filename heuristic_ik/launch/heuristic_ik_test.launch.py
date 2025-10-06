# heuristic_ik/launch/heuristic_ik_test.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = get_package_share_directory('heuristic_ik')
    desc_share = get_package_share_directory('piper_description')
    urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
    rviz_cfg = os.path.join(pkg_share, 'rviz', 'piper.rviz') if os.path.exists(os.path.join(pkg_share,'rviz','piper.rviz')) else ''

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('solver', default_value='DQ_FABRIK'),
        DeclareLaunchArgument('fabrik_max_iter', default_value='120'),
        DeclareLaunchArgument('fabrik_tol_pos', default_value='0.001'),
        DeclareLaunchArgument('fabrik_align_passes', default_value='2'),
        DeclareLaunchArgument('fabrik_tol_align', default_value='0.002'),
        DeclareLaunchArgument('ori_max_iter', default_value='30'),
        DeclareLaunchArgument('ori_tol_deg', default_value='1.0'),
        DeclareLaunchArgument('ori_step', default_value='1.0'),

        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_desc}]),
        Node(package='tf2_ros', executable='static_transform_publisher', output='screen',
             arguments=['0','0','0','0','0','0','map','base_link']),
        Node(package='heuristic_ik', executable='ik_node', output='screen', parameters=[
            {'solver': LaunchConfiguration('solver')},
            {'fabrik_max_iter': LaunchConfiguration('fabrik_max_iter')},
            {'fabrik_tol_pos': LaunchConfiguration('fabrik_tol_pos')},
            {'fabrik_align_passes': LaunchConfiguration('fabrik_align_passes')},
            {'fabrik_tol_align': LaunchConfiguration('fabrik_tol_align')},
            {'ori_max_iter': LaunchConfiguration('ori_max_iter')},
            {'ori_tol_deg': LaunchConfiguration('ori_tol_deg')},
            {'ori_step': LaunchConfiguration('ori_step')},
        ]),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=(['-d', rviz_cfg] if rviz_cfg else [])),
    ])

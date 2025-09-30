# numerical_ik/launch/numerical_ik_test.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = get_package_share_directory('numerical_ik')
    desc_share = get_package_share_directory('piper_description')
    urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
    rviz_cfg = os.path.join(pkg_share, 'rviz', 'piper.rviz') if os.path.exists(os.path.join(pkg_share,'rviz','piper.rviz')) else ''

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('solver', default_value='dls'),
        DeclareLaunchArgument('max_iter', default_value='150'),
        DeclareLaunchArgument('tol_pos', default_value='0.001'),
        DeclareLaunchArgument('tol_rot_deg', default_value='1.0'),
        DeclareLaunchArgument('alpha', default_value='0.7'),
        DeclareLaunchArgument('w_pos', default_value='1.0'),
        DeclareLaunchArgument('w_rot', default_value='0.7'),
        DeclareLaunchArgument('lmbda', default_value='0.05'),
        DeclareLaunchArgument('Kp_pos', default_value='2.0'),
        DeclareLaunchArgument('Kp_rot', default_value='1.5'),
        DeclareLaunchArgument('dt', default_value='0.02'),

        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_desc}]),
        Node(package='tf2_ros', executable='static_transform_publisher', output='screen',
             arguments=['0','0','0','0','0','0','map','base_link']),
        Node(package='numerical_ik', executable='ik_node', output='screen', parameters=[
            {'solver': LaunchConfiguration('solver')},
            {'max_iter': LaunchConfiguration('max_iter')},
            {'tol_pos': LaunchConfiguration('tol_pos')},
            {'tol_rot_deg': LaunchConfiguration('tol_rot_deg')},
            {'alpha': LaunchConfiguration('alpha')},
            {'w_pos': LaunchConfiguration('w_pos')},
            {'w_rot': LaunchConfiguration('w_rot')},
            {'lmbda': LaunchConfiguration('lmbda')},
            {'Kp_pos': LaunchConfiguration('Kp_pos')},
            {'Kp_rot': LaunchConfiguration('Kp_rot')},
            {'dt': LaunchConfiguration('dt')},
        ]),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=(['-d', rviz_cfg] if rviz_cfg else [])),
    ])

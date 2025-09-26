import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = get_package_share_directory('mik_tutorial')
    desc_share = get_package_share_directory('piper_description')
    urdf_file = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'piper.rviz')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument('solver', default_value='jacobian'),
        DeclareLaunchArgument('publish_joint_rate_hz', default_value='30'),
        DeclareLaunchArgument('joint_roles_override', default_value=''),

        # Jacobian
        DeclareLaunchArgument('jacobian_max_iter', default_value='150'),
        DeclareLaunchArgument('jacobian_tol_pos', default_value='0.001'),
        DeclareLaunchArgument('jacobian_tol_rot_deg', default_value='1.0'),
        DeclareLaunchArgument('jacobian_lambda', default_value='0.05'),
        DeclareLaunchArgument('jacobian_alpha', default_value='0.7'),
        DeclareLaunchArgument('jacobian_w_pos', default_value='1.0'),
        DeclareLaunchArgument('jacobian_w_rot', default_value='0.7'),

        # FABRIK
        DeclareLaunchArgument('fabrik_max_iter', default_value='5'),
        DeclareLaunchArgument('fabrik_tol_pos', default_value='0.001'),
        DeclareLaunchArgument('fabrik_tol_rot_deg', default_value='1.0'),
        DeclareLaunchArgument('fabrik_q_gain', default_value='0.9'),
        DeclareLaunchArgument('fabrik_q_reg', default_value='0.02'),
        DeclareLaunchArgument('fabrik_smooth_q', default_value='0.30'),
        DeclareLaunchArgument('fabrik_max_step_deg', default_value='6.0'),
        DeclareLaunchArgument('fabrik_orient_gate_mul', default_value='5.0'),

        # Debug viz
        DeclareLaunchArgument('debug_viz', default_value='true'),
        DeclareLaunchArgument('debug_max_points', default_value='80'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
        ),
        Node(
            package='mik_tutorial',
            executable='ik_node',
            name='ik_test_node',
            output='screen',
            parameters=[
                {'solver': LaunchConfiguration('solver')},
                {'publish_joint_rate_hz': LaunchConfiguration('publish_joint_rate_hz')},
                {'joint_roles_override': LaunchConfiguration('joint_roles_override')},
                # Jacobian
                {'jacobian_max_iter': LaunchConfiguration('jacobian_max_iter')},
                {'jacobian_tol_pos': LaunchConfiguration('jacobian_tol_pos')},
                {'jacobian_tol_rot_deg': LaunchConfiguration('jacobian_tol_rot_deg')},
                {'jacobian_lambda': LaunchConfiguration('jacobian_lambda')},
                {'jacobian_alpha': LaunchConfiguration('jacobian_alpha')},
                {'jacobian_w_pos': LaunchConfiguration('jacobian_w_pos')},
                {'jacobian_w_rot': LaunchConfiguration('jacobian_w_rot')},
                # FABRIK
                {'fabrik_max_iter': LaunchConfiguration('fabrik_max_iter')},
                {'fabrik_tol_pos': LaunchConfiguration('fabrik_tol_pos')},
                {'fabrik_tol_rot_deg': LaunchConfiguration('fabrik_tol_rot_deg')},
                {'fabrik_q_gain': LaunchConfiguration('fabrik_q_gain')},
                {'fabrik_q_reg': LaunchConfiguration('fabrik_q_reg')},
                {'fabrik_smooth_q': LaunchConfiguration('fabrik_smooth_q')},
                {'fabrik_max_step_deg': LaunchConfiguration('fabrik_max_step_deg')},
                {'fabrik_orient_gate_mul': LaunchConfiguration('fabrik_orient_gate_mul')},
                # Debug
                {'debug_viz': LaunchConfiguration('debug_viz')},
                {'debug_max_points': LaunchConfiguration('debug_max_points')},
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
        ),
    ])

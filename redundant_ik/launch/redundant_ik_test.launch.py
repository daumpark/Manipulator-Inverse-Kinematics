# redundant_ik/launch/redundant_ik_test.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share   = get_package_share_directory('redundant_ik')
    desc_share  = get_package_share_directory('piper_description')
    urdf_file   = os.path.join(desc_share, 'urdf', 'piper_no_gripper_description.urdf')
    rviz_local  = os.path.join(pkg_share, 'rviz', 'piper.rviz')
    rviz_fallback = os.path.join(desc_share, 'rviz', 'piper_no_gripper.rviz')
    rviz_cfg = rviz_local if os.path.exists(rviz_local) else (rviz_fallback if os.path.exists(rviz_fallback) else '')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    # 런치 인자
    solver_arg   = DeclareLaunchArgument('solver',   default_value='nullspace')
    dt_arg       = DeclareLaunchArgument('dt',       default_value='0.02')
    lam_arg      = DeclareLaunchArgument('lam',      default_value='0.01')
    k_ns_arg     = DeclareLaunchArgument('k_ns',     default_value='0.2')
    Kp_arg       = DeclareLaunchArgument('Kp',       default_value='1.0')
    gamma_arg    = DeclareLaunchArgument('gamma_max',default_value='0.2')
    nu_arg       = DeclareLaunchArgument('nu',       default_value='10.0')
    s0_arg       = DeclareLaunchArgument('sigma0',   default_value='0.005')

    return LaunchDescription([
        solver_arg, dt_arg, lam_arg, k_ns_arg, Kp_arg, gamma_arg, nu_arg, s0_arg,

        # robot_state_publisher (URDF 직접 주입)
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_desc}]),
        Node(package='tf2_ros', executable='static_transform_publisher', output='screen',
             arguments=['0','0','0','0','0','0','map','base_link']),
        # IK 노드
        Node(package='redundant_ik', executable='ik_node', name='redundant_ik_node', output='screen',
             parameters=[
               {'solver': LaunchConfiguration('solver')},
               {'frame':  'base_link'},
               {'dt':     LaunchConfiguration('dt')},
               {'lam':    LaunchConfiguration('lam')},
               {'k_ns':   LaunchConfiguration('k_ns')},
               {'Kp':     LaunchConfiguration('Kp')},
               {'gamma_max': LaunchConfiguration('gamma_max')},
               {'nu':     LaunchConfiguration('nu')},
               {'sigma0': LaunchConfiguration('sigma0')},
             ]),

        # RViz
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=(['-d', rviz_cfg] if rviz_cfg else [])),
    ])

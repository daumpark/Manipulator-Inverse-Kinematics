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
        # ---- declare CLI arguments ----
        DeclareLaunchArgument(
            'solver',
            default_value='jacobian',
            description="IK solver: 'jacobian' or 'fabrik'"
        ),
        DeclareLaunchArgument(
            'publish_joint_rate_hz',
            default_value='30',
            description='JointState publish throttle (Hz)'
        ),

        # ---- nodes ----
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

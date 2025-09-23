import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    # Package share directory
    pkg_share = get_package_share_directory('analytical_ik_demo')

    # Launch arguments
    l1_arg = DeclareLaunchArgument(
        'L1', default_value='0.25', description='Length of link 1'
    )
    l2_arg = DeclareLaunchArgument(
        'L2', default_value='0.20', description='Length of link 2'
    )

    # URDF file
    urdf_file_path = os.path.join(pkg_share, 'urdf', 'planar_2dof.urdf.xacro')
    robot_description_content = Command([
        'xacro ', urdf_file_path, ' ',
        'L1:=', LaunchConfiguration('L1'), ' ',
        'L2:=', LaunchConfiguration('L2')
    ])
    robot_description = {'robot_description': robot_description_content}

    # RViz config file
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'planar.rviz')

    # Nodes
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    planar_ik_node = Node(
        package='analytical_ik_demo',
        executable='planar_ik_node',
        name='planar_ik_node',
        output='screen',
        parameters=[
            {'L1': LaunchConfiguration('L1')},
            {'L2': LaunchConfiguration('L2')},
            {'rate_hz': 50.0},
            {'toggle_elbow_every_sec': 1.5}
        ]
    )

    # planar_target_publisher_node is removed because we now use an interactive marker.

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        l1_arg,
        l2_arg,
        robot_state_publisher_node,
        planar_ik_node,
        rviz_node
    ])
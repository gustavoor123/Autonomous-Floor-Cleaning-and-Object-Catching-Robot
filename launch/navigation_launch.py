import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('my_bot')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(pkg_share, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        Node(
            package='my_bot',
            executable='nav_wander.py',
            name='nav_wander',
            output='screen',
        ),

        Node(
            package='my_bot',
            executable='cmd_vel_mux.py',
            name='cmd_vel_mux',
            output='screen',
        ),

        Node(
            package='my_bot',
            executable='odom_tf_relay.py',
            name='odom_tf_relay',
            output='screen',
        ),

        Node(
            package='depth_image_proc',
            executable='point_cloud_xyz_node',
            name='oak_pointcloud',
            remappings=[
                ('image_rect', '/oak/stereo/image_raw'),
                ('camera_info', '/oak/stereo/camera_info'),
                ('points', '/oak/stereo/points'),
            ],
            output='screen',
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            remappings=[('/cmd_vel', '/cmd_vel_nav')],
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),

        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            remappings=[('/cmd_vel', '/cmd_vel_nav')],
        ),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'autostart': True},
                        {'node_names': [
                            'controller_server',
                            'planner_server',
                            'behavior_server',
                            'bt_navigator',
                        ]},
                        {'use_sim_time': use_sim_time},
                        {'bond_timeout': 0.0}],
        ),
    ])

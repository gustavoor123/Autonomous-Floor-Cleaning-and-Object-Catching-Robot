import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_path = get_package_share_directory('my_bot')
    params_file = os.path.join(pkg_path, 'config', 'mapper_params_online_async.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    script_path = os.path.join(pkg_path, 'launch', 'slam_params_setter.sh')

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')]),
        launch_arguments={
            'slam_params_file': params_file,
            'use_sim_time': use_sim_time
        }.items()
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),
        # Restamp /scan_reliable (Pi clock) with VM clock so SLAM and Nav2
        # message filters never see timestamps older than the local TF buffer.
        Node(
            package='my_bot',
            executable='scan_restamper.py',
            name='scan_restamper',
            output='screen',
        ),
        slam_toolbox,
    ])

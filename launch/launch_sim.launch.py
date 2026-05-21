import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    package_name = 'my_bot'
    pkg_path = os.path.join(get_package_share_directory(package_name))
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Load the controllers configuration
    config_path = os.path.join(pkg_path, 'config', 'gazebo_controllers.yaml')

    # World file path
    world_file = os.path.join(pkg_path, 'worlds', 'empty.sdf')

    # Process the URDF file once
    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file).toxml()

    # robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config,
            'use_sim_time': use_sim_time
        }]
    )
    
    # robot_description publisher
    node_desc_pub = Node(
        package='my_bot',
        executable='publish_robot_description.py',
        output='screen',
        parameters=[{'robot_description': robot_description_config}]
    )

    # Gazebo launch file
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    # gazebo spawner
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'my_bot', '-z', '0.1'],
        output='screen'
    )
    spawn_entity = TimerAction(period=0.5, actions=[spawn_entity]) 
    joint_state_spawner = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen',
            )
        ]
    )

    # Spawner for the diff_drive_base_controller
    diff_drive_spawner = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['diff_drive_base_controller'],
                output='screen',
            )
        ]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/depth_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true'),

        node_robot_state_publisher,
        node_desc_pub,
        gazebo,
        bridge,
        spawn_entity,
        joint_state_spawner,
        diff_drive_spawner,
    ])

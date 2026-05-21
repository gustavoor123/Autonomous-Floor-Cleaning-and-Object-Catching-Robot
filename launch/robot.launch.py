import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():

    pkg_path = get_package_share_directory('my_bot')
    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file).toxml()

    serial_port = LaunchConfiguration('serial_port')

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config,
            'use_sim_time': False
        }]
    )

    # Arduino code for motors and encoders node
    arduino_bridge = Node(
        package='my_bot',
        executable='arduino_bridge.py',
        name='arduino_bridge',
        output='screen',
        parameters=[{
            'serial_port': serial_port,
            'baud_rate': 57600
        }]
    )

    # RPLidar node
    rplidar = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        name='rplidar',
        output='screen',
        parameters=[{
            'serial_port': '/dev/rplidar',
            'frame_id': 'lidar_frame',
            'serial_baudrate': 115200,
            'angle_compensate': True
        }]
    )

    # Scan relay node
    scan_relay = Node(
        package='my_bot',
        executable='scan_relay.py',
        name='scan_relay',
        output='screen',
    )

    oak_d_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='oak_d_tf',
        arguments=['0.117', '0.0', '0.262', '0', '0', '0',
                   'base_footprint', 'oak-d_frame'],
        output='screen',
    )

    # cmd_vel mux node
    cmd_vel_mux = Node(
        package='my_bot',
        executable='cmd_vel_mux.py',
        name='cmd_vel_mux',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyUSB0',
            description='Serial port for Arduino'),
        robot_state_publisher,
        arduino_bridge,
        rplidar,
        scan_relay,
        oak_d_tf,
        cmd_vel_mux,
    ])

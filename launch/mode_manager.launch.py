import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_mode = LaunchConfiguration('default_mode')
    joy_dev = LaunchConfiguration('joy_dev')

    # robot_mode_manager
    mode_manager = Node(
        package='my_bot',
        executable='robot_mode_manager.py',
        name='robot_mode_manager',
        output='screen',
        parameters=[{
            'default_mode': default_mode,
        }],
    )

    # joy_node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'dev': joy_dev,
            'deadzone': 0.1,
            'autorepeat_rate': 20.0,
        }],
    )

    # joy_mode_switch
    joy_mode_switch = Node(
        package='my_bot',
        executable='joy_mode_switch.py',
        name='joy_mode_switch',
        output='screen',
        parameters=[{
            'mode_button': 4,     # L1 on PS4/PS5
            'teleop_button': 5,   # R1  (deadman for IDLE teleop)
            'linear_axis': 1,     # left stick Y
            'angular_axis': 3,    # right stick X
            'max_linear': 0.5,
            'max_angular': 2.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'default_mode',
            default_value='IDLE',
            description='Initial robot mode (IDLE | NAV | TRASH_CATCH)'),
        DeclareLaunchArgument(
            'joy_dev',
            default_value='/dev/input/js0',
            description='Joystick device path'),
        mode_manager,
        joy_node,
        joy_mode_switch,
    ])

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotDescriptionPublisher(Node):
    def __init__(self):
        super().__init__('robot_description_publisher')
        self.declare_parameter('robot_description', '')
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, 'robot_description', qos)
        # publish once at startup; transient-local QoS ensures late subscribers see it
        self.publish()

    def publish(self):
        desc = self.get_parameter('robot_description').get_parameter_value().string_value
        if desc:
            msg = String()
            msg.data = desc
            self.pub.publish(msg)
            self.get_logger().info('Published robot_description (%d bytes)' % len(desc))


def main(args=None):
    rclpy.init(args=args)
    node = RobotDescriptionPublisher()
    # give subscribers a moment to receive the latched message
    rclpy.spin_once(node, timeout_sec=0.2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

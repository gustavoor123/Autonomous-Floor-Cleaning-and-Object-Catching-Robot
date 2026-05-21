import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Maps mode name to which input topic to forward
MODE_SOURCE = {
    'IDLE':        '/cmd_vel_joy',
    'NAV':         '/cmd_vel_nav',
    'TRASH_CATCH': '/cmd_vel_trash',
}


class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')

        self._mode: str = 'IDLE'

        # Final output
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Mode subscription
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/robot_mode', self._mode_cb, latched_qos)

        # Command sources
        self.create_subscription(Twist, '/cmd_vel_joy',   self._joy_cb,   10)
        self.create_subscription(Twist, '/cmd_vel_nav',   self._nav_cb,   10)
        self.create_subscription(Twist, '/cmd_vel_trash', self._trash_cb, 10)

        self.get_logger().info(
            'cmd_vel_mux ready  —  IDLE→/cmd_vel_joy  '
            'NAV→/cmd_vel_nav  TRASH_CATCH→/cmd_vel_trash'
        )

    # mode callback

    def _mode_cb(self, msg: String):
        new_mode = msg.data.strip().upper()
        if new_mode == self._mode:
            return
        self.get_logger().info(
            f'Mode {self._mode} → {new_mode}: '
            f'now forwarding {MODE_SOURCE.get(new_mode, "???")} to /cmd_vel'
        )
        self._mode = new_mode
        # Safety stop on every transition
        self._cmd_pub.publish(Twist())

    # source callback

    def _joy_cb(self, msg: Twist):
        if self._mode == 'IDLE':
            self._cmd_pub.publish(msg)

    def _nav_cb(self, msg: Twist):
        if self._mode == 'NAV':
            self._cmd_pub.publish(msg)

    def _trash_cb(self, msg: Twist):
        if self._mode == 'TRASH_CATCH':
            self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

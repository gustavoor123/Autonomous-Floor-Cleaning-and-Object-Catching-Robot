import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist


class JoyModeSwitch(Node):
    def __init__(self):
        super().__init__('joy_mode_switch')

        # parameters for the ps5 controller
        self.declare_parameter('mode_button', 4)       # L1
        self.declare_parameter('teleop_button', 5)     # R1 (deadman)
        self.declare_parameter('linear_axis', 1)       # left stick Y
        self.declare_parameter('angular_axis', 3)      # right stick X
        self.declare_parameter('max_linear', 0.5)      # m/s
        self.declare_parameter('max_angular', 2.0)     # rad/s

        self._mode_btn   = self.get_parameter('mode_button').value
        self._teleop_btn = self.get_parameter('teleop_button').value
        self._lin_axis   = self.get_parameter('linear_axis').value
        self._ang_axis   = self.get_parameter('angular_axis').value
        self._max_lin    = self.get_parameter('max_linear').value
        self._max_ang    = self.get_parameter('max_angular').value

        # current state
        self._prev_buttons: list = []
        self._current_mode: str = 'IDLE'

        # publishers and subscribers
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_joy', 10)

        self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/robot_mode', self._mode_cb, latched_qos)

        # this is the mode cycling
        self._set_mode_cli = self.create_client(
            SetBool, '/robot_mode_manager/set_mode')

        self.get_logger().info(
            f'Joy mode switch ready  –  mode_button={self._mode_btn}, '
            f'teleop_button={self._teleop_btn}')

    # callbacks
    def _mode_cb(self, msg: String):
        self._current_mode = msg.data

    def _joy_cb(self, msg: Joy):
        buttons = msg.buttons

        # detection of button pressed
        if self._prev_buttons:
            if (len(buttons) > self._mode_btn and
                    len(self._prev_buttons) > self._mode_btn):
                if buttons[self._mode_btn] == 1 and self._prev_buttons[self._mode_btn] == 0:
                    self._call_set_mode()
        self._prev_buttons = list(buttons)

        # teleop operation
        if self._current_mode != 'IDLE':
            return
        if len(buttons) <= self._teleop_btn or buttons[self._teleop_btn] != 1:
            return  # deadman not held

        twist = Twist()
        if len(msg.axes) > self._lin_axis:
            twist.linear.x = msg.axes[self._lin_axis] * self._max_lin
        if len(msg.axes) > self._ang_axis:
            twist.angular.z = msg.axes[self._ang_axis] * self._max_ang
        self._cmd_pub.publish(twist)

    # the helpers
    def _call_set_mode(self):
        if not self._set_mode_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('set_mode service not available')
            return
        request = SetBool.Request()
        request.data = True  # ignored by the mode manager; it just cycles
        future = self._set_mode_cli.call_async(request)
        future.add_done_callback(self._set_mode_done)

    def _set_mode_done(self, future):
        try:
            result = future.result()
            self.get_logger().info(f'Mode switched → {result.message}')
        except Exception as e:
            self.get_logger().error(f'set_mode call failed: {e}')


# the entrypoint
def main(args=None):
    rclpy.init(args=args)
    node = JoyModeSwitch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

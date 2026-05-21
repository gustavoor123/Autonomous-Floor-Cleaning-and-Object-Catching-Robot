import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from action_msgs.srv import CancelGoal
from action_msgs.msg import GoalInfo
from unique_identifier_msgs.msg import UUID


# list of modes
MODES = ['IDLE', 'NAV', 'TRASH_CATCH']


class RobotModeManager(Node):
    def __init__(self):
        super().__init__('robot_mode_manager')

        # the parameters
        self.declare_parameter('default_mode', 'IDLE')
        self._mode: str = self.get_parameter('default_mode').value
        if self._mode not in MODES:
            self.get_logger().warn(
                f'Unknown default_mode "{self._mode}", falling back to IDLE')
            self._mode = 'IDLE'

        # publishers
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._mode_pub = self.create_publisher(String, '/robot_mode', latched_qos)
        self._cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)

        # services
        self.create_service(SetBool, '~/set_mode', self._set_mode_cb)
        # Nav2 goal cancellation after leaving navigation
        self._nav2_cancel_cli = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        # publish initial mode + keep-alive republish
        self._publish_mode()
        self.create_timer(2.0, self._publish_mode)   # heartbeat so watchdogs don't time out
        self.get_logger().info(f'Mode manager started  ──  mode: {self._mode}')

    # mode helpers

    def _cycle_mode(self) -> str:
        """Advance to the next mode in the MODES list (wraps around)."""
        idx = MODES.index(self._mode)
        self._mode = MODES[(idx + 1) % len(MODES)]
        return self._mode

    def _publish_mode(self):
        msg = String()
        msg.data = self._mode
        self._mode_pub.publish(msg)

    def _safety_stop(self):
        """Send a zero-velocity command so the robot halts between modes."""
        self._cmd_pub.publish(Twist())   # all fields default to 0.0

    def _cancel_nav2(self):
        """Cancel all active Nav2 navigate_to_pose goals."""
        if not self._nav2_cancel_cli.service_is_ready():
            self.get_logger().info('Nav2 not running — skip goal cancel')
            return
        req = CancelGoal.Request()
        req.goal_info = GoalInfo()   # all-zero UUID = cancel ALL goals
        req.goal_info.goal_id = UUID()
        self._nav2_cancel_cli.call_async(req)
        self.get_logger().info('Sent cancel to Nav2 navigate_to_pose')

    def _set_mode_cb(self, request, response):
        """Cycle to the next mode.  request.data is currently ignored."""
        prev = self._mode
        self._cycle_mode()
        self._safety_stop()
        self._publish_mode()
        self.get_logger().info(f'Mode switch: {prev} → {self._mode}')
        response.success = True
        response.message = self._mode
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotModeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

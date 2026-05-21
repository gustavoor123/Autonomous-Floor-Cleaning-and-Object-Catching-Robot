import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray


class TrashApproach(Node):
    """2-D centering controller for an upward-facing camera."""

    def __init__(self):
        super().__init__('trash_approach')

        # parameters
        self.declare_parameter('centre_tolerance', 0.06)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('max_linear', 0.35)
        self.declare_parameter('max_angular', 1.5)
        self.declare_parameter('min_linear', 0.10)   # overcome motor deadband
        self.declare_parameter('min_angular', 0.20)  # overcome motor deadband
        self.declare_parameter('invert_y', False)
        self.declare_parameter('lost_timeout', 0.5)
        self.declare_parameter('cmd_smoothing', 0.0)  # EMA: 0=off, ~0.4 light filter
        self.declare_parameter('target_image_x', 0.50)
        self.declare_parameter('target_image_y', 0.64)

        self._centre_tol = self.get_parameter('centre_tolerance').value
        self._kp_ang     = self.get_parameter('kp_angular').value
        self._kp_lin     = self.get_parameter('kp_linear').value
        self._max_lin    = self.get_parameter('max_linear').value
        self._max_ang    = self.get_parameter('max_angular').value
        self._min_lin    = self.get_parameter('min_linear').value
        self._min_ang    = self.get_parameter('min_angular').value
        self._invert_y   = self.get_parameter('invert_y').value
        self._lost_sec   = self.get_parameter('lost_timeout').value
        self._smooth_a   = float(self.get_parameter('cmd_smoothing').value)
        self._img_cx     = self.get_parameter('target_image_x').value
        self._img_cy     = self.get_parameter('target_image_y').value

        # state
        self._mode: str = 'IDLE'
        self._last_det_time: float = 0.0
        self._aligned = False       # True when centred under the object
        self._last_twist = None     # None = no active command; output_timer skips

        # publishers and subscribers
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_trash', 10)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/robot_mode', self._mode_cb, latched_qos)
        self.create_subscription(
            Detection2DArray, '/trash_detections', self._det_cb, 10)

        # Periodic check: if we haven't seen a detection in a while, stop.
        # Run at 20 Hz so we react within ~50ms of a stale detection.
        self.create_timer(0.05, self._watchdog)
        # Output timer: re-publish last twist at 10 Hz so arduino_bridge
        # never times out (0.5s) between 1fps YOLO inference frames.
        self.create_timer(0.1, self._output_timer)

        self.get_logger().info('Trash approach controller ready  (upward camera)')

    # callbacks
    def _mode_cb(self, msg: String):
        prev = self._mode
        self._mode = msg.data
        if prev != self._mode:
            self._aligned = False
            if self._mode != 'TRASH_CATCH':
                self._cmd_pub.publish(Twist())

    def _det_cb(self, msg: Detection2DArray):
        """React to the latest batch of detections."""
        if self._mode != 'TRASH_CATCH':
            return
        if not msg.detections:
            return

        self._last_det_time = time.monotonic()

        # Pick the highest-confidence detection.
        best = max(msg.detections,
                   key=lambda d: d.results[0].hypothesis.score
                   if d.results else 0.0)

        # Bounding-box centre in normalised image coords
        det_cx = best.bbox.center.position.x
        det_cy = best.bbox.center.position.y

        # Errors: positive error_x means target is to the LEFT of centre,
        #         positive error_y means target is ABOVE centre in image.
        error_x = self._img_cx - det_cx
        error_y = self._img_cy - det_cy

        # checks if aligned
        if abs(error_x) < self._centre_tol and abs(error_y) < self._centre_tol:
            if not self._aligned:
                self.get_logger().info('Aligned under target – ready to catch!')
                self._aligned = True
            self._last_twist = None  # stop publishing; arduino_bridge times out naturally
            # TODO: trigger the catch mechanism here
            return

        self._aligned = False

        # 2-D proportional centering with deadband boost
        twist = Twist()

        # Image X → angular.z  (turn left/right)
        if abs(error_x) > self._centre_tol:
            raw = self._kp_ang * error_x
            # Boost to min_angular if proportional output is below motor deadband
            if abs(raw) < self._min_ang:
                raw = math.copysign(self._min_ang, raw)
            twist.angular.z = self._clamp(raw, -self._max_ang, self._max_ang)

        # Image Y → linear.x  (drive forward/backward)
        if abs(error_y) > self._centre_tol:
            y_sign = -1.0 if self._invert_y else 1.0
            raw = y_sign * self._kp_lin * error_y
            if abs(raw) < self._min_lin:
                raw = math.copysign(self._min_lin, raw)
            twist.linear.x = self._clamp(raw, -self._max_lin, self._max_lin)

        # Light EMA smoothing to suppress jitter without lag (a in [0,1))
        a = self._smooth_a
        if 0.0 < a < 1.0 and self._last_twist is not None:
            twist.linear.x  = a * self._last_twist.linear.x  + (1.0 - a) * twist.linear.x
            twist.angular.z = a * self._last_twist.angular.z + (1.0 - a) * twist.angular.z

        self._last_twist = twist  # output timer publishes at 10 Hz

    def _watchdog(self):
        """Stop the robot if detections have gone stale."""
        if self._mode != 'TRASH_CATCH':
            return
        if self._aligned:
            return
        elapsed = time.monotonic() - self._last_det_time
        if self._last_det_time > 0 and elapsed > self._lost_sec:
            if self._last_twist is not None:
                # Publish several explicit zeros so the robot stops
                # immediately even if a single message is dropped.
                zero = Twist()
                for _ in range(3):
                    self._cmd_pub.publish(zero)
                self._last_twist = None
            self.get_logger().warn(
                f'No detection for {elapsed:.2f}s – stopping',
                throttle_duration_sec=2.0)

    def _output_timer(self):
        """Republish last nonzero twist at 10 Hz so arduino_bridge never starves."""
        if self._mode != 'TRASH_CATCH':
            return
        if self._last_twist is None:
            return  # No active detection — let arduino_bridge timeout naturally
        self._cmd_pub.publish(self._last_twist)

    # helpers

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))

def main(args=None):
    rclpy.init(args=args)
    node = TrashApproach()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

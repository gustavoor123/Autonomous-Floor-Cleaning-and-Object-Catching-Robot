import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan


def _norm_deg(d: float) -> float:
    """Normalise angle to [0, 360)."""
    d = d % 360.0
    if d < 0.0:
        d += 360.0
    return d


class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')

        # filter params
        self.declare_parameter('rear_blank_deg_min', 0.0)
        self.declare_parameter('rear_blank_deg_max', 0.0)
        self._blank_min = float(self.get_parameter('rear_blank_deg_min').value)
        self._blank_max = float(self.get_parameter('rear_blank_deg_max').value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._pub = self.create_publisher(LaserScan, '/scan_reliable', reliable_qos)
        self.create_subscription(LaserScan, '/scan', self._cb, sensor_qos)

        if self._blank_min == self._blank_max:
            self.get_logger().info(
                'scan_relay ready — /scan (BEST_EFFORT) → /scan_reliable (RELIABLE); '
                'rear blank filter: OFF')
        else:
            self.get_logger().info(
                f'scan_relay ready — /scan → /scan_reliable; '
                f'rear blank filter: [{self._blank_min:.1f}°, {self._blank_max:.1f}°] (lidar frame)')

    # helpers
    def _in_blank_window(self, deg: float) -> bool:
        """Return True if angle (in deg, normalised) falls in the blank window."""
        a = _norm_deg(deg)
        lo = _norm_deg(self._blank_min)
        hi = _norm_deg(self._blank_max)
        if lo == hi:
            return False
        if lo < hi:
            return lo <= a <= hi
        # wrap-around window (e.g. 350° → 10°)
        return a >= lo or a <= hi

    # ── callback ──────────────────────────────────────────────────────

    def _cb(self, msg: LaserScan):
        if self._blank_min == self._blank_max:
            self._pub.publish(msg)
            return

        # Mutate a copy of ranges; leave intensities and other fields untouched.
        ranges = list(msg.ranges)
        n = len(ranges)
        ang_min = msg.angle_min
        ang_inc = msg.angle_increment
        for i in range(n):
            ang_deg = math.degrees(ang_min + i * ang_inc)
            if self._in_blank_window(ang_deg):
                ranges[i] = float('inf')
        msg.ranges = ranges
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo


_BE = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)


class DepthRelay(Node):
    def __init__(self):
        super().__init__('depth_relay')
        self.declare_parameter('max_fps', 5.0)
        max_fps = self.get_parameter('max_fps').value
        self._min_dt = 1.0 / max(1.0, max_fps)
        self._last_t = 0.0

        self._img_pub  = self.create_publisher(Image,      '/oak/stereo/image_raw_nav',   _BE)
        self._info_pub = self.create_publisher(CameraInfo, '/oak/stereo/camera_info_nav', _BE)

        # best effort subscriber
        self.create_subscription(Image,      '/oak/stereo/image_raw',   self._img_cb,  _BE)
        self.create_subscription(CameraInfo, '/oak/stereo/camera_info', self._info_cb, _BE)

        self.get_logger().info(
            f'depth_relay ready — throttling to {max_fps:.1f} Hz BEST_EFFORT')

    def _img_cb(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_t < self._min_dt:
            return
        self._last_t = now
        self._img_pub.publish(msg)

    def _info_cb(self, msg: CameraInfo):
        self._info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import serial
import math
import time
import threading


class ArduinoBridge(Node):

    # MOSFET pin definitions
    VACUUM_MOTOR_PIN = 11
    FILTER_MOTOR_PIN = 12

    def __init__(self):
        super().__init__('arduino_bridge')

        # Robot physical parameters
        self.wheel_radius = 0.1016
        self.wheel_separation = 0.28067
        self.ticks_per_rev = 340
        self.meters_per_tick = (2.0 * math.pi * self.wheel_radius) / self.ticks_per_rev

        self.pid_rate = 30

        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 57600)

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value

        try:
            self.serial = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)
            self.serial.reset_input_buffer()
            self.get_logger().info(f'Connected to Arduino on {port} at {baud}')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to connect to Arduino: {e}')
            raise

        self.serial_lock = threading.Lock()

        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_ticks = 0
        self.prev_right_ticks = 0
        self.first_reading = True

        self.last_cmd_vel = None
        self.cmd_vel_timeout = time.time()

        # Mode tracking — default OFF/IDLE at startup
        self._current_mode = 'IDLE'
        self._last_mode_time = time.time()

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # QoS matching robot_mode_manager publisher
        mode_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.mode_sub = self.create_subscription(
            String,
            '/robot_mode',
            self._mode_callback,
            mode_qos
        )

        # Timers
        self.create_timer(1.0 / 30.0, self.update_odometry)
        self.create_timer(2.0, self._watchdog_callback)

        # Reset encoders on startup
        self.send_command('r')
        self.serial.reset_input_buffer()

        # Initializes the mosfet pins as output and default off
        self._init_aux_pins()

        self.get_logger().info('Arduino bridge started')

    def _init_aux_pins(self):
        self.send_command(f'c {self.VACUUM_MOTOR_PIN} 1')
        self.send_command(f'c {self.FILTER_MOTOR_PIN} 1')
        # 'w PIN 0' = digitalWrite(PIN, LOW) — OFF
        self._set_aux_motors(False)
        self.get_logger().info('Aux motor pins initialized as OUTPUT and OFF')

    def _mode_callback(self, msg: String):
        mode = msg.data.strip().upper()
        self._last_mode_time = time.time()

        if mode == self._current_mode:
            return

        self._current_mode = mode
        self.get_logger().info(f'Mode changed to: {mode}')

        if mode == 'NAV':
            self._set_aux_motors(True)
        else:
            # IDLE, TRASH_CATCH, or unknown — always OFF
            self._set_aux_motors(False)

    def _watchdog_callback(self):
        """Turn OFF aux motors if no mode message received for 5 seconds."""
        elapsed = time.time() - self._last_mode_time
        if elapsed > 5.0 and self._current_mode != 'IDLE':
            self.get_logger().warn('Mode timeout — turning OFF aux motors (safety)')
            self._current_mode = 'IDLE'
            self._set_aux_motors(False)

    def _set_aux_motors(self, on: bool):
        """Turn vacuum and filter motors ON or OFF via DIGITAL_WRITE (w = DIGITAL_WRITE)."""
        state = 1 if on else 0
        # 'w PIN 1' = digitalWrite(PIN, HIGH) — ON
        # 'w PIN 0' = digitalWrite(PIN, LOW)  — OFF
        self.send_command(f'w {self.VACUUM_MOTOR_PIN} {state}')
        self.send_command(f'w {self.FILTER_MOTOR_PIN} {state}')
        self.get_logger().info(f'Aux motors {"ON" if on else "OFF"}')

    # ...existing code...
    def send_command(self, cmd):
        """Send a command to the Arduino and return the response."""
        with self.serial_lock:
            try:
                self.serial.reset_input_buffer()
                self.serial.write((cmd + '\r').encode())
                time.sleep(0.01)
                response = self.serial.readline().decode().strip()
                return response
            except serial.SerialException as e:
                self.get_logger().error(f'Serial error: {e}')
                return ''

    def cmd_vel_callback(self, msg):
        """Store the latest cmd_vel to be sent in the update loop."""
        self.last_cmd_vel = msg
        self.cmd_vel_timeout = time.time()

    def send_motor_command(self):
        """Convert stored cmd_vel to wheel speeds and send to Arduino."""
        if self.last_cmd_vel is None or (time.time() - self.cmd_vel_timeout) > 0.5:
            self.send_command('m 0 0')
            return

        linear_x = -self.last_cmd_vel.linear.x  # negated: ROS +x now drives toward LiDAR
        angular_z = self.last_cmd_vel.angular.z

        left_vel = linear_x - (angular_z * self.wheel_separation / 2.0)
        right_vel = linear_x + (angular_z * self.wheel_separation / 2.0)

        ticks_per_meter = 1.0 / self.meters_per_tick
        left_ticks_per_frame = int(left_vel * ticks_per_meter / self.pid_rate)
        right_ticks_per_frame = int(right_vel * ticks_per_meter / self.pid_rate)

        cmd = f'm {right_ticks_per_frame} {left_ticks_per_frame}'
        self.get_logger().info(
            f'CMD_VEL  lin={linear_x:+.3f} ang={angular_z:+.3f} | '
            f'left_vel={left_vel:+.3f} right_vel={right_vel:+.3f} | '
            f'L_tpf={left_ticks_per_frame:+d} R_tpf={right_ticks_per_frame:+d} | '
            f'cmd="{cmd}"'
        )
        self.send_command(cmd)

    def update_odometry(self):
        """Read encoders, compute odometry, publish odom + TF."""
        self.send_motor_command()

        response = self.send_command('e')

        if not response:
            return

        try:
            parts = response.split()
            if len(parts) != 2:
                return
            left_ticks = -int(parts[0])  # swap+negate for 180° orientation flip
            right_ticks = -int(parts[1])  # swap+negate for 180° orientation flip
        except (ValueError, IndexError):
            return

        if self.first_reading:
            self.prev_left_ticks = left_ticks
            self.prev_right_ticks = right_ticks
            self.first_reading = False
            return

        delta_left = left_ticks - self.prev_left_ticks
        delta_right = right_ticks - self.prev_right_ticks
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks

        dist_left = delta_left * self.meters_per_tick
        dist_right = delta_right * self.meters_per_tick

        dist_center = (dist_left + dist_right) / 2.0
        delta_theta = (dist_right - dist_left) / self.wheel_separation

        if abs(delta_left) > 0 or abs(delta_right) > 0:
            self.get_logger().info(
                f'ENC raw parts[0]={parts[0]} parts[1]={parts[1]} | '
                f'left_ticks={left_ticks:+d} right_ticks={right_ticks:+d} | '
                f'dL={delta_left:+d} dR={delta_right:+d} | '
                f'dist_L={dist_left:+.4f} dist_R={dist_right:+.4f} | '
                f'dist_center={dist_center:+.4f} dTheta={delta_theta:+.4f}'
            )

        self.x += dist_center * math.cos(self.theta + delta_theta / 2.0)
        self.y += dist_center * math.sin(self.theta + delta_theta / 2.0)
        self.theta += delta_theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        dt = 1.0 / 30.0
        vx = dist_center / dt
        vtheta = delta_theta / dt

        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        q = self.euler_to_quaternion(0, 0, self.theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.pose.covariance = [
            0.01, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.01, 0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  1e6,  0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  1e6,  0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  1e6,  0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.03,
        ]

        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = vtheta

        odom.twist.covariance = [
            0.01, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.01, 0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  1e6,  0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  1e6,  0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  1e6,  0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.03,
        ]

        self.odom_pub.publish(odom)

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        return [x, y, z, w]

    def destroy_node(self):
        """Stop motors and aux motors on shutdown."""
        self.send_command('m 0 0')
        self._set_aux_motors(False)  # Safety: turn OFF aux motors on shutdown
        self.serial.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

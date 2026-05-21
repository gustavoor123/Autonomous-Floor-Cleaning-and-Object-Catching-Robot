import math
import random
import time as pytime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState
import tf2_ros


class NavWander(Node):
    def __init__(self):
        super().__init__('nav_wander')

        # ── parameters ────────────────────────────────────────────────
        self.declare_parameter('retry_delay', 1.5)
        self.declare_parameter('nav_timeout', 30.0)
        self.declare_parameter('min_clearance', 0.40)   # metres from any obstacle
        self.declare_parameter('min_goal_distance', 0.90)  # min dist from robot to goal
        self.declare_parameter('max_goal_distance', 1.50)  # max dist from robot to goal

        self._retry_sec       = self.get_parameter('retry_delay').value
        self._nav_timeout     = self.get_parameter('nav_timeout').value
        self._min_clearance   = self.get_parameter('min_clearance').value
        self._min_goal_dist   = self.get_parameter('min_goal_distance').value
        self._max_goal_dist   = self.get_parameter('max_goal_distance').value

        # state
        self._mode: str = 'IDLE'
        self._map: OccupancyGrid | None = None
        self._free_cells: list | None = None     # cached [(row, col), ...]
        self._current_goal_handle = None
        self._retry_timer = None
        self._tf_stable_since: float | None = None   # wall time when TF first became valid

        # the action client
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._lifecycle_nodes = [
            '/bt_navigator',
            '/controller_server',
            '/planner_server',
            '/behavior_server',
        ]
        self._lifecycle_clients = {
            n: self.create_client(GetState, f'{n}/get_state')
            for n in self._lifecycle_nodes
        }

        # transform verification
        self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # subscriptions
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/robot_mode', self._mode_cb, latched_qos)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, latched_qos)

        self.get_logger().info(
            'nav_wander ready  —  will explore autonomously when mode=NAV')

    def _mode_cb(self, msg: String):
        new_mode = msg.data.strip().upper()
        self.get_logger().info(f'[nav_wander] robot_mode received: {new_mode}')
        if new_mode == self._mode:
            return
        prev = self._mode
        self._mode = new_mode
        self.get_logger().info(f'[nav_wander] Mode: {prev} → {new_mode}')

        if new_mode == 'NAV':
            self._tf_stable_since = None   # reset stability clock on every NAV entry
            self._schedule_next_goal(delay=2.0)
        else:
            self._cancel_goal()

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg
        self._free_cells = None    # invalidate cache on map update

    def _nav2_services_ready(self) -> bool:
        for node_name, client in self._lifecycle_clients.items():
            if not client.service_is_ready():
                self.get_logger().warn(
                    f'[nav_wander] Nav2 lifecycle: {node_name}/get_state not discovered yet')
                return False
        return True

    # goal for nav
    def _schedule_next_goal(self, delay: float | None = None):
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        secs = delay if delay is not None else self._retry_sec
        if secs <= 0.0:
            self._send_next_goal()
        else:
            self._retry_timer = self.create_timer(secs, self._on_retry_timer)

    def _on_retry_timer(self):
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        self._send_next_goal()

    def _send_next_goal(self):
        if self._mode != 'NAV':
            self.get_logger().info('[nav_wander] not in NAV; skipping')
            return

        self.get_logger().info('[nav_wander] NAV active, preparing goal')

        if self._map is None:
            self.get_logger().warn('[nav_wander] map received: NO — retrying in 2 s')
            self._schedule_next_goal(delay=2.0)
            return
        self.get_logger().info('[nav_wander] map received: yes')

        # Non-blocking lifecycle services check
        if not self._nav2_services_ready():
            self.get_logger().warn('[nav_wander] Nav2 lifecycle services not ready — retrying in 3 s')
            self._schedule_next_goal(delay=3.0)
            return
        self.get_logger().info('[nav_wander] Nav2 lifecycle services: ready')

        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('[nav_wander] /navigate_to_pose action server not ready — retrying in 2 s')
            self._schedule_next_goal(delay=2.0)
            return
        self.get_logger().info('[nav_wander] /navigate_to_pose action server: ready')

        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1))
            cur_x = tf.transform.translation.x
            cur_y = tf.transform.translation.y
            now = pytime.monotonic()
            if self._tf_stable_since is None:
                self._tf_stable_since = now
                self.get_logger().info('[nav_wander] TF appeared — waiting 4 s for buffer to fill...')
                self._schedule_next_goal(delay=4.0)
                return
            if now - self._tf_stable_since < 4.0:
                self._schedule_next_goal(delay=1.0)
                return
            self.get_logger().info(
                f'[nav_wander] TF map→base_footprint: ({cur_x:.2f}, {cur_y:.2f}) stable')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self._tf_stable_since = None   # reset if TF disappears
            self.get_logger().warn(f'[nav_wander] TF not ready: {e} — retrying in 1 s')
            self._schedule_next_goal(delay=1.0)
            return

        pose = self._pick_random_pose(cur_x, cur_y)
        if pose is None:
            # No cell is far enough — map may be too small/crowded right now.
            self.get_logger().warn(
                '[nav_wander] no goal far enough from robot — retrying in 3 s')
            self._free_cells = None   # invalidate cache so next attempt rescans map
            self._schedule_next_goal(delay=3.0)
            return
        self.get_logger().info(
            f'[nav_wander] selected goal: ({pose.pose.position.x:.2f}, '
            f'{pose.pose.position.y:.2f})')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.get_logger().info('[nav_wander] sending goal to /navigate_to_pose')

        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('[nav_wander] goal REJECTED by Nav2 — retrying')
            self._schedule_next_goal()
            return
        self.get_logger().info('[nav_wander] goal ACCEPTED by Nav2')
        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        result = future.result()
        self._current_goal_handle = None

        if self._mode != 'NAV':
            return

        # GoalStatus: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
        status = result.status
        label = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(status, f'status={status}')
        self.get_logger().info(f'[nav_wander] goal result: {label} — picking next')
        self._schedule_next_goal()

    def _cancel_goal(self):
        """Cancel Nav2 goal and any pending retry when leaving NAV mode."""
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
            self.get_logger().info('Nav2 goal cancelled (mode change)')

    # ── goal selection ─────────────────────────────────────────────────

    def _get_free_cells(self) -> list:
        """Return (and cache) free cells that have minimum clearance from obstacles."""
        if self._free_cells is not None:
            return self._free_cells

        m = self._map
        data = np.array(m.data, dtype=np.int8).reshape(
            (m.info.height, m.info.width))

        free_mask   = (data == 0)
        danger_mask = (data != 0)   # occupied (100) or unknown (-1)

        # Inflate danger zone by min_clearance so goals stay well clear of walls
        clearance_cells = max(1, int(self._min_clearance / m.info.resolution))
        r = clearance_cells
        yi, xi = np.ogrid[-r:r + 1, -r:r + 1]
        struct = (xi * xi + yi * yi <= r * r)
        try:
            from scipy.ndimage import binary_dilation
            danger_inflated = binary_dilation(danger_mask, structure=struct)
        except ImportError:
            danger_inflated = danger_mask   # fallback: no clearance filtering

        safe_free = free_mask & ~danger_inflated
        rows, cols = np.where(safe_free)
        self._free_cells = list(zip(rows.tolist(), cols.tolist()))
        self.get_logger().info(f'Goal pool: {len(self._free_cells)} clear cells')
        return self._free_cells

    def _pick_random_pose(self, cur_x: float, cur_y: float) -> PoseStamped | None:
        free = self._get_free_cells()
        if not free:
            return None

        m = self._map
        res = m.info.resolution
        ox  = m.info.origin.position.x
        oy  = m.info.origin.position.y

        candidates = list(free)
        random.shuffle(candidates)

        rejected = 0
        for row, col in candidates:
            wx = ox + (col + 0.5) * res
            wy = oy + (row + 0.5) * res
            dist = math.hypot(wx - cur_x, wy - cur_y)
            if dist < self._min_goal_dist or dist > self._max_goal_dist:
                rejected += 1
                continue
            if rejected:
                self.get_logger().debug(
                    f'[nav_wander] skipped {rejected} goals too close (<{self._min_goal_dist:.2f} m)')
            self.get_logger().info(
                f'[nav_wander] selected goal: ({wx:.2f}, {wy:.2f})  dist={dist:.2f} m')
            # Face the direction of travel
            yaw = math.atan2(wy - cur_y, wx - cur_x)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp.sec = 0
            pose.header.stamp.nanosec = 0
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            return pose

        self.get_logger().warn(
            f'[nav_wander] all {len(candidates)} cells outside [{self._min_goal_dist:.2f}, {self._max_goal_dist:.2f}] m — no valid goal')
        return None


def main(args=None):
    rclpy.init(args=args)
    node = NavWander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

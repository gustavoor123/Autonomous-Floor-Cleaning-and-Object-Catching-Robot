import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

# YOLO and OpenCV imports
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    _HAS_YOLO = True
except ImportError as e:
    _HAS_YOLO = False
    _IMPORT_ERR = str(e)

# cv_bridge converts ROS Image and OpenCV numpy array
try:
    from cv_bridge import CvBridge
    _HAS_BRIDGE = True
except ImportError:
    _HAS_BRIDGE = False


class TrashDetector(Node):
    """Publish YOLOv8 detections gated on TRASH_CATCH mode."""

    def __init__(self):
        super().__init__('trash_detector')

        # parameters 
        self.declare_parameter('target_class_id', 39)     
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('model_name', 'yolov8n.pt')
        self.declare_parameter('image_size', 320)
        self.declare_parameter('queue_depth', 1)
        self.declare_parameter('frame_skip', 1)
        self.declare_parameter('detection_holdover', 0.5)  # seconds to repeat last detection
        self.declare_parameter('yolo_reconfirm_frames', 30)  # re-run YOLO every N tracked frames
        self.declare_parameter('yolo_miss_tolerance', 4)  # # consecutive YOLO misses tolerated
        self.declare_parameter('use_tracker', True)  # False = YOLO-only fallback (rollback)
        # ── detection_mode: 'motion' (new fast path), 'hybrid', 'yolo_only' ──
        self.declare_parameter('detection_mode', 'motion')
        # ── Motion detection params (background subtraction) ────────────
        self.declare_parameter('motion_min_area_frac', 0.001)  # min blob area as frac of img
        self.declare_parameter('motion_max_area_frac', 0.5)    # max blob area as frac of img
        self.declare_parameter('motion_history', 200)          # MOG2 history length
        self.declare_parameter('motion_var_threshold', 32.0)   # MOG2 sensitivity
        self.declare_parameter('motion_score', 0.30)           # synthetic score for motion-only detections
        # ── Stricter motion filters (single-frame, no delay) ────────────
        self.declare_parameter('motion_min_aspect', 0.25)      # min h/w (reject very flat blobs)
        self.declare_parameter('motion_max_aspect', 4.0)       # max h/w (reject very tall sliver)
        self.declare_parameter('motion_min_score', 0.20)       # min synthetic score to publish
        self.declare_parameter('motion_global_reject_frac', 0.25)  # if mask coverage > this, frame is camera motion
        self.declare_parameter('motion_edge_margin_frac', 0.05)    # reject blobs within X% of any edge
        self.declare_parameter('motion_central_bias', True)        # prefer central blobs over edge blobs
        # Hand/limb rejection: blobs that are BOTH large AND wide (aspect<1)
        # are almost certainly a hand/arm close to the camera, not a bottle.
        self.declare_parameter('motion_huge_wide_area_frac', 0.15) # area frac threshold for the huge-wide reject
        self.declare_parameter('motion_huge_wide_aspect', 1.0)     # aspect (h/w) below which big blobs are rejected
        self.declare_parameter('motion_tall_bonus', True)          # score boost for tall (aspect>1) blobs
        # Fast-object boosters (single frame, no delay):
        # frame differencing catches fast thrown objects that MOG2 misses
        # because they don't persist long enough to be learned as foreground.
        self.declare_parameter('motion_use_frame_diff', True)      # OR a frame-diff mask with MOG2
        self.declare_parameter('motion_diff_threshold', 18)        # uint8 threshold for |gray_t - gray_t-1|
        self.declare_parameter('motion_morph_kernel', 3)           # morph kernel (odd; smaller preserves small fast blobs)
        self.declare_parameter('motion_downscale', 2)              # input downscale factor (2 or 3 — Pi 5 CPU)
        # ── Tracker validation thresholds (anti-drift) ──────────────
        self.declare_parameter('tracker_max_jump_frac', 0.20)  # max bbox-center jump per frame, frac of img
        self.declare_parameter('tracker_area_min_ratio', 0.4)  # min area vs init area
        self.declare_parameter('tracker_area_max_ratio', 2.5)  # max area vs init area
        self.declare_parameter('tracker_edge_margin', 4)       # px from edge to count as 'partial'
        self.declare_parameter('tracker_yolo_grace_sec', 1.5)  # max time without YOLO confirm

        self._target_id   = self.get_parameter('target_class_id').value
        self._conf_thresh = self.get_parameter('confidence_threshold').value
        self._debug_img   = self.get_parameter('publish_debug_image').value
        model_name        = self.get_parameter('model_name').value
        self._imgsz       = self.get_parameter('image_size').value
        queue_depth       = self.get_parameter('queue_depth').value
        self._frame_skip  = self.get_parameter('frame_skip').value
        self._frame_count = 0
        self._holdover_sec  = self.get_parameter('detection_holdover').value
        self._yolo_reconfirm = self.get_parameter('yolo_reconfirm_frames').value
        self._yolo_miss_tol  = self.get_parameter('yolo_miss_tolerance').value
        self._use_tracker    = self.get_parameter('use_tracker').value
        self._det_mode       = self.get_parameter('detection_mode').value
        self._mot_min_area   = float(self.get_parameter('motion_min_area_frac').value)
        self._mot_max_area   = float(self.get_parameter('motion_max_area_frac').value)
        self._mot_history    = int(self.get_parameter('motion_history').value)
        self._mot_var_thresh = float(self.get_parameter('motion_var_threshold').value)
        self._mot_score      = float(self.get_parameter('motion_score').value)
        self._mot_min_aspect = float(self.get_parameter('motion_min_aspect').value)
        self._mot_max_aspect = float(self.get_parameter('motion_max_aspect').value)
        self._mot_min_score  = float(self.get_parameter('motion_min_score').value)
        self._mot_global_reject = float(self.get_parameter('motion_global_reject_frac').value)
        self._mot_edge_margin   = float(self.get_parameter('motion_edge_margin_frac').value)
        self._mot_central_bias  = bool(self.get_parameter('motion_central_bias').value)
        self._mot_huge_wide_area   = float(self.get_parameter('motion_huge_wide_area_frac').value)
        self._mot_huge_wide_aspect = float(self.get_parameter('motion_huge_wide_aspect').value)
        self._mot_tall_bonus       = bool(self.get_parameter('motion_tall_bonus').value)
        self._mot_use_diff      = bool(self.get_parameter('motion_use_frame_diff').value)
        self._mot_diff_thresh   = int(self.get_parameter('motion_diff_threshold').value)
        self._mot_kernel_size   = int(self.get_parameter('motion_morph_kernel').value)
        if self._mot_kernel_size < 1:
            self._mot_kernel_size = 1
        if self._mot_kernel_size % 2 == 0:
            self._mot_kernel_size += 1
        self._mot_downscale     = max(1, int(self.get_parameter('motion_downscale').value))
        self._mot_prev_gray     = None  # previous downscaled gray frame for diff
        self._trk_max_jump   = self.get_parameter('tracker_max_jump_frac').value
        self._trk_area_min   = self.get_parameter('tracker_area_min_ratio').value
        self._trk_area_max   = self.get_parameter('tracker_area_max_ratio').value
        self._trk_edge_marg  = self.get_parameter('tracker_edge_margin').value
        self._trk_yolo_grace = self.get_parameter('tracker_yolo_grace_sec').value
        self._last_det_array = None   # last Detection2DArray with detections
        self._tracker = None              # OpenCV tracker instance or None
        self._frames_since_yolo = 0       # frames since last YOLO confirmation
        self._yolo_miss_count = 0         # consecutive YOLO heartbeat misses
        self._trk_init_area = 0.0         # bbox area at tracker init (for size validation)
        self._trk_last_cx = 0.0           # last bbox center (for jump validation)
        self._trk_last_cy = 0.0
        self._trk_last_yolo_confirm = 0.0 # time.monotonic() of last YOLO that re-init'd tracker
        self._last_det_time  = 0.0    # time.monotonic() of last real detection

        # mode gate
        self._mode: str = 'IDLE'
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/robot_mode', self._mode_cb, latched_qos)

        # detection publisher
        self._det_pub = self.create_publisher(
            Detection2DArray, '/trash_detections', 10)

        # debug publisher
        if self._debug_img:
            self._debug_pub = self.create_publisher(
                Image, '/trash_detector/debug_image', 5)
            self._debug_compressed_pub = self.create_publisher(
                CompressedImage, '/trash_detector/debug_image/compressed', 5)
        else:
            self._debug_pub = None
            self._debug_compressed_pub = None

        # dependency checks
        if not _HAS_YOLO:
            self.get_logger().error(
                f'ultralytics / OpenCV not found ({_IMPORT_ERR}).  '
                'Activate the yolo_venv or check the shebang line.')
            return

        if not _HAS_BRIDGE:
            self.get_logger().error(
                'cv_bridge not found.  Install:  '
                'sudo apt install ros-jazzy-cv-bridge')
            return

        self.get_logger().info(f'Loading YOLO model: {model_name}  '
                               f'(imgsz={self._imgsz})')
        self._model = YOLO(model_name)

        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model(dummy, imgsz=self._imgsz, verbose=False)
        self.get_logger().info('YOLO model loaded and warmed up')

        # motion detection so background subtractor
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=self._mot_history,
            varThreshold=self._mot_var_thresh,
            detectShadows=False,
        )
        # Morphology kernel for cleaning the motion mask
        self._mot_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._mot_kernel_size, self._mot_kernel_size))

        # async YOLO worker
        self._yolo_lock = threading.Lock()
        self._yolo_request = None     # (frame, request_id) waiting for YOLO
        self._yolo_result  = None     # (bbox_xyxy_or_None, conf, frame_used, request_id)
        self._yolo_busy = False
        self._yolo_req_id = 0
        self._yolo_stop = False
        self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self._yolo_thread.start()
        self._bridge = CvBridge()
        self.create_subscription(
            Image,
            '/upward_camera/image_raw',
            self._image_cb,
            queue_depth,
        )

        self.create_timer(0.2, self._holdover_timer)

        self.get_logger().info(
            'Trash detector ready – subscribing to /upward_camera/image_raw  '
            '(waiting for TRASH_CATCH mode)')

    # callbacks
    def _mode_cb(self, msg: String):
        if self._mode != msg.data:
            self.get_logger().info(f'Mode changed → {msg.data}')
            # Reset tracker state on every mode transition
            self._tracker = None
            self._frames_since_yolo = 0
            self._yolo_miss_count = 0
            self._trk_init_area = 0.0
            self._trk_last_cx = 0.0
            self._trk_last_cy = 0.0
            self._trk_last_yolo_confirm = 0.0
            # Reset background model so a fresh mode entry doesn't see stale "motion"
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=self._mot_history,
                varThreshold=self._mot_var_thresh,
                detectShadows=False,
            )
            self._mot_prev_gray = None
            with self._yolo_lock:
                self._yolo_request = None
                self._yolo_result = None
        self._mode = msg.data

    def _holdover_timer(self):
        """Republish last detection at 5 Hz during holdover window."""
        if self._mode != 'TRASH_CATCH':
            return
        if self._last_det_array is None:
            return
        if (time.monotonic() - self._last_det_time) < self._holdover_sec:
            self._det_pub.publish(self._last_det_array)

    def _make_tracker(self):
        """Create a fresh KCF tracker (fast, ~5ms/frame on Pi 5)."""
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            return cv2.legacy.TrackerKCF_create()

    def _run_yolo(self, cv_image):
        """Run YOLO and return (best_bbox_xyxy, confidence) or (None, 0.0)."""
        results = self._model(
            cv_image,
            imgsz=self._imgsz,
            conf=self._conf_thresh,
            verbose=False,
        )
        best_box = None
        best_conf = 0.0
        n_total = 0
        for r in results:
            for box in r.boxes:
                n_total += 1
                if int(box.cls[0]) != self._target_id:
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        if best_box is not None:
            img_h, img_w = cv_image.shape[:2]
            cx_n = ((best_box[0] + best_box[2]) / 2.0) / img_w
            cy_n = ((best_box[1] + best_box[3]) / 2.0) / img_h
            self.get_logger().info(
                f'YOLO hit: cx={cx_n:.2f} cy={cy_n:.2f} conf={best_conf:.2f} '
                f'(total_boxes={n_total})',
                throttle_duration_sec=0.5)
        return best_box, best_conf

    def _publish_detection(self, x1, y1, x2, y2, img_w, img_h, stamp, score):
        """Publish a single Detection2DArray from a bbox in full-image coords."""
        det_array = Detection2DArray()
        det_array.header = Header()
        det_array.header.stamp = stamp
        det_array.header.frame_id = 'camera_link_optical'
        d2d = Detection2D()
        d2d.bbox.center.position.x = ((x1 + x2) / 2.0) / img_w
        d2d.bbox.center.position.y = ((y1 + y2) / 2.0) / img_h
        d2d.bbox.size_x = (x2 - x1) / img_w
        d2d.bbox.size_y = (y2 - y1) / img_h
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(self._target_id)
        hyp.hypothesis.score = score
        d2d.results.append(hyp)
        det_array.detections.append(d2d)
        self._det_pub.publish(det_array)
        self._last_det_array = det_array
        self._last_det_time  = time.monotonic()

    def _image_cb(self, msg: Image):
        """Dispatch to motion-fast / hybrid / yolo-only path."""
        # FPS logger (every ~3 s)
        now_t = time.monotonic()
        if not hasattr(self, '_fps_window_start'):
            self._fps_window_start = now_t
            self._fps_frame_count = 0
        self._fps_frame_count += 1
        if now_t - self._fps_window_start >= 3.0:
            fps = self._fps_frame_count / (now_t - self._fps_window_start)
            self.get_logger().info(
                f'[fps] image_cb={fps:.1f} Hz  mode={self._mode}  det_mode={self._det_mode}')
            self._fps_window_start = now_t
            self._fps_frame_count = 0
        if self._det_mode == 'motion':
            self._image_cb_motion(msg)
        elif self._use_tracker:
            self._image_cb_hybrid(msg)
        else:
            self._image_cb_yolo_only(msg)

    def _yolo_worker(self):
        """Background thread: pulls frames from _yolo_request, runs YOLO,
        deposits result in _yolo_result. Never blocks _image_cb."""
        while not self._yolo_stop:
            with self._yolo_lock:
                req = self._yolo_request
                self._yolo_request = None
            if req is None:
                time.sleep(0.02)
                continue
            frame, req_id = req
            try:
                bbox, conf = self._run_yolo(frame)
            except Exception as e:
                self.get_logger().warn(f'YOLO worker error: {e}')
                bbox, conf = None, 0.0
            with self._yolo_lock:
                self._yolo_result = (bbox, conf, frame, req_id)

    def _submit_yolo(self, frame):
        """Hand a frame to the YOLO worker, replacing any pending request."""
        self._yolo_req_id += 1
        with self._yolo_lock:
            # Always drop older request — YOLO works on the freshest frame
            self._yolo_request = (frame, self._yolo_req_id)

    def _take_yolo_result(self):
        """Return latest (bbox, conf, frame) or None. Clears the slot."""
        with self._yolo_lock:
            r = self._yolo_result
            self._yolo_result = None
        return r

    # motion detection for object
    def _detect_motion(self, cv_image):
        ds = self._mot_downscale
        if ds > 1:
            small = cv2.resize(cv_image, (cv_image.shape[1] // ds,
                                          cv_image.shape[0] // ds))
        else:
            small = cv_image
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # MOG2 mask
        mog = self._bg_sub.apply(small)
        # Hard-threshold (drop any residual shadow/uncertain pixels)
        _, mog = cv2.threshold(mog, 200, 255, cv2.THRESH_BINARY)

        # Frame-diff mask (catches fast thrown objects)
        if self._mot_use_diff and self._mot_prev_gray is not None \
                and self._mot_prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._mot_prev_gray)
            _, diff = cv2.threshold(diff, self._mot_diff_thresh, 255,
                                    cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mog, diff)
        else:
            mask = mog
        self._mot_prev_gray = gray

        # Cleanup: tiny open removes specks, dilate joins fragments.
        # Keep kernel small so small fast blobs aren't erased.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._mot_kernel)
        mask = cv2.dilate(mask, self._mot_kernel, iterations=1)

        h_s, w_s = mask.shape[:2]
        img_area = float(w_s * h_s)

        # Global motion rejection: if too much of the frame is moving,
        # this is camera shake / global lighting change — ignore.
        coverage = float(cv2.countNonZero(mask)) / img_area
        if coverage > self._mot_global_reject:
            self.get_logger().info(
                f'Motion: global motion {coverage:.2f} > '
                f'{self._mot_global_reject} — ignoring frame',
                throttle_duration_sec=1.0)
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Edge margin in mask (downscaled) coords
        em_x = int(w_s * self._mot_edge_margin)
        em_y = int(h_s * self._mot_edge_margin)
        cx_img = w_s / 2.0
        cy_img = h_s / 2.0

        best = None
        best_metric = -1.0
        for c in contours:
            a = cv2.contourArea(c)
            if a < self._mot_min_area * img_area:
                continue
            if a > self._mot_max_area * img_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Edge rejection
            if (x < em_x or y < em_y
                    or (x + w) > w_s - em_x
                    or (y + h) > h_s - em_y):
                continue
            if w <= 0 or h <= 0:
                continue
            metric = a
            if self._mot_central_bias:
                bx = x + w / 2.0
                by = y + h / 2.0
                # distance from center, normalized to half-diagonal
                d = ((bx - cx_img) ** 2 + (by - cy_img) ** 2) ** 0.5
                d_max = (cx_img ** 2 + cy_img ** 2) ** 0.5
                # Multiplier in [0.5, 1.0]: central blobs preferred but
                # off-center valid blobs still picked over none.
                metric = metric * (1.0 - 0.5 * (d / d_max))
            if metric > best_metric:
                best_metric = metric
                best = (x, y, w, h)

        if best is None:
            return None
        x, y, w, h = best
        # Scale bbox back up to full image coords
        return (x * ds, y * ds, (x + w) * ds, (y + h) * ds)

    # motion-driven image callback
    def _image_cb_motion(self, msg: Image):
        """Fast pre-detection: motion drives /trash_detections every frame.
        YOLO runs async in a background thread for class confirmation +
        tracker re-init. Tracker, when locked, takes priority over motion."""
        in_trash_mode = (self._mode == 'TRASH_CATCH')
        if not in_trash_mode and self._debug_pub is None:
            return
        if not _HAS_YOLO or not _HAS_BRIDGE:
            return

        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        img_h, img_w = cv_image.shape[:2]
        bbox = None
        score = 0.0
        source = None  # 'TRK' | 'YOLO' | 'MOT'

        # Tracker is highest priority after yolo
        if self._tracker is not None:
            ok, tbox = self._tracker.update(cv_image)
            drop_reason = None
            if not ok:
                drop_reason = 'KCF.update returned False'
            else:
                tx, ty, tw, th = tbox
                tx2, ty2 = tx + tw, ty + th
                cx = (tx + tx2) / 2.0
                cy = (ty + ty2) / 2.0
                m = self._trk_edge_marg
                if (tx < m or ty < m
                        or tx2 > img_w - m or ty2 > img_h - m):
                    drop_reason = 'bbox left frame edge'
                elif tw < 8 or th < 8:
                    drop_reason = 'bbox degenerate'
                elif tw > img_w * 0.9 or th > img_h * 0.9:
                    drop_reason = 'bbox huge'
                else:
                    bbox = (tx, ty, tx2, ty2)
                    score = 0.9
                    source = 'TRK'
                    self._trk_last_cx = cx
                    self._trk_last_cy = cy
            if drop_reason is not None:
                self.get_logger().info(f'Tracker dropped: {drop_reason}',
                                       throttle_duration_sec=0.5)
                self._tracker = None

        # Motion path runs only when tracker is not locked.
        motion_bbox = None
        if bbox is None:
            motion_bbox = self._detect_motion(cv_image)
        else:
            try:
                ds = self._mot_downscale
                if ds > 1:
                    small = cv2.resize(
                        cv_image,
                        (cv_image.shape[1] // ds, cv_image.shape[0] // ds))
                else:
                    small = cv_image
                self._bg_sub.apply(small)
                self._mot_prev_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass

        # If no tracker is present, use motion bbox immediately for reaction
        if bbox is None and motion_bbox is not None:
            if self._mot_score >= self._mot_min_score:
                bbox = motion_bbox
                score = self._mot_score
                source = 'MOT'
            else:
                self.get_logger().info(
                    f'Motion bbox rejected: score {self._mot_score:.2f} '
                    f'< min {self._mot_min_score:.2f}',
                    throttle_duration_sec=2.0)

        # current frame sent to yolo worker
        self._submit_yolo(cv_image)

        # If a YOLO result is ready, use it to (re)acquire / confirm
        yres = self._take_yolo_result()
        if yres is not None:
            ybbox, yconf, yframe, _rid = yres
            if ybbox is not None:
                # YOLO confirmed bottle — (re)init tracker on YOLO's frame
                yx1, yy1, yx2, yy2 = ybbox
                self._tracker = self._make_tracker()
                try:
                    self._tracker.init(
                        yframe,
                        (int(yx1), int(yy1),
                         int(yx2 - yx1), int(yy2 - yy1)))
                    self._trk_init_area = max(1.0,
                                              (yx2 - yx1) * (yy2 - yy1))
                    self._trk_last_cx = (yx1 + yx2) / 2.0
                    self._trk_last_cy = (yy1 + yy2) / 2.0
                    self._trk_last_yolo_confirm = time.monotonic()
                    # Override bbox with YOLO's authoritative box this frame
                    bbox = ybbox
                    score = yconf
                    source = 'YOLO'
                except Exception as e:
                    self.get_logger().warn(f'Tracker init failed: {e}')
                    self._tracker = None

        # Detection is published
        if bbox is not None and in_trash_mode:
            x1, y1, x2, y2 = bbox
            self._publish_detection(x1, y1, x2, y2, img_w, img_h,
                                    msg.header.stamp, score)

        # Debug overlay
        if self._debug_pub is not None:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                color = {
                    'TRK':  (0, 255,   0),  # green
                    'YOLO': (0, 255, 255),  # yellow
                    'MOT':  (255, 0, 255),  # magenta
                }.get(source, (200, 200, 200))
                cv2.rectangle(cv_image, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                label = f'{source} {score:.2f}'
                cv2.putText(cv_image, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            try:
                small = cv2.resize(cv_image, (cv_image.shape[1] // 2,
                                              cv_image.shape[0] // 2))
                debug_msg = self._bridge.cv2_to_imgmsg(small, encoding='bgr8')
                debug_msg.header = msg.header
                self._debug_pub.publish(debug_msg)
                ok, buf = cv2.imencode('.jpg', small,
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    comp = CompressedImage()
                    comp.header = msg.header
                    comp.format = 'jpeg'
                    comp.data = buf.tobytes()
                    self._debug_compressed_pub.publish(comp)
            except Exception as e:
                self.get_logger().warn(f'Debug image publish failed: {e}')

    def _image_cb_yolo_only(self, msg: Image):
        """ROLLBACK PATH: original YOLO-every-frame behavior."""
        in_trash_mode = (self._mode == 'TRASH_CATCH')
        if not in_trash_mode and self._debug_pub is None:
            return
        if not _HAS_YOLO or not _HAS_BRIDGE:
            return
        self._frame_count += 1
        if self._frame_count % self._frame_skip != 0:
            return
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return
        img_h, img_w = cv_image.shape[:2]
        bbox, conf = self._run_yolo(cv_image)
        if bbox is not None and in_trash_mode:
            x1, y1, x2, y2 = bbox
            self._publish_detection(x1, y1, x2, y2, img_w, img_h,
                                    msg.header.stamp, conf)
        if self._debug_pub is not None:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(cv_image, f'YOLO {conf:.2f}', (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            try:
                small = cv2.resize(cv_image, (cv_image.shape[1] // 2, cv_image.shape[0] // 2))
                debug_msg = self._bridge.cv2_to_imgmsg(small, encoding='bgr8')
                debug_msg.header = msg.header
                self._debug_pub.publish(debug_msg)
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    comp = CompressedImage()
                    comp.header = msg.header
                    comp.format = 'jpeg'
                    comp.data = buf.tobytes()
                    self._debug_compressed_pub.publish(comp)
            except Exception as e:
                self.get_logger().warn(f'Debug image publish failed: {e}')

    def _image_cb_hybrid(self, msg: Image):
        """Hybrid YOLO + KCF tracker: fast tracking with periodic YOLO confirmation."""
        in_trash_mode = (self._mode == 'TRASH_CATCH')
        if not in_trash_mode and self._debug_pub is None:
            return
        if not _HAS_YOLO or not _HAS_BRIDGE:
            return

        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        img_h, img_w = cv_image.shape[:2]
        bbox = None       # (x1, y1, x2, y2) for publishing/drawing
        score = 0.0
        used_yolo = False

        # Tracker is used first
        if self._tracker is not None:
            ok, tbox = self._tracker.update(cv_image)
            drop_reason = None
            if not ok:
                drop_reason = 'KCF.update returned False'
            else:
                tx, ty, tw, th = tbox
                tx2, ty2 = tx + tw, ty + th
                cx = (tx + tx2) / 2.0
                cy = (ty + ty2) / 2.0
                area = max(1.0, tw * th)
                m = self._trk_edge_marg
                # Validation 1: bbox fully inside frame (with edge margin)
                if (tx < m or ty < m
                        or tx2 > img_w - m or ty2 > img_h - m):
                    drop_reason = 'bbox touching/leaving frame edge'
                # Validation 2: bbox not degenerate / not too large
                elif tw < 8 or th < 8:
                    drop_reason = 'bbox degenerate (too small)'
                elif tw > img_w * 0.9 or th > img_h * 0.9:
                    drop_reason = 'bbox huge (>90% of frame)'
                # Validation 3: area sanity vs init area
                elif self._trk_init_area > 0:
                    ratio = area / self._trk_init_area
                    if ratio < self._trk_area_min or ratio > self._trk_area_max:
                        drop_reason = f'area ratio {ratio:.2f} out of [{self._trk_area_min},{self._trk_area_max}]'
                # Validation 4: no impossible jump between frames
                if drop_reason is None and (self._trk_last_cx or self._trk_last_cy):
                    jump = max(abs(cx - self._trk_last_cx) / img_w,
                               abs(cy - self._trk_last_cy) / img_h)
                    if jump > self._trk_max_jump:
                        drop_reason = f'bbox jumped {jump:.2f} (>{self._trk_max_jump})'
                # Validation 5: YOLO must confirm within grace window
                if drop_reason is None and self._trk_last_yolo_confirm > 0:
                    since_conf = time.monotonic() - self._trk_last_yolo_confirm
                    if since_conf > self._trk_yolo_grace:
                        drop_reason = f'no YOLO confirm for {since_conf:.1f}s'

                if drop_reason is None:
                    bbox = (tx, ty, tx2, ty2)
                    score = 0.9  # synthetic — tracker is locked
                    self._frames_since_yolo += 1
                    self._trk_last_cx = cx
                    self._trk_last_cy = cy

            if drop_reason is not None:
                self.get_logger().info(f'Tracker dropped: {drop_reason}',
                                       throttle_duration_sec=0.5)
                self._tracker = None
                self._trk_init_area = 0.0
                self._trk_last_cx = 0.0
                self._trk_last_cy = 0.0

        need_yolo = (self._tracker is None
                     or self._frames_since_yolo >= self._yolo_reconfirm)
        if need_yolo:
            yolo_bbox, yolo_conf = self._run_yolo(cv_image)
            if yolo_bbox is not None:
                x1, y1, x2, y2 = yolo_bbox
                self._tracker = self._make_tracker()
                self._tracker.init(cv_image,
                                   (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                self._frames_since_yolo = 0
                self._trk_init_area = max(1.0, (x2 - x1) * (y2 - y1))
                self._trk_last_cx = (x1 + x2) / 2.0
                self._trk_last_cy = (y1 + y2) / 2.0
                self._trk_last_yolo_confirm = time.monotonic()
                bbox = (x1, y1, x2, y2)
                score = yolo_conf
                used_yolo = True
            elif self._tracker is not None:
                # YOLO missed but tracker is still locked → KEEP TRACKING.
                # Try YOLO again on the next frame instead of waiting a full
                # reconfirm window, so we refresh as soon as it reappears.
                self._frames_since_yolo = max(self._frames_since_yolo - 1,
                                              self._yolo_reconfirm - 1)

        # Publish detection if there is
        if bbox is not None and in_trash_mode:
            x1, y1, x2, y2 = bbox
            self._publish_detection(x1, y1, x2, y2, img_w, img_h,
                                    msg.header.stamp, score)

        # Debug image
        if self._debug_pub is not None:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 255) if used_yolo else (0, 255, 0)  # yellow=YOLO, green=tracker
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f'{"YOLO" if used_yolo else "TRK"} {score:.2f}'
                cv2.putText(cv_image, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            try:
                small = cv2.resize(cv_image, (cv_image.shape[1] // 2, cv_image.shape[0] // 2))
                debug_msg = self._bridge.cv2_to_imgmsg(small, encoding='bgr8')
                debug_msg.header = msg.header
                self._debug_pub.publish(debug_msg)
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    comp = CompressedImage()
                    comp.header = msg.header
                    comp.format = 'jpeg'
                    comp.data = buf.tobytes()
                    self._debug_compressed_pub.publish(comp)
            except Exception as e:
                self.get_logger().warn(f'Debug image publish failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TrashDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

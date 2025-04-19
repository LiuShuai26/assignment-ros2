#!/usr/bin/env python3

import os

# os.environ["RCUTILS_LOGGING_FORMAT"] = "[{severity}] [{name}]: {message}"
# os.environ["RCUTILS_LOGGING_USE_STDOUT"] = "1"

import sys
import select
import termios
import threading
import time
import math
from collections import deque, defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class ArucoNavigator(Node):
    def __init__(self):
        super().__init__("aruco_navigator")
        self.get_logger().info("Aruco navigator started.")

        # QoS for image subscriber
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers & subscribers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.image_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, qos_profile=qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 50
        )
        self.bridge = CvBridge()

        # ArUco detection params
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.marker_length = 0.16  # meters
        self.camera_matrix = np.array(
            [
                [456.82000732, 0.0, 326.66424561],
                [0.0, 456.82000732, 243.38911438],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # Odometry state
        self.odom_x = self.odom_y = self.odom_yaw = 0.0

        # Map-frame state (after initialization)
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.offset_x = self.offset_y = self.offset_yaw = 0.0
        self.initialized = False

        # Marker estimates & smoothing buffers
        self.marker_positions = {}  # mid -> (x,y)
        self._marker_buffers = defaultdict(lambda: deque(maxlen=5))

        self.alpha = 0.2  # EMA weight for updates

        # Center point & navigation state
        self.target_center = None
        self.state = None

        self.obstacle_too_close = False
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT: attempt to deliver samples, but may lose them if the network is not robust.
            durability=DurabilityPolicy.VOLATILE,  # VOLATILE: no attempt is made to persist samples.
            history=HistoryPolicy.KEEP_LAST,  # KEEP_LAST: only store up to N samples, configurable via the queue depth option.
            depth=10,  # a queue size of 10 to buffer messages if they arrive faster than they can be processed
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "scan",
            self.scan_callback,
            qos_profile=qos_profile,  # Replace with your lidar topic
        )

        # Control loop timer
        self.create_timer(0.1, self.control_loop)

        # Keyboard setup
        self._setup_keyboard()

    def scan_callback(self, scan: LaserScan):
        # we only care about +/-10° around straight ahead (i.e. index center)
        # LaserScan.angle_min, angle_increment → compute index range
        mid = len(scan.ranges) // 2
        window = int(math.radians(10) / scan.angle_increment)
        front_ranges = scan.ranges[mid - window : mid + window + 1]
        # filter out invalid readings
        front = [r for r in front_ranges if not math.isinf(r)]

        if front and min(front) < 0.5:
            self.obstacle_too_close = True
        else:
            self.obstacle_too_close = False

    def _setup_keyboard(self):
        try:
            self._tty = open("/dev/tty")
        except OSError:
            self.get_logger().warn("Could not open /dev/tty for keyboard input")
            return

        fd = self._tty.fileno()
        self._orig_termios = termios.tcgetattr(fd)
        new_t = termios.tcgetattr(fd)
        new_t[3] &= ~(termios.ECHO | termios.ICANON)
        new_t[3] |= termios.ISIG
        new_t[6][termios.VMIN] = 0
        new_t[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSADRAIN, new_t)

        def keyboard_loop():
            self.get_logger().info(
                "Keyboard: W/A/S/D to move, G to go to center, Q to quit"
            )
            try:
                while rclpy.ok():
                    r, _, _ = select.select([self._tty], [], [], 0.1)
                    k = self._tty.read(1) if r else ""
                    if k == "\x03" or k == "q":
                        self.get_logger().info("Shutting down.")
                        rclpy.shutdown()
                        break

                    cmd = Twist()
                    if k == "w":
                        cmd.linear.x = 0.5
                    elif k == "s":
                        cmd.linear.x = -0.5
                    elif k == "a":
                        cmd.angular.z = 0.5
                    elif k == "d":
                        cmd.angular.z = -0.5
                    elif k == "i":
                        self.state = "explore"
                    elif k == "g":
                        threading.Thread(target=self.gotoposition).start()

                    # if cmd.linear.x > 0 and self.obstacle_too_close:
                    #     cmd.linear.x = 0.0
                    #     self.get_logger().warn("Stopping: obstacle < 0.5m ahead")

                    self.cmd_pub.publish(cmd)
                    time.sleep(0.1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._orig_termios)
                try:
                    self._tty.close()
                except:
                    pass

        threading.Thread(target=keyboard_loop).start()

    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        _, _, yaw = r.as_euler("xyz", degrees=False)
        self.odom_x, self.odom_y, self.odom_yaw = px, py, yaw

        if self.initialized:
            self.robot_x = self.odom_x + self.offset_x
            self.robot_y = self.odom_y + self.offset_y
            self.robot_yaw = self.odom_yaw + self.offset_yaw

    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            if ids is None:
                return

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            for i, mid in enumerate(ids.flatten()):
                x_cam, _, z_cam = tvecs[i][0]
                forward, left = z_cam, -x_cam

                # Initial alignment on marker 1
                if mid == 1:
                    # compute camera yaw relative to marker
                    R_cm, _ = cv2.Rodrigues(rvecs[i][0])  # marker → camera
                    R_mc = R_cm.T  # camera → marker

                    # 1) get the ZYX angles (camera→marker)
                    e_zyx = R.from_matrix(R_mc).as_euler("zyx", degrees=False)

                    # 2) extract the pitch (second element)
                    yaw_cam = e_zyx[1]

                    marker_world_yaw = -math.pi / 2
                    measured_yaw = marker_world_yaw + yaw_cam

                    dx = (
                        math.cos(self.robot_yaw) * forward
                        - math.sin(self.robot_yaw) * left
                    )
                    dy = (
                        math.sin(self.robot_yaw) * forward
                        + math.cos(self.robot_yaw) * left
                    )

                    self.get_logger().info(f"Marker {mid}: dx={dy:.2f}, dy={dx:.2f}")

                    meas_x, meas_y = -dx, -dy

                    measured_yaw_deg = math.degrees(measured_yaw)
                    # logging forword, left, yaw and dx and dy
                    self.get_logger().info(
                        f"Marker {mid}: forward={forward:.2f}, left={left:.2f}"
                    )

                    if not self.initialized:
                        # first detection: set offsets directly
                        self.robot_x, self.robot_y, self.robot_yaw = (
                            meas_x,
                            meas_y,
                            measured_yaw,
                        )
                        self.offset_x = self.robot_x - self.odom_x
                        self.offset_y = self.robot_y - self.odom_y
                        self.offset_yaw = self.robot_yaw - self.odom_yaw
                        self.initialized = True
                        self.get_logger().info(
                            f"Init from marker1 Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={measured_yaw_deg:.2f}"
                        )
                    else:
                        # refine offsets via EMA
                        new_off_x = meas_x - self.odom_x
                        new_off_y = meas_y - self.odom_y
                        new_off_yaw = measured_yaw - self.odom_yaw
                        self.offset_x = (
                            1 - self.alpha
                        ) * self.offset_x + self.alpha * new_off_x
                        self.offset_y = (
                            1 - self.alpha
                        ) * self.offset_y + self.alpha * new_off_y
                        self.offset_yaw = (
                            1 - self.alpha
                        ) * self.offset_yaw + self.alpha * new_off_yaw
                        # update global pose
                        self.robot_x = self.odom_x + self.offset_x
                        self.robot_y = self.odom_y + self.offset_y
                        self.robot_yaw = self.odom_yaw + self.offset_yaw
                        self.get_logger().info(
                            f"Refined from marker1 Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={measured_yaw_deg:.2f}"
                        )
                    # always store marker1 at origin
                    self.marker_positions[1] = (0.0, 0.0)
                    continue

                # For markers 2–4, after initialization
                if self.initialized and mid in (2, 3, 4):

                    # rotate into world frame by current robot_yaw
                    dx = (
                        math.cos(self.robot_yaw) * forward
                        - math.sin(self.robot_yaw) * left
                    )
                    dy = (
                        math.sin(self.robot_yaw) * forward
                        + math.cos(self.robot_yaw) * left
                    )
                    # logging dx and dy
                    self.get_logger().info(f"Marker {mid}: dx={dx:.2f}, dy={dy:.2f}")

                    # absolute marker position in world
                    meas_mx = self.robot_x + dx
                    meas_my = self.robot_y + dy

                    # smooth via EMA if we’ve seen this marker before
                    if mid in self.marker_positions:
                        old_x, old_y = self.marker_positions[mid]
                        mx = (1 - self.alpha) * old_x + self.alpha * meas_mx
                        my = (1 - self.alpha) * old_y + self.alpha * meas_my
                    else:
                        mx, my = meas_mx, meas_my

                    self.marker_positions[mid] = (mx, my)
                    self.get_logger().info(f"Marker {mid}: x={mx:.2f}, y={my:.2f}")

            if self.initialized:
                self.get_logger().info(
                    f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, degree={math.degrees(self.robot_yaw):.2f}"
                )

            # compute center when all 4 markers seen
            if (
                self.initialized
                and len(self.marker_positions) >= 4
                and self.target_center is None
            ):
                xs = [p[0] for p in self.marker_positions.values()]
                ys = [p[1] for p in self.marker_positions.values()]
                self.target_center = (sum(xs) / 4.0, sum(ys) / 4.0)
                self.state = "explored"
                self.get_logger().info(
                    f"Quad center: x={self.target_center[0]:.2f}, y={self.target_center[1]:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def control_loop(self):
        twist = Twist()

        if self.state == "explore":

            twist.angular.z = -0.2
            self.cmd_pub.publish(twist)

    def gotoposition(self):
        if not self.initialized or self.target_center is None:
            self.get_logger().warn("Not ready: waiting for initial pose & all markers.")
            return

        self.get_logger().info("Navigating to center…")
        rate = self.create_rate(10)
        while rclpy.ok():
            dx = self.target_center[0] - self.robot_x
            dy = self.target_center[1] - self.robot_y
            dist = math.hypot(dx, dy)
            if dist < 0.2:
                self.get_logger().info("Arrived at center.")
                self.cmd_pub.publish(Twist())
                break

            theta = math.atan2(dy, dx)
            err = (theta - self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
            cmd = Twist()
            if abs(err) > 0.4:
                # turn faster the larger the error, but keep a minimum base turn
                turn_speed = 0.1 + 1.5 * abs(err)
                # choose direction based on sign of err
                cmd.angular.z = turn_speed if err > 0 else -turn_speed
            else:
                # only drive forward when roughly facing the target
                cmd.linear.x = 0.1 + 0.5 * dist

            cmd.linear.x = max(min(cmd.linear.x, 0.5), -0.5)
            cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)

            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f"center: x={self.target_center[0]:.2f}, y={self.target_center[1]:.2f}"
            )
            self.get_logger().info(
                f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, degree={math.degrees(self.robot_yaw):.2f}"
            )
            self.get_logger().info(f"To center: dist={dist:.2f}, angle_err={err:.2f}")
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # nothing else needed — keyboard thread restores terminal
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

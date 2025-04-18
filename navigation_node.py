#!/usr/bin/env python3

import os
os.environ['RCUTILS_LOGGING_FORMAT']  = '[{severity}] [{name}]: {message}'
os.environ['RCUTILS_LOGGING_USE_STDOUT'] = '1'

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
        self.aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.marker_length = 0.16  # meters
        self.camera_matrix = np.array([
            [456.82000732,   0.0,            326.66424561],
            [0.0,            456.82000732,   243.38911438],
            [0.0,            0.0,            1.0]
        ], dtype=np.float32)
        self.dist_coeffs   = np.zeros((5,1), dtype=np.float32)

        # Odometry state
        self.odom_x = self.odom_y = self.odom_yaw = 0.0

        # Map-frame state (after initialization)
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.offset_x = self.offset_y = self.offset_yaw = 0.0
        self.initialized = False

        # Marker estimates & smoothing buffers
        self.marker_positions = {}  # mid -> (x,y)
        self._marker_buffers   = defaultdict(lambda: deque(maxlen=5))

        # Center point & navigation state
        self.target_center = None
        self.state = None

        # Control loop timer
        self.create_timer(0.1, self.control_loop)

        # Keyboard setup
        self._setup_keyboard()

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
            self.get_logger().info("Keyboard: W/A/S/D to move, G to go to center, Q to quit")
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
        q  = msg.pose.pose.orientation
        r  = R.from_quat([q.x, q.y, q.z, q.w])
        _, _, yaw = r.as_euler("xyz", degrees=False)
        self.odom_x, self.odom_y, self.odom_yaw = px, py, yaw

        if self.initialized:
            self.robot_x   = self.odom_x + self.offset_x
            self.robot_y   = self.odom_y + self.offset_y
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

                # Initialization on marker 1
                if mid == 1 and not self.initialized:
                    marker_yaw = math.pi/2
                    R_ct, _ = cv2.Rodrigues(rvecs[i][0])
                    R_tc = R_ct.T
                    yaw_cam = R.from_matrix(R_tc).as_euler("xyz")[2]
                    self.robot_yaw = yaw_cam - marker_yaw

                    dx = math.cos(self.robot_yaw)*forward - math.sin(self.robot_yaw)*left
                    dy = math.sin(self.robot_yaw)*forward + math.cos(self.robot_yaw)*left
                    self.robot_x, self.robot_y = -dx, -dy

                    self.offset_x   = self.robot_x - self.odom_x
                    self.offset_y   = self.robot_y - self.odom_y
                    self.offset_yaw = self.robot_yaw - self.odom_yaw
                    self.initialized = True
                    self.get_logger().info(
                        f"Init from marker1: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}"
                    )
                    continue

                # After init: compute raw world coords
                if self.initialized:
                    dx = math.cos(self.robot_yaw)*forward - math.sin(self.robot_yaw)*left
                    dy = math.sin(self.robot_yaw)*forward + math.cos(self.robot_yaw)*left
                    mx_raw, my_raw = self.robot_x + dx, self.robot_y + dy

                    # add to smoothing buffer
                    buf = self._marker_buffers[mid]
                    buf.append((mx_raw, my_raw))
                    xs, ys = zip(*buf)
                    # median filter
                    mx = sorted(xs)[len(xs)//2]
                    my = sorted(ys)[len(ys)//2]
                    self.marker_positions[mid] = (mx, my)
                    self.get_logger().info(f"Marker {mid}: x={mx:.2f}, y={my:.2f}")

            if self.initialized:
                self.get_logger().info(
                    f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}"
                )

            # compute center when all 4 markers seen
            if self.initialized and len(self.marker_positions) >= 4 and self.target_center is None:
                xs = [p[0] for p in self.marker_positions.values()]
                ys = [p[1] for p in self.marker_positions.values()]
                self.target_center = (sum(xs)/4.0, sum(ys)/4.0)
                self.state = "explored"
                self.get_logger().info(
                    f"Quad center: x={self.target_center[0]:.2f}, y={self.target_center[1]:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def control_loop(self):
        if self.state == "explore":
            twist = Twist()
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
            if dist < 0.05:
                self.get_logger().info("Arrived at center.")
                self.cmd_pub.publish(Twist())
                break

            theta = math.atan2(dy, dx)
            err = (theta - self.robot_yaw + math.pi) % (2*math.pi) - math.pi
            cmd = Twist()
            if abs(err) > 0.1:
                cmd.angular.z = 0.2 + 0.5*err
            else:
                cmd.linear.x = 0.1 + 0.3*dist

            cmd.linear.x  = max(min(cmd.linear.x,  0.5), -0.5)
            cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)

            self.cmd_pub.publish(cmd)
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

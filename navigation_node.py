#!/usr/bin/env python3

import sys
import select
import termios
import tty
import threading
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np

# import tf2_ros
# import tf_transformations
# from tf_transformations import euler_from_quaternion, euler_from_matrix
from scipy.spatial.transform import Rotation as R


class ArucoNavigator(Node):
    def __init__(self):
        super().__init__("aruco_navigator")
        self.get_logger().info("Aruco navigator started.")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT: attempt to deliver samples, but may lose them if the network is not robust.
            durability=DurabilityPolicy.VOLATILE,  # VOLATILE: no attempt is made to persist samples.
            history=HistoryPolicy.KEEP_LAST,  # KEEP_LAST: only store up to N samples, configurable via the queue depth option.
            depth=10,  # a queue size of 10 to buffer messages if they arrive faster than they can be processed
        )

        # Publishers & subscribers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.image_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile,
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 50
        )
        self.bridge = CvBridge()

        # TF2 listener (unused now, but kept for potential future)
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ArUco detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.marker_length = 0.16  # meters

        # Camera intrinsics (replace with your calibration)
        self.camera_matrix = np.array(
            [
                [456.82000732, 0.0, 326.66424561],
                [0.0, 456.82000732, 243.38911438],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # Odometry pose (in odom frame)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        # Map‐frame pose (initialized on marker1 detection)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        self.initialized = False

        # Storage for landmark positions (map frame)
        self.marker_positions = {}  # mid -> (x, y)
        self.target_center = None  # (x, y)

        self.state = None

        # Control loop at 10 Hz
        self.create_timer(0.1, self.control_loop)

        # Optional keyboard control
        self._setup_keyboard()

    def _setup_keyboard(self):
        try:
            inp = sys.stdin if sys.stdin.isatty() else open("/dev/tty")
        except:
            return

        def get_key():
            settings = termios.tcgetattr(inp)
            tty.setraw(inp.fileno())
            r, _, _ = select.select([inp], [], [], 0.1)
            k = inp.read(1) if r else ""
            termios.tcsetattr(inp, termios.TCSADRAIN, settings)
            return k

        def loop():
            self.get_logger().info(
                "Keyboard: W/A/S/D to move, G to go to center, Q to quit"
            )
            while rclpy.ok():
                k = get_key()
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
                    threading.Thread(target=self.gotoposition, daemon=True).start()
                elif k == "q":
                    self.get_logger().info("Shutting down.")
                    rclpy.shutdown()
                    break
                self.cmd_pub.publish(cmd)
                time.sleep(0.1)

        threading.Thread(target=loop, daemon=True).start()

    def odom_callback(self, msg: Odometry):
        # 1) read current odometry
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # note: scipy wants [x, y, z, w]
        r = R.from_quat([q.x, q.y, q.z, q.w])
        # get roll, pitch, yaw in radians (order ‘xyz’ corresponds to roll, pitch, yaw)
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)

        self.odom_x = px
        self.odom_y = py
        self.odom_yaw = yaw

        # 2) if we have an initial landmark fix, update map‐frame pose
        if self.initialized:
            self.robot_x = self.odom_x + self.offset_x
            self.robot_y = self.odom_y + self.offset_y
            self.robot_yaw = self.odom_yaw + self.offset_yaw

    def image_callback(self, msg: Image):
        try:
            # detect markers
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            if ids is None:
                return

            # estimate poses in camera frame
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            # tf_map_base = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            # t = tf_map_base.transform.translation
            # q = tf_map_base.transform.rotation
            # _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # process each marker
            for i, mid in enumerate(ids.flatten()):
                # camera‐relative coordinates
                x_cam, _, z_cam = tvecs[i][0]
                forward = z_cam
                left = -x_cam

                if mid == 1 and not self.initialized:
                    # 1) compute initial map‐frame position from this detection
                    #    we assume marker1 lies at (0,0) facing +Y (world yaw=π/2)
                    marker_world_yaw = math.pi / 2

                    # 2) extract yaw of camera relative to marker
                    # 1) get the 3×3 camera‑to‑marker rotation
                    R_ct, _ = cv2.Rodrigues(rvecs[i][0])
                    # 2) invert it to get marker‑to‑camera
                    R_tc = R_ct.T
                    # 3) build a Rotation and extract Euler angles
                    rot = R.from_matrix(R_tc)
                    roll, pitch, yaw_cam = rot.as_euler("xyz", degrees=False)

                    # 3) estimated robot yaw in world
                    self.robot_yaw = -marker_world_yaw + yaw_cam

                    # 4) rotate the camera‐relative vector into world frame
                    dx = (
                        math.cos(self.robot_yaw) * forward
                        - math.sin(self.robot_yaw) * left
                    )
                    dy = (
                        math.sin(self.robot_yaw) * forward
                        + math.cos(self.robot_yaw) * left
                    )

                    # robot is opposite direction from marker at (0,0)
                    self.robot_x = -dx
                    self.robot_y = -dy

                    # 5) compute offsets to align odom → map
                    self.offset_x = self.robot_x - self.odom_x
                    self.offset_y = self.robot_y - self.odom_y
                    self.offset_yaw = self.robot_yaw - self.odom_yaw

                    self.initialized = True
                    self.get_logger().info(
                        f"Init from marker1: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}"
                    )
                    continue

                # once initialized, or for markers 2–4, compute world position
                if self.initialized:
                    dx = (
                        math.cos(self.robot_yaw) * forward
                        - math.sin(self.robot_yaw) * left
                    )
                    dy = (
                        math.sin(self.robot_yaw) * forward
                        + math.cos(self.robot_yaw) * left
                    )
                    mx = self.robot_x + dx
                    my = self.robot_y + dy
                    self.marker_positions[int(mid)] = (mx, my)
                    self.get_logger().info(f"Marker {mid}: x={mx:.2f}, y={my:.2f}")

            # if self.initialized:
            #     self.get_logger().info(f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}, GT={yaw:.2f}")
            if self.initialized:
                self.get_logger().info(
                    f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}"
                )

            # compute center when we have all four
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
        # spinning scan
        if self.state == "explore":
            twist = Twist()
            twist.angular.z = -0.2
            self.cmd_pub.publish(twist)
            return

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

            theta_target = math.atan2(dy, dx)
            angle_error = (theta_target - self.robot_yaw + math.pi) % (
                2 * math.pi
            ) - math.pi

            cmd = Twist()
            if abs(angle_error) > 0.1:
                cmd.angular.z = 0.2 + 0.5 * angle_error
            else:
                cmd.linear.x = 0.1 + 0.3 * dist

            cmd.linear.x = max(min(cmd.linear.x, 0.5), -0.5)
            cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f"To center: dist={dist:.2f}, angle_err={angle_error:.2f}"
            )
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

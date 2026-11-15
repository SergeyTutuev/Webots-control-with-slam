#!/usr/bin/env python3
"""Webots ↔ ROS2 bridge with direct /control/ speed and steering support."""

import rclpy
from rclpy.node import Node
from controller import Robot
import sys
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros import StaticTransformBroadcaster
from rclpy.time import Time

class WebotsBridge(Node):
    def __init__(self):
        super().__init__('webots_bridge')
        self.get_logger().info("Connecting to Webots...")

        # --- WEBOTS ---
        try:
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())
            self.get_logger().info("Connected to Webots")
        except Exception as e:
            self.get_logger().error(f"Error connecting to Webots: {e}")
            sys.exit(1)

        # --- LIDAR ---
        try:
            self.lidar = self.robot.getDevice('lidar')
            self.lidar.enable(self.timestep)
            self.get_logger().info("Lidar enabled")
        except Exception:
            self.get_logger().error("Could not find lidar")
            sys.exit(1)

        # --- IMU ---
        try:
            self.imu = self.robot.getDevice('imu')
            self.imu.enable(self.timestep)
            self.get_logger().info("IMU enabled")
        except Exception:
            self.get_logger().error("Could not find IMU")
            sys.exit(1)

        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.initial_yaw = self.imu.getRollPitchYaw()[2]

        # --- MOTORS ---
        self.motors = {}
        motor_names = ['front left wheel', 'back left wheel',
                       'front right wheel', 'back right wheel']
        for name in motor_names:
            try:
                motor = self.robot.getDevice(name)
                self.motors[name] = motor
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
                self.get_logger().info(f"Motor configured: {name}")
            except Exception as e:
                self.get_logger().error(f"Error configuring motor {name}: {e}")
                sys.exit(1)

        self.max_motor_velocity = 6.4
        self.wheel_radius = 0.1110          # Pioneer 3-AT
        self.wheel_separation = 0.32        # колея, м

        # --- CAMERAS (MultiSense S21) ---
        try:
            self.left_camera = self.robot.getDevice('car_camera left camera')
            self.right_camera = self.robot.getDevice('car_camera right camera')
            if self.left_camera is None:
                raise RuntimeError("Left camera not found")
            self.left_camera.enable(self.timestep)
            self.get_logger().info("Left camera enabled")
            if self.right_camera is not None:
                self.right_camera.enable(self.timestep)
                self.get_logger().info("Right camera enabled")
        except Exception as e:
            self.get_logger().error(f"Stereo camera init failed: {e}")
            sys.exit(1)

        self.bridge = CvBridge()
        self.car_camera_pub = self.create_publisher(Image, '/car_camera/image', 10)

        # --- CLOCK ---
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        # --- CONTROL SUBSCRIBERS ---
        self.speed_sub = self.create_subscription(
            Float32, '/control/speed', self.speed_callback, 10)
        self.steering_sub = self.create_subscription(
            Float32, '/control/steering', self.steering_callback, 10)

        self.current_speed = 0.0
        self.current_steering = 0.0

        # --- ODOMETRY ---
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        static_t = TransformStamped()
        static_t.header.stamp = Time(seconds=0, nanoseconds=0).to_msg()
        static_t.header.frame_id = 'base_link'
        static_t.child_frame_id = 'lidar_frame'
        static_t.transform.translation.x = 0.0
        static_t.transform.translation.y = 0.0
        static_t.transform.translation.z = 0.28 
        static_t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(static_t)

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.prev_time = self.robot.getTime()

        self.timer = self.create_timer(self.timestep / 1000.0, self.publish_data)
        self.get_logger().info("Webots Bridge ready (Direct Control Mode)")

    # ===================== CONTROL =====================
    def speed_callback(self, msg):
        self.current_speed = float(msg.data)
        self._apply_direct_control()

    def steering_callback(self, msg):
        self.current_steering = max(-0.5, min(float(msg.data), 0.5))
        self._apply_direct_control()

    def _apply_direct_control(self):
        """Прямое управление через /control/speed и /control/steering."""
        speed_scale = 0.08
        base_speed = self.current_speed * speed_scale
        turn_scale = 3.0
        turn_speed = self.current_steering * turn_scale

        INPLACE_STEER_THRESHOLD = 0.5
        if abs(self.current_steering) >= INPLACE_STEER_THRESHOLD:
            left_speed  = -turn_speed
            right_speed =  turn_speed
        else:
            left_speed  = base_speed - turn_speed
            right_speed = base_speed + turn_speed

        left_speed  = max(-self.max_motor_velocity, min(left_speed,  self.max_motor_velocity))
        right_speed = max(-self.max_motor_velocity, min(right_speed, self.max_motor_velocity))

        self.motors['front left wheel'].setVelocity(left_speed)
        self.motors['back left wheel'].setVelocity(left_speed)
        self.motors['front right wheel'].setVelocity(right_speed)
        self.motors['back right wheel'].setVelocity(right_speed)

    # ===================== PUBLISH =====================
    def publish_data(self):
        if self.robot.step(self.timestep) == -1:
            return

        try:
            # --- CAMERA ---
            car_image_data = self.left_camera.getImage()
            if car_image_data:
                car_image = np.frombuffer(car_image_data, np.uint8).reshape(
                    (self.left_camera.getHeight(),
                     self.left_camera.getWidth(), 4))
                car_image_rgb = cv2.cvtColor(car_image, cv2.COLOR_BGRA2BGR)
                car_msg = self.bridge.cv2_to_imgmsg(car_image_rgb, "bgr8")
                self.car_camera_pub.publish(car_msg)

            # --- CLOCK & ODOMETRY ---
            sim_t = float(self.robot.getTime())
            sec = int(sim_t)
            nsec = int((sim_t - sec) * 1e9)

            clk = Clock()
            clk.clock.sec = sec
            clk.clock.nanosec = nsec
            self.clock_pub.publish(clk)

            rpy = self.imu.getRollPitchYaw()
            raw_yaw = rpy[2] - self.initial_yaw
            self.odom_yaw = (raw_yaw + math.pi) % (2 * math.pi) - math.pi

            vl = (self.motors['front left wheel'].getVelocity() +
                  self.motors['back left wheel'].getVelocity()) / 2.0
            vr = (self.motors['front right wheel'].getVelocity() +
                  self.motors['back right wheel'].getVelocity()) / 2.0
            linear_speed = ((vl + vr) / 2.0) * self.wheel_radius

            current_time = self.robot.getTime()
            dt = current_time - self.prev_time
            if dt > 0.0:
                self.odom_x += linear_speed * math.cos(self.odom_yaw) * dt
                self.odom_y += linear_speed * math.sin(self.odom_yaw) * dt
                self.prev_time = current_time

            odom = Odometry()
            odom.header.stamp.sec = sec
            odom.header.stamp.nanosec = nsec
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = self.odom_x
            odom.pose.pose.position.y = self.odom_y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.x = 0.0
            odom.pose.pose.orientation.y = 0.0
            odom.pose.pose.orientation.z = math.sin(self.odom_yaw / 2.0)
            odom.pose.pose.orientation.w = math.cos(self.odom_yaw / 2.0)
            odom.twist.twist.linear.x = float(linear_speed)
            odom.twist.twist.angular.z = 0.0
            self.odom_pub.publish(odom)

            # --- TF ---
            t = TransformStamped()
            t.header.stamp = odom.header.stamp
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = self.odom_x
            t.transform.translation.y = self.odom_y
            t.transform.translation.z = 0.0
            t.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)

            # --- LIDAR ---
            if self.lidar:
                scan_data = self.lidar.getRangeImage()
                h_res = self.lidar.getHorizontalResolution()
                n_layers = self.lidar.getNumberOfLayers()

                if n_layers > 1:
                    layer = 0
                    scan_data = scan_data[layer * h_res : (layer + 1) * h_res]

                scan_data_filtered = [r if r > 0.2 else float('inf') for r in scan_data]

                scan_msg = LaserScan()
                scan_msg.header.stamp.sec = sec
                scan_msg.header.stamp.nanosec = nsec
                scan_msg.header.frame_id = 'lidar_frame'
                scan_msg.angle_min = self.lidar.getFov() / 2
                scan_msg.angle_max = -self.lidar.getFov() / 2
                scan_msg.angle_increment = -self.lidar.getFov() / h_res
                scan_msg.range_min = self.lidar.getMinRange()
                scan_msg.range_max = self.lidar.getMaxRange()
                scan_msg.ranges = scan_data_filtered
                self.scan_pub.publish(scan_msg)

        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WebotsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
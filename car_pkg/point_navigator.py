#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math

class BugNavigator(Node):
    def __init__(self):
        super().__init__('point_navigator')
        
        # --- ФИЗИЧЕСКИЕ ПАРАМЕТРЫ РОБОТА ---
        self.robot_radius = 0.3
        self.safety_margin = 0.1
        self.safe_dist = self.robot_radius + self.safety_margin
        
        # --- НАСТРОЙКИ НАВИГАЦИИ ---
        self.target_x = -1.5
        self.target_y = 5.0
        self.goal_tolerance = 0.5
        self.obstacle_threshold = self.safe_dist + 0.5  # 1.2 м
        self.wall_distance = self.safe_dist + 0.1       # 0.8 м
        
        # --- НАСТРОЙКИ ДВИЖЕНИЯ НАЗАД ---
        self.max_reverse_steps = 30       # ~1.5 с при 20 Гц
        self.min_reverse_steps = 5        # ~0.25 с (минимум, чтобы реально отъехать)
        self.reverse_cooldown_steps = 0   # защита от осцилляций
        self.reverse_speed = -20.0        # отрицательная скорость = назад
        
        # --- СОСТОЯНИЕ ---
        self.state = 'GO_TO_GOAL'
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Дистанции с лидара
        self.min_front_dist = float('inf')
        self.min_right_dist = float('inf')
        self.min_left_dist = float('inf')
        self.min_rear_dist = float('inf')
        self.has_rear_data = False 

        self.reverse_steps = 0  # счётчик шагов в режиме назад

        # --- ROS ИНТЕРФЕЙСЫ ---
        self.speed_pub = self.create_publisher(Float32, '/control/speed', 10)
        self.steer_pub = self.create_publisher(Float32, '/control/steering', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"Bug Navigator started. Target: ({self.target_x}, {self.target_y})")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.min_front_dist = float('inf')
        self.min_right_dist = float('inf')
        self.min_left_dist = float('inf')
        self.min_rear_dist = float('inf')
        self.has_rear_data = False

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or r < msg.range_min:
                continue
            
            angle = msg.angle_min + i * msg.angle_increment
            # Нормализуем угол в диапазон [-pi, pi]
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            
            # Фронтальный сектор (прямо перед роботом)
            if -0.5 < angle < 0.5:
                self.min_front_dist = min(self.min_front_dist, r)
            # Правый сектор
            elif -1.57 < angle <= -0.5:
                self.min_right_dist = min(self.min_right_dist, r)
            # Левый сектор
            elif 0.5 <= angle < 1.57:
                self.min_left_dist = min(self.min_left_dist, r)
            # Задний сектор (работает, если лидар покрывает 360°)
            elif abs(angle) >= 1.57:
                self.has_rear_data = True
                self.min_rear_dist = min(self.min_rear_dist, r)

    def control_loop(self):
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.hypot(dx, dy)

        if distance < self.goal_tolerance:
            self.get_logger().info("Goal reached!")
            self.stop_robot()
            return

        min_any = min(self.min_front_dist, self.min_left_dist,
                  self.min_right_dist, self.min_rear_dist)
        
        if min_any < 0.01:
            self.get_logger().warn(
                f"EMERGENCY: obstacle at {min_any:.2f}m. Forcing avoidance.")
            self.stop_robot()
            self.state = 'AVOID_OBSTACLE'
            self.reverse_cooldown_steps = 0
            return
    
        if self.state == 'GO_TO_GOAL':
            side_too_close = (self.min_left_dist  < self.safe_dist or
                            self.min_right_dist < self.safe_dist)
            if self.min_front_dist < self.obstacle_threshold or side_too_close:
                self.get_logger().warn(
                    f"Obstacle: front={self.min_front_dist:.2f} "
                    f"left={self.min_left_dist:.2f} "
                    f"right={self.min_right_dist:.2f}. Switching to AVOID_OBSTACLE.")
                self.state = 'AVOID_OBSTACLE'
            else:
                self.drive_to_goal(dx, dy)

        elif self.state == 'AVOID_OBSTACLE':
            # Препятствие объехали — возвращаемся к цели
            if self.min_front_dist > self.obstacle_threshold and \
               self.min_right_dist > self.safe_dist and \
               self.min_left_dist > self.safe_dist:
                self.get_logger().info("Obstacle cleared. Returning to goal.")
                self.state = 'GO_TO_GOAL'
                return
            
            # Кулдаун после недавнего реверса — не даём осциллировать
            if self.reverse_cooldown_steps > 0:
                self.reverse_cooldown_steps -= 1
                self.follow_wall()
                return
            
            # Решаем: сдавать назад или объезжать поворотом?
            if self.should_reverse():
                self.get_logger().info(
                    f"Reversing (front={self.min_front_dist:.2f}m, "
                    f"left={self.min_left_dist:.2f}m, right={self.min_right_dist:.2f}m, "
                    f"rear={self.min_rear_dist:.2f}m)")
                self.state = 'REVERSE'
                self.reverse_steps = 0
            else:
                self.follow_wall()

        elif self.state == 'REVERSE':
            self.reverse_steps += 1
            rear_blocked  = self.min_rear_dist < self.safe_dist * 0.7
            front_cleared = self.min_front_dist > self.obstacle_threshold
            
            if rear_blocked:
                self.get_logger().info("Rear blocked. Aborting reverse.")
                self.state = 'AVOID_OBSTACLE'
                self.reverse_cooldown_steps = 5
            elif self.reverse_steps >= self.max_reverse_steps:
                self.get_logger().info("Max reverse steps reached.")
                self.state = 'AVOID_OBSTACLE'
                self.reverse_cooldown_steps = 5
            elif self.reverse_steps >= self.min_reverse_steps and front_cleared:
                self.get_logger().info("Front cleared. Continuing avoidance.")
                self.state = 'AVOID_OBSTACLE'
                self.reverse_cooldown_steps = 5
            else:
                self.do_reverse()

    def should_reverse(self):
        """Определяет, выгоднее ли сдать назад, чем разворачиваться."""
        # Спереди ещё есть запас — можно объехать поворотом
        if self.min_front_dist > self.safe_dist * 0.9:
            return False
        
        # Сзади препятствие — сдавать назад нельзя
        if self.min_rear_dist < self.safe_dist:
            return False
        
        # Спереди критически близко — сдаём назад, чтобы освободить место
        if self.min_front_dist < self.safe_dist * 0.6:
            return True
        
        # Проверяем, есть ли место для манёвра в стороны
        left_room  = self.min_left_dist  - self.safe_dist
        right_room = self.min_right_dist - self.safe_dist
        
        # Обе стороны перекрыты → развернуться негде → назад
        if left_room < 0.3 and right_room < 0.3:
            return True
        
        return False

    def do_reverse(self):
        """Движение назад с подворотом в сторону свободного пространства."""
        # Подворачиваем в сторону, где больше места
        if self.min_left_dist >= self.min_right_dist:
            steer_cmd = 0.3    # пятясь, крутим влево
        else:
            steer_cmd = -0.3   # пятясь, крутим вправо
        self.steer_pub.publish(Float32(data=steer_cmd))
        self.speed_pub.publish(Float32(data=self.reverse_speed))

    def drive_to_goal(self, dx, dy):
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - self.current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        steer_cmd = max(-0.5, min(0.5, yaw_error * 1.5))
        self.steer_pub.publish(Float32(data=steer_cmd))

        if abs(yaw_error) < 1.5:
            self.speed_pub.publish(Float32(data=40.0))
        else:
            self.speed_pub.publish(Float32(data=0.0))

    def follow_wall(self):
        steer_cmd = 0.0
        
        # 1. Препятствие впереди — поворачиваем в сторону, где больше места
        if self.min_front_dist < self.safe_dist:
            left_ok  = self.min_left_dist  > self.safe_dist
            right_ok = self.min_right_dist > self.safe_dist
            if left_ok and not right_ok:
                steer_cmd = 0.5     # только влево безопасно
            elif right_ok and not left_ok:
                steer_cmd = -0.5    # только вправо безопасно
            elif left_ok and right_ok:
                # обе стороны свободны — крутим к более просторной
                steer_cmd = 0.5 if self.min_left_dist >= self.min_right_dist else -0.5
            else:
                # обе стороны близко — should_reverse уже должен был дать REVERSE;
                steer_cmd = 0.5 if self.min_left_dist >= self.min_right_dist else -0.5    
        # 2. Стена справа слишком близко — отворачиваем влево
        elif self.min_right_dist < self.safe_dist:
            steer_cmd = 0.4
            
        # 3. Стена слева слишком близко — отворачиваем вправо
        elif self.min_left_dist < self.safe_dist:
            steer_cmd = -0.4
            
        # 4. Стена справа слишком далеко — плавно подруливаем к ней
        elif self.min_right_dist > self.wall_distance + 0.3:
            steer_cmd = -0.2
            
        self.steer_pub.publish(Float32(data=steer_cmd))
        self.speed_pub.publish(Float32(data=20.0))

    def stop_robot(self):
        self.speed_pub.publish(Float32(data=0.0))
        self.steer_pub.publish(Float32(data=0.0))


def main(args=None):
    rclpy.init(args=args)
    node = BugNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
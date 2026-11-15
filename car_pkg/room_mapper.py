#!/usr/bin/env python3
"""Frontier-based room mapper with Bug algorithm navigation and Interactive Map."""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math
import os
import time
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon as MplPolygon
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class RoomMapper(Node):
    def __init__(self):
        super().__init__('room_mapper')

        # --- EXPLORATION ---
        self.exploration_timeout = 900.0
        self.frontier_scan_interval = 2.0
        self.goal_max_time = 120.0

        # --- FRONTIER MEMORY ---
        self.frontier_visit_radius = 0.3
        self.frontier_fail_radius = 0.5
        self.frontier_retry_timeout = 60.0

        # --- BUG ALGORITHM (ИЗ POINT_NAVIGATOR) ---
        self.robot_radius = 0.3
        self.safety_margin = 0.1
        self.safe_dist = self.robot_radius + self.safety_margin
        
        self.goal_tolerance = 0.5
        self.obstacle_threshold = self.safe_dist + 0.5  # 1.2 м
        self.wall_distance = self.safe_dist + 0.1       # 0.8 м
        
        self.max_reverse_steps = 30       # ~1.5 с при 20 Гц
        self.min_reverse_steps = 5        # ~0.25 с (минимум, чтобы реально отъехать)
        self.reverse_cooldown_steps = 0   # защита от осцилляций
        self.reverse_speed = -20.0        # отрицательная скорость = назад
        
        self.state = 'GO_TO_GOAL'
        self.reverse_steps = 0

        self.min_front_dist = float('inf')
        self.min_right_dist = float('inf')
        self.min_left_dist = float('inf')
        self.min_rear_dist = float('inf')
        self.has_rear_data = False

        # --- STATE ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False

        self.target_x = None
        self.target_y = None
        self.current_goal_cell = None
        self.goal_start_time = None
        self.manual_goal = False

        self.visited_frontiers = set()
        self.failed_frontiers = {}
        self.exploration_start_time = None
        self.exploration_done = False
        self.last_frontier_scan = None

        self.trajectory = []
        self.trajectory_step = 0.05

        # --- MAP ---
        self.map_resolution = 0.1
        self.map_origin_x = -8.0
        self.map_origin_y = -6.0
        self.map_size_x = 22.0
        self.map_size_y = 16.0
        self.map_w = int(self.map_size_x / self.map_resolution)
        self.map_h = int(self.map_size_y / self.map_resolution)
        self.occupancy = np.zeros((self.map_h, self.map_w), dtype=np.int8)
        self.min_valid_dist = 0.25

        # --- ROS ---
        map_qos = QoSProfile(
                    depth=1,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=QoSReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        
        self.speed_pub = self.create_publisher(Float32, '/control/speed', 10)
        self.steer_pub = self.create_publisher(Float32, '/control/steering', 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # --- VISUALIZATION ---
        self.headless = (not os.environ.get('DISPLAY') and os.name != 'nt')
        self.fig = None
        self.ax = None
        self.im = None
        self.selected_x = None
        self.selected_y = None
        self.click_marker = None

        if HAS_MPL and not self.headless:
            try:
                matplotlib.use('TkAgg', force=True)
                self.setup_interactive_plot()
            except Exception as e:
                self.get_logger().error(f"Viz init error: {e}")
                self.headless = True

        self.create_timer(0.05, self.control_loop)
        self.create_timer(10.0, self.save_map)
        self.create_timer(0.5, self.update_plot_timer)

        self.get_logger().info(f"Frontier Mapper started. GUI: {not self.headless}")

    def setup_interactive_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        cmap = ListedColormap(['#ffffff', '#d9d9d9', '#000000'])
        
        self.im = self.ax.imshow(
            self.occupancy, cmap=cmap, origin='lower',
            extent=[self.map_origin_x, self.map_origin_x + self.map_size_x,
                    self.map_origin_y, self.map_origin_y + self.map_size_y],
            vmin=0, vmax=2, interpolation='nearest'
        )
        
        self.ax.set_title("Interactive Map: Click to set goal, press Enter to go")
        self.ax.set_xlabel('X, m')
        self.ax.set_ylabel('Y, m')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_map_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _on_map_click(self, event):
        if event.inaxes != self.ax: return
        self.selected_x = max(self.map_origin_x, min(event.xdata, self.map_origin_x + self.map_size_x))
        self.selected_y = max(self.map_origin_y, min(event.ydata, self.map_origin_y + self.map_size_y))
        
        if self.click_marker: self.click_marker.remove()
        self.click_marker, = self.ax.plot([self.selected_x], [self.selected_y], 'X', 
                                          color='magenta', markersize=12, markeredgewidth=2,
                                          label='Выбранная цель')
        self.ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()
        self.get_logger().info(f"Точка выбрана: ({self.selected_x:.2f}, {self.selected_y:.2f}). Нажмите Enter.")

    def _on_key_press(self, event):
        if event.key == 'enter' and self.selected_x is not None:
            self.target_x = self.selected_x
            self.target_y = self.selected_y
            self.manual_goal = True
            self.state = 'GO_TO_GOAL'
            self.goal_start_time = self.get_clock().now()
            self.exploration_done = False
            self.current_goal_cell = None
            
            self.get_logger().info(f"Ручная цель принята: ({self.target_x:.2f}, {self.target_y:.2f})")
            
            if self.click_marker:
                self.click_marker.remove()
                self.click_marker = None
            self.selected_x = None
            self.selected_y = None
            self.fig.canvas.draw_idle()

    def odom_callback(self, msg):
        self.odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.trajectory:
            self.trajectory.append((self.current_x, self.current_y))
        else:
            lx, ly = self.trajectory[-1]
            if math.hypot(self.current_x - lx, self.current_y - ly) >= self.trajectory_step:
                self.trajectory.append((self.current_x, self.current_y))

    def scan_callback(self, msg):
        if not self.odom_received: return

        self._update_map(msg)
        if not hasattr(self, '_map_pub_counter'):
            self._map_pub_counter = 0
        self._map_pub_counter += 1
        if self._map_pub_counter % 10 == 0:
            self.publish_map()

        self.min_front_dist = float('inf')
        self.min_right_dist = float('inf')
        self.min_left_dist = float('inf')
        self.min_rear_dist = float('inf')
        self.has_rear_data = False

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or r < msg.range_min: continue
            
            angle = msg.angle_min + i * msg.angle_increment
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            
            if -0.5 < angle < 0.5: self.min_front_dist = min(self.min_front_dist, r)
            elif -1.57 < angle <= -0.5: self.min_right_dist = min(self.min_right_dist, r)
            elif 0.5 <= angle < 1.57: self.min_left_dist = min(self.min_left_dist, r)
            elif abs(angle) >= 1.57:
                self.has_rear_data = True
                self.min_rear_dist = min(self.min_rear_dist, r)

    def _update_map(self, msg):
        rx = int((self.current_x - self.map_origin_x) / self.map_resolution)
        ry = int((self.current_y - self.map_origin_y) / self.map_resolution)

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r): continue
            if r < msg.range_min or r > msg.range_max: continue
            if r < self.min_valid_dist: continue

            angle = msg.angle_min + i * msg.angle_increment
            wx = self.current_x + r * math.cos(self.current_yaw + angle)
            wy = self.current_y + r * math.sin(self.current_yaw + angle)
            ex = int((wx - self.map_origin_x) / self.map_resolution)
            ey = int((wy - self.map_origin_y) / self.map_resolution)

            for cx, cy in self._bresenham(rx, ry, ex, ey):
                if 0 <= cx < self.map_w and 0 <= cy < self.map_h:
                    if self.occupancy[cy, cx] != 2:
                        self.occupancy[cy, cx] = 1

            no_hit = r >= msg.range_max - 0.001
            if not no_hit:
                if 0 <= ex < self.map_w and 0 <= ey < self.map_h:
                    self.occupancy[ey, ex] = 2

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield (x0, y0)
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x0 += sx
            if e2 < dx:
                err += dx; y0 += sy

    def _find_frontier_cells(self):
        occ = self.occupancy; h, w = occ.shape
        free = (occ == 1); unknown = (occ == 0)
        has_unknown_neighbor = np.zeros_like(free)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                shifted = np.zeros_like(unknown)
                r_src = slice(max(0, dr), h + min(0, dr))
                r_dst = slice(max(0, -dr), h + min(0, -dr))
                c_src = slice(max(0, dc), w + min(0, dc))
                c_dst = slice(max(0, -dc), w + min(0, -dc))
                shifted[r_dst, c_dst] = unknown[r_src, c_src]
                has_unknown_neighbor |= shifted

        frontier_mask = free & has_unknown_neighbor
        return np.argwhere(frontier_mask)

    def _select_new_goal(self):
        now = time.time()
        self.failed_frontiers = {
            c: t for c, t in self.failed_frontiers.items()
            if now - t < self.frontier_retry_timeout
        }

        frontier_cells = self._find_frontier_cells()
        if len(frontier_cells) == 0: return None

        rx = int((self.current_x - self.map_origin_x) / self.map_resolution)
        ry = int((self.current_y - self.map_origin_y) / self.map_resolution)

        best, best_dist = None, float('inf')
        for (fr, fc) in frontier_cells:
            fr, fc = int(fr), int(fc)
            if (fr, fc) in self.visited_frontiers: continue
            if (fr, fc) in self.failed_frontiers: continue

            is_safe = True
            check_radius = 3
            for r_off in range(-check_radius, check_radius + 1):
                for c_off in range(-check_radius, check_radius + 1):
                    check_r, check_c = fr + r_off, fc + c_off
                    if 0 <= check_r < self.map_h and 0 <= check_c < self.map_w:
                        if self.occupancy[check_r, check_c] == 2:
                            is_safe = False
                            break
                if not is_safe: break
            
            if not is_safe: continue

            d = math.hypot(fr - ry, fc - rx)
            if d < best_dist:
                best_dist = d
                best = (fr, fc)
        return best

    def _set_new_goal_from_cell(self, cell):
        fr, fc = cell
        self.current_goal_cell = cell
        self.target_x = self.map_origin_x + (fc + 0.5) * self.map_resolution
        self.target_y = self.map_origin_y + (fr + 0.5) * self.map_resolution
        self.goal_start_time = self.get_clock().now()
        self.state = 'GO_TO_GOAL'
        self.reverse_cooldown_steps = 0
        self.manual_goal = False
        self.get_logger().info(f"New frontier goal: ({self.target_x:.2f}, {self.target_y:.2f})")

    def _mark_frontier_visited(self):
        if self.current_goal_cell is None: return
        gr, gc = self.current_goal_cell
        r_cells = int(self.frontier_visit_radius / self.map_resolution)
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    r, c = gr + dr, gc + dc
                    if 0 <= r < self.map_h and 0 <= c < self.map_w:
                        if self.occupancy[r, c] == 1:
                            self.visited_frontiers.add((int(r), int(c)))

    def _mark_frontier_failed(self):
        if self.current_goal_cell is None: return
        gr, gc = self.current_goal_cell
        r_cells = int(self.frontier_fail_radius / self.map_resolution)
        now = time.time()
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    r, c = gr + dr, gc + dc
                    if 0 <= r < self.map_h and 0 <= c < self.map_w:
                        self.failed_frontiers[(int(r), int(c))] = now

    def _clear_goal(self):
        self.target_x = None
        self.target_y = None
        self.current_goal_cell = None
        self.goal_start_time = None
        self.state = 'GO_TO_GOAL'
        self.reverse_cooldown_steps = 0
        self.manual_goal = False

    # ===================== BUG ALGORITHM (ИЗ POINT_NAVIGATOR) =====================
    def should_reverse(self):
        """Определяет, выгоднее ли сдать назад, чем разворачиваться."""
        if self.min_front_dist > self.safe_dist * 0.9:
            return False
        
        if self.min_rear_dist < self.safe_dist:
            return False
        
        if self.min_front_dist < self.safe_dist * 0.6:
            return True
        
        left_room  = self.min_left_dist  - self.safe_dist
        right_room = self.min_right_dist - self.safe_dist
        
        if left_room < 0.3 and right_room < 0.3:
            return True
        
        return False

    def do_reverse(self):
        """Движение назад с подворотом в сторону свободного пространства."""
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
        
        if self.min_front_dist < self.safe_dist:
            left_ok  = self.min_left_dist  > self.safe_dist
            right_ok = self.min_right_dist > self.safe_dist
            if left_ok and not right_ok:
                steer_cmd = 0.5
            elif right_ok and not left_ok:
                steer_cmd = -0.5
            elif left_ok and right_ok:
                steer_cmd = 0.5 if self.min_left_dist >= self.min_right_dist else -0.5
            else:
                steer_cmd = 0.5 if self.min_left_dist >= self.min_right_dist else -0.5    
        elif self.min_right_dist < self.safe_dist:
            steer_cmd = 0.4
        elif self.min_left_dist < self.safe_dist:
            steer_cmd = -0.4
        elif self.min_right_dist > self.wall_distance + 0.3:
            steer_cmd = -0.2
            
        self.steer_pub.publish(Float32(data=steer_cmd))
        self.speed_pub.publish(Float32(data=20.0))

    def stop_robot(self):
        self.speed_pub.publish(Float32(data=0.0))
        self.steer_pub.publish(Float32(data=0.0))

    # ===================== CONTROL LOOP =====================
    def control_loop(self):
        if not self.odom_received: return

        if self.exploration_done:
            self.stop_robot()
            return

        if self.exploration_start_time is None:
            self.exploration_start_time = self.get_clock().now()

        elapsed = (self.get_clock().now() - self.exploration_start_time).nanoseconds / 1e9
        if elapsed > self.exploration_timeout:
            self.get_logger().info(f"Exploration timeout ({elapsed:.0f}s). Done.")
            self.exploration_done = True
            self.stop_robot()
            self.save_map()
            return

        if self.target_x is not None:
            goal_elapsed = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
            max_time = 300.0 if self.manual_goal else self.goal_max_time
            if goal_elapsed > max_time:
                self.get_logger().warn(f"Goal timeout ({goal_elapsed:.0f}s) — cancelling.")
                self.stop_robot()
                if self.current_goal_cell: self._mark_frontier_failed()
                self._clear_goal()
                return

            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            distance = math.hypot(dx, dy)

            if distance < self.goal_tolerance:
                self.get_logger().info("Goal reached!")
                self.stop_robot()
                if self.current_goal_cell: self._mark_frontier_visited()
                self._clear_goal()
                return

            min_any = min(self.min_front_dist, self.min_left_dist, self.min_right_dist, self.min_rear_dist)
            
            if min_any < 0.01:
                self.get_logger().warn(f"EMERGENCY: obstacle at {min_any:.2f}m.")
                self.stop_robot()
                self.state = 'AVOID_OBSTACLE'
                self.reverse_cooldown_steps = 0
                return
        
            if self.state == 'GO_TO_GOAL':
                side_too_close = (self.min_left_dist < self.safe_dist or self.min_right_dist < self.safe_dist)
                if self.min_front_dist < self.obstacle_threshold or side_too_close:
                    self.get_logger().warn("Obstacle. Switching to AVOID_OBSTACLE.")
                    self.state = 'AVOID_OBSTACLE'
                else:
                    self.drive_to_goal(dx, dy)

            elif self.state == 'AVOID_OBSTACLE':
                if self.min_front_dist > self.obstacle_threshold and \
                   self.min_right_dist > self.safe_dist and \
                   self.min_left_dist > self.safe_dist:
                    self.get_logger().info("Obstacle cleared. Returning to goal.")
                    self.state = 'GO_TO_GOAL'
                    return
                
                if self.reverse_cooldown_steps > 0:
                    self.reverse_cooldown_steps -= 1
                    self.follow_wall()
                    return
                
                if self.should_reverse():
                    self.get_logger().info("Reversing.")
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
            return

        # Если цели нет — стоим и ищем новый фронтир
        self.stop_robot()

        now_ts = time.time()
        if self.last_frontier_scan is not None and now_ts - self.last_frontier_scan < self.frontier_scan_interval:
            return
        self.last_frontier_scan = now_ts

        cell = self._select_new_goal()
        if cell is None:
            self.get_logger().info("No more frontiers — exploration complete. Click map and press Enter.")
            self.exploration_done = True
            self.stop_robot()
            self.save_map()
            return

        self._set_new_goal_from_cell(cell)

    def update_plot_timer(self):
        if not HAS_MPL or self.headless or self.fig is None: return
        try:
            self.im.set_data(self.occupancy)
            
            for artist in list(self.ax.lines):
                if artist.get_label() not in ['Выбранная цель']:
                    artist.remove()
            for patch in list(self.ax.patches):
                patch.remove()
            
            if self.trajectory:
                tx = [p[0] for p in self.trajectory]
                ty = [p[1] for p in self.trajectory]
                self.ax.plot(tx, ty, '-', color='#1976d2', linewidth=1.5, label='Траектория')
            
            if self.visited_frontiers:
                vx = [self.map_origin_x + (c + 0.5) * self.map_resolution for (r, c) in self.visited_frontiers]
                vy = [self.map_origin_y + (r + 0.5) * self.map_resolution for (r, c) in self.visited_frontiers]
                self.ax.plot(vx, vy, '.', color='#4caf50', markersize=2, alpha=0.4, label='Посещено')
            
            if self.target_x is not None:
                self.ax.plot([self.target_x], [self.target_y], '*', color='#ffeb3b',
                             markersize=18, markeredgecolor='black', markeredgewidth=1, zorder=10, label='Текущая цель')

            size = 0.35
            x, y, yaw = self.current_x, self.current_y, self.current_yaw
            pts = [
                (x + size * math.cos(yaw), y + size * math.sin(yaw)),
                (x + size * math.cos(yaw + 2.5), y + size * math.sin(yaw + 2.5)),
                (x + size * math.cos(yaw - 2.5), y + size * math.sin(yaw - 2.5)),
            ]
            self.ax.add_patch(MplPolygon(pts, closed=True, facecolor='#e53935', edgecolor='black', zorder=5, label='Робот'))

            self.ax.legend(loc='upper right', fontsize=8)
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def save_map(self):
        try:
            out_dir = '/tmp'
            pgm = np.full((self.map_h, self.map_w), 205, dtype=np.uint8)
            pgm[self.occupancy == 1] = 254
            pgm[self.occupancy == 2] = 0
            pgm = np.flipud(pgm)

            with open(os.path.join(out_dir, 'room_map.pgm'), 'wb') as f:
                f.write(b'P5\n')
                f.write(f'{self.map_w} {self.map_h}\n'.encode())
                f.write(b'255\n')
                f.write(pgm.tobytes())

            with open(os.path.join(out_dir, 'room_map.yaml'), 'w') as f:
                f.write('image: room_map.pgm\n')
                f.write(f'resolution: {self.map_resolution}\n')
                f.write(f'origin: [{self.map_origin_x:.3f}, {self.map_origin_y:.3f}, 0.0]\n')
                f.write('negate: 0\n')
                f.write('occupied_thresh: 0.65\n')
                f.write('free_thresh: 0.196\n')

            pts = []
            for j in range(self.map_h):
                for i in range(self.map_w):
                    cell = self.occupancy[j, i]
                    if cell == 0: continue
                    x = self.map_origin_x + (i + 0.5) * self.map_resolution
                    y = self.map_origin_y + (j + 0.5) * self.map_resolution
                    color = (255, 0, 0) if cell == 2 else (170, 170, 170)
                    pts.append((x, y, 0.0, *color))
            with open(os.path.join(out_dir, 'room_map.ply'), 'w') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write(f'element vertex {len(pts)}\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
                f.write('end_header\n')
                for p in pts:
                    f.write(f'{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]} {p[4]} {p[5]}\n')

            self.get_logger().info(f"Map saved: known={np.count_nonzero(self.occupancy)}")
        except Exception as e:
            self.get_logger().error(f"Save error: {e}")

    def publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'odom'
        grid.info.resolution = self.map_resolution
        grid.info.width = self.map_w
        grid.info.height = self.map_h
        grid.info.origin.position.x = self.map_origin_x
        grid.info.origin.position.y = self.map_origin_y
        grid.info.origin.orientation.w = 1.0

        data = np.full((self.map_h, self.map_w), -1, dtype=np.int8)
        data[self.occupancy == 1] = 0
        data[self.occupancy == 2] = 100
        grid.data = np.flipud(data).flatten().tolist()
        self.map_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = RoomMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_map()
        node.get_logger().info("Exiting.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
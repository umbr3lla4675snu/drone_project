#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import termios
import tty
import select
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from std_msgs.msg import Float32


def quat_to_yaw(qx, qy, qz, qw):
    """Quaternion -> yaw (rad)"""
    # ZYX 순
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class CfKeyboardTeleop(Node):
    """
    Crazyflie 2.1+ 키보드 컨트롤러 노드

    - /cf/odom, /cf/battery 를 subscribe 해서 상태 출력
    - /camera/image 를 subscribe 해서 카메라 FPS 집계
    - /cf/hl/goto (PoseStamped) 로 목표 위치/자세 명령 publish
    - /cf/hl/takeoff, /cf/hl/land 이륙/착륙 publish
    - /cf/stop (Trigger) 서비스 호출

    기본 키 매핑:
      t: takeoff
      l: land
      SPACE: emergency stop
      w/s: forward/back
      a/d: left/right
      r/f: up/down
      q/e: yaw left/right
      x: hover (현재 위치로 goto)
      h: help
    """

    def __init__(self):
        super().__init__('cf_keyboard_teleop')

        # ---- Parameters ----
        self.declare_parameter('step_xy_m', 0.5)
        self.declare_parameter('step_z_m', 0.3)
        self.declare_parameter('step_yaw_deg', 15.0)
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.step_xy = float(self.get_parameter('step_xy_m').value)
        self.step_z = float(self.get_parameter('step_z_m').value)
        self.step_yaw = math.radians(float(self.get_parameter('step_yaw_deg').value))
        self.world_frame = self.get_parameter('world_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # ---- QoS ----
        qos_ctrl = QoSProfile(depth=10)
        qos_ctrl.reliability = ReliabilityPolicy.RELIABLE
        qos_ctrl.history = HistoryPolicy.KEEP_LAST

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        cam_qos = QoSProfile(depth=1)
        cam_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        cam_qos.history = HistoryPolicy.KEEP_LAST

        # ---- State ----
        self._lock = threading.Lock()
        self._have_odom = False
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._yaw = 0.0
        self._battery_v = None

        # 카메라 상태 (FPS 계산용)
        self._cam_frame_count = 0
        self._cam_last_report_t = time.monotonic()

        # ---- Subscriptions ----
        self.sub_odom = self.create_subscription(
            Odometry, '/cf/odom', self._odom_cb, qos
        )
        self.sub_batt = self.create_subscription(
            BatteryState, '/cf/battery', self._battery_cb, qos
        )
        # ai_deck_camera_node.py 가 publish 하는 영상
        self.sub_cam = self.create_subscription(
            Image, '/camera/image', self._cam_cb, cam_qos
        )

        # ---- Publisher (goto) ----
        self.pub_goto    = self.create_publisher(PoseStamped, '/cf/hl/goto', qos_ctrl)
        self.pub_takeoff = self.create_publisher(Float32, '/cf/hl/takeoff', qos_ctrl)
        self.pub_land    = self.create_publisher(Float32, '/cf/hl/land', qos_ctrl)

        # ---- Service clients ----
        self.cli_emerg = self.create_client(Trigger, '/cf/stop')

        # 서비스가 올라왔는지 확인 (논블로킹으로 몇 번만 로그)
        self._check_services_once()

        # ---- Status 타이머 ----
        self.create_timer(1.0, self._status_timer)

        # ---- 키보드 스레드 ----
        self._key_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._key_thread.start()

        self._print_help()

    # ---------- Callbacks ----------
    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._have_odom = True
            self._x = msg.pose.pose.position.x
            self._y = msg.pose.pose.position.y
            self._z = msg.pose.pose.position.z
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            self._yaw = quat_to_yaw(qx, qy, qz, qw)

    def _battery_cb(self, msg: BatteryState):
        with self._lock:
            self._battery_v = msg.voltage

    def _cam_cb(self, msg: Image):
        # 단순히 카운트만 증가 (FPS 계산용)
        with self._lock:
            self._cam_frame_count += 1

    # ---------- Helpers ----------
    def _check_services_once(self):
        # 0초 타임아웃으로 즉시 확인만 하고, 결과는 로그로만 남김
        for name, cli in [
            # ('/cf/hl/takeoff', self.cli_takeoff),
            # ('/cf/hl/land', self.cli_land),
            ('/cf/emergency_stop', self.cli_emerg),
        ]:
            if not cli.service_is_ready():
                self.get_logger().warn(f'Service not ready yet: {name}\r')

    def _status_timer(self):
        with self._lock:
            have_odom = self._have_odom
            x, y, z, yaw = self._x, self._y, self._z, self._yaw
            v = self._battery_v

            # 카메라 FPS 계산
            now = time.monotonic()
            dt = now - self._cam_last_report_t
            cam_frames = self._cam_frame_count
            if dt > 0.0:
                cam_fps = cam_frames / dt
            else:
                cam_fps = 0.0
            # 다음 윈도우를 위해 리셋
            self._cam_frame_count = 0
            self._cam_last_report_t = now

        if have_odom:
            self.get_logger().info(
                f'POS (map): x={x:.2f} y={y:.2f} z={z:.2f} yaw={math.degrees(yaw):.1f} deg\r'
            )
        if v is not None:
            self.get_logger().info(f'Battery: {v:.2f} V\r')
        # 카메라 스트림 상태
        # self.get_logger().info(f'Camera FPS (approx): {cam_fps:.2f} Hz')

    def _send_goto(self, target_x, target_y, target_z, target_yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.pose.position.x = float(target_x)
        msg.pose.position.y = float(target_y)
        msg.pose.position.z = float(target_z)

        # yaw만 반영 (roll, pitch=0)
        cy = math.cos(target_yaw * 0.5)
        sy = math.sin(target_yaw * 0.5)
        msg.pose.orientation.z = sy
        msg.pose.orientation.w = cy
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0

        self.pub_goto.publish(msg)
        self.get_logger().info(
            f'GOTO → x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, yaw={math.degrees(target_yaw):.1f}deg\r'
        )

    def _call_trigger_async(self, cli: rclpy.client.Client, name: str):
        if not cli.service_is_ready():
            self.get_logger().warn(f'Service not ready: {name}\r')
            return
        req = Trigger.Request()
        future = cli.call_async(req)

        # 결과를 별도 콜백으로 로그만 출력
        def _done_cb(fut):
            try:
                resp = fut.result()
                self.get_logger().info(f'{name} → success={resp.success}, msg="{resp.message}"\r')
            except Exception as e:
                self.get_logger().error(f'{name} call failed: {e}\r')

        future.add_done_callback(_done_cb)

    # ---------- Keyboard ----------
    def _get_key(self, timeout=0.1):
        """non-blocking key read (Linux 전용)"""
        fd = sys.stdin.fileno()
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None

    def _keyboard_loop(self):
        # 터미널 설정 저장
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            self.get_logger().info('Keyboard teleop started. Press h for help.\r')

            while rclpy.ok():
                key = self._get_key(timeout=0.1)
                if key is None:
                    continue

                if key == '\x03':  # Ctrl+C
                    # 메인 스레드에서 KeyboardInterrupt 로 처리될 것
                    break

                self._handle_key(key)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str):
        key = key.lower()

        if key == 'h':
            self._print_help()
            return

        if key == 't':
            self.get_logger().info('[KEY] Takeoff\r')
            msg = Float32()
            msg.data = 0.4  # 원하는 takeoff 높이(m)
            self.pub_takeoff.publish(msg)
            return

        if key == 'l':
            self.get_logger().info('[KEY] Land\r')
            msg = Float32()
            msg.data = 0.0
            self.pub_land.publish(msg)
            return

        if key == ' ':
            self.get_logger().warn('[KEY] EMERGENCY STOP\r')
            self._call_trigger_async(self.cli_emerg, '/cf/emergency_stop')
            return

        # 위치 기반 명령은 odom 있어야 함
        with self._lock:
            if not self._have_odom:
                have_odom = False
            else:
                have_odom = True
                x, y, z, yaw = self._x, self._y, self._z, self._yaw

        if not have_odom:
            self.get_logger().warn('No /cf/odom yet, cannot move. Wait for odom.\r')
            return

        target_x, target_y, target_z, target_yaw = x, y, z, yaw

        # 몸체 기준 step → world frame 으로 회전
        dx_body = 0.0
        dy_body = 0.0
        dz = 0.0
        dyaw = 0.0

        if key == 'w':
            dx_body = self.step_xy
        elif key == 's':
            dx_body = -self.step_xy
        elif key == 'a':
            dy_body = self.step_xy
        elif key == 'd':
            dy_body = -self.step_xy
        elif key == 'r':
            dz = self.step_z
        elif key == 'f':
            dz = -self.step_z
        elif key == 'q':
            dyaw = self.step_yaw
        elif key == 'e':
            dyaw = -self.step_yaw
        elif key == 'x':
            # 현재 위치로 hover: goto 현재 위치 (yaw 유지)
            self.get_logger().info('[KEY] Hover at current pose\r')
            self._send_goto(x, y, z, yaw)
            return
        else:
            # 알 수 없는 키는 무시
            return

        # body → world
        dx_world = dx_body * math.cos(yaw) - dy_body * math.sin(yaw)
        dy_world = dx_body * math.sin(yaw) + dy_body * math.cos(yaw)

        target_x += dx_world
        target_y += dy_world
        target_z += dz
        target_yaw += dyaw

        # 안전상 z 최소/최대 제한 (예: 0.0 ~ 2.0m)
        target_z = max(0.0, min(2.0, target_z))

        self._send_goto(target_x, target_y, target_z, target_yaw)

    def _print_help(self):
        msg = """
===== Crazyflie Keyboard Teleop =====
t       : takeoff
l       : land
SPACE   : EMERGENCY STOP

w/s     : forward/back (step_xy_m)
a/d     : left/right   (step_xy_m)
r/f     : up/down      (step_z_m)
q/e     : yaw left/right (step_yaw_deg)

x       : hover (goto current pose)
h       : show this help
Ctrl+C  : exit
=====================================
"""
        print(msg)
        self.get_logger().info('Help printed\r')


def main(args=None):
    rclpy.init(args=args)
    node = CfKeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

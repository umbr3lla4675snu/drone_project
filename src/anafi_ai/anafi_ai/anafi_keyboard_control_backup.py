#!/usr/bin/env python3
# anafi_ai_moveby_keyboard.py
#
# Keyboard teleop for Parrot Anafi / Anafi AI using MoveByCommand
#
# - Publishes:
#     /anafi/drone/moveby       (anafi_ros_interfaces/MoveByCommand)
# - Subscribes:
#     /anafi/drone/moveby_done  (std_msgs/Bool)   ← 브리지에서 extended_move_by 결과 피드백
#     /anafi/camera/image       (sensor_msgs/Image, 기본값, 파라미터로 변경 가능)
# - Calls services:
#     /anafi/drone/takeoff      (std_srvs/Trigger)
#     /anafi/drone/land         (std_srvs/Trigger)
#     /anafi/drone/rth          (std_srvs/Trigger)
#     /anafi/drone/halt         (std_srvs/Trigger)
#     /anafi/skycontroller/offboard (std_srvs/SetBool, 옵션)
#
# MoveBy 축 정의 (bridge 코드 주석 기준):
#   d_x  : front (+) / back (−)
#   d_y  : right (+) / left (−)  ← 하지만 여기서는 오른쪽 이동을 dy<0 로 맞춰 사용
#   d_z  : down (+) / up (−)
#   d_psi: heading rotation [rad] (양수=CCW)
#
# 키 매핑 (한 번 누를 때마다 한 step 움직임):
#   w/s : 전/후 (±dx)
#   a/d : 좌/우 (±dy, d=오른쪽)
#   r/f : 상/하 (dz=∓step, r=위로)
#   q/e : 좌/우 yaw (±d_psi)
#   대문자(WASDQERFQ E): 터보 스텝 (기본의 turbo_multiplier 배)
#
#   t : 이륙
#   l : 착륙
#   h : RTH
#   k : HALT (강제 정지)
#   o : skycontroller/offboard 토글
#   z/x : linear/yaw step 크기 ↓/↑
#   ? : 도움말

import sys
import termios
import tty
import select
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool
from std_srvs.srv import Trigger, SetBool
from anafi_ros_interfaces.msg import MoveByCommand
from sensor_msgs.msg import Image


HELP_TEXT = r"""
Anafi AI MoveBy Keyboard Controller

이동 (MoveBy, 한 번 누를 때마다 한 스텝)
  w/s : +dx / -dx   (전/후)
  a/d : +dy / -dy   (좌/우, d=오른쪽)
  r/f : 위 / 아래  (dz = -/+ step, up/down)
  q/e : +yaw / -yaw (좌/우 회전, deg → rad)

스텝 크기
  기본 linear step : lin_step (m)  [기본 0.5 m]
  기본 yaw step    : yaw_step (deg) [기본 15 deg]
  대문자 입력(WASDQERFQE) : turbo_multiplier 배 (기본 2.0배)
  z/x : linear/yaw step 축소/확대

동작
  t : 이륙 (/anafi/drone/takeoff)
  l : 착륙 (/anafi/drone/land)
  h : 귀환 (/anafi/drone/rth)
  k : HALT 강제 정지 (/anafi/drone/halt)
  o : offboard 토글 (/anafi/skycontroller/offboard)
  ? : 이 도움말 출력

카메라
  /anafi/<camera_topic> (기본: /anafi/camera/image)를 subscribe 해서
  프레임 수신 여부와 간단한 정보만 로그로 남깁니다.

Ctrl+C 로 종료
"""


def _make_qos(depth=10, reliable=True):
    q = QoSProfile(
        depth=depth,
        reliability=(ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT),
        history=HistoryPolicy.KEEP_LAST,
    )
    return q


class _Keyboard:
    """터미널 비차단 단일 문자 입력 헬퍼"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def getch(self) -> Optional[str]:
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            try:
                return sys.stdin.read(1)
            except Exception:
                return None
        return None

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class AnafiMoveByKeyboard(Node):
    def __init__(self):
        # /anafi 네임스페이스에 고정
        super().__init__('anafi_moveby_keyboard', namespace='/anafi')

        # ---------- 파라미터 ----------
        # 스텝 크기
        self.declare_parameter('lin_step', 0.5)          # 기본 linear step [m]
        self.declare_parameter('yaw_step_deg', 15.0)     # 기본 yaw step [deg]
        self.declare_parameter('turbo_multiplier', 2.0)  # 대문자 입력시 배수

        # 카메라 토픽 (namespace 상대 경로)
        self.declare_parameter('camera_topic', 'camera/image')

        # 시작 시 offboard ON 여부
        self.declare_parameter('enable_offboard_on_start', True)

        self.lin_step = float(self.get_parameter('lin_step').value)
        self.yaw_step_deg = float(self.get_parameter('yaw_step_deg').value)
        self.turbo_multiplier = float(self.get_parameter('turbo_multiplier').value)
        self.camera_topic = self.get_parameter('camera_topic').value

        # ---------- 토픽/서비스 ----------
        qos_ctrl = _make_qos(depth=10, reliable=True)
        qos_cam = _make_qos(depth=5, reliable=False)

        # MoveBy pub/sub
        self.pub_moveby = self.create_publisher(MoveByCommand, 'drone/moveby', qos_ctrl)
        self.sub_moveby_done = self.create_subscription(
            Bool, 'drone/moveby_done', self._on_moveby_done, qos_ctrl
        )

        # 카메라
        # self.sub_camera = self.create_subscription(
        #     Image, self.camera_topic, self._on_camera, qos_cam
        # )

        # 서비스
        self.cli_takeoff = self.create_client(Trigger, 'drone/takeoff')
        self.cli_land = self.create_client(Trigger, 'drone/land')
        self.cli_rth = self.create_client(Trigger, 'drone/rth')
        self.cli_halt = self.create_client(Trigger, 'drone/halt')
        self.cli_offboard = self.create_client(SetBool, 'skycontroller/offboard')

        # 카메라 상태
        self._camera_frame_count = 0

        # 키보드 초기화
        self._kb = None
        try:
            self._kb = _Keyboard()
        except Exception as e:
            self.get_logger().warn(f"키보드 초기화 실패 (tty 필요): {e!r}")

        # 키보드용 타이머 (스레드 대신 타이머로 폴링)
        if self._kb:
            self.timer_kb = self.create_timer(0.01, self._keyboard_tick)
        else:
            self.get_logger().warn("키보드 입력이 없으므로 텔레옵은 동작하지 않습니다.")

        # 로그
        self.get_logger().info(HELP_TEXT)
        self.get_logger().info(f"MoveBy 토픽: /anafi/drone/moveby")
        self.get_logger().info(f"MoveBy 완료 토픽: /anafi/drone/moveby_done")
        self.get_logger().info(f"카메라 토픽: /anafi/{self.camera_topic}")
        self.get_logger().info(
            f"초기 스텝: lin_step={self.lin_step:.2f} m, yaw_step={self.yaw_step_deg:.1f} deg"
        )

        # 시작 시 offboard ON
        if bool(self.get_parameter('enable_offboard_on_start').value):
            self._set_offboard(True)

    # ---------- 카메라 콜백 ----------
    def _on_camera(self, msg: Image):
        self._camera_frame_count += 1
        if self._camera_frame_count == 1:
            self.get_logger().info(
                f"카메라 첫 프레임 수신: {msg.width}x{msg.height}, encoding={msg.encoding}"
            )
        elif self._camera_frame_count % 100 == 0:
            self.get_logger().debug(
                f"카메라 프레임 {self._camera_frame_count}개 수신 중..."
            )

    # ---------- MoveBy 완료 콜백 ----------
    def _on_moveby_done(self, msg: Bool):
        if msg.data:
            self.get_logger().info("MoveBy 완료 ✅")
        else:
            self.get_logger().warn("MoveBy 실패 ❌")

    # ---------- 서비스 헬퍼 ----------
    def _call_trigger_async(self, client, name: str):
        if not client.service_is_ready():
            self.get_logger().info(f"{name} 서비스 대기 중...")
            if not client.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn(f"{name} 서비스 준비 안 됨")
                return
        fut = client.call_async(Trigger.Request())

        def _done(_):
            try:
                resp = fut.result()
                if resp is None:
                    self.get_logger().warn(f"{name} 응답 없음")
                    return
                if resp.success:
                    self.get_logger().info(f"{name}: {resp.message}")
                else:
                    self.get_logger().warn(f"{name} 실패: {resp.message}")
            except Exception as e:
                self.get_logger().error(f"{name} 호출 오류: {e!r}")

        fut.add_done_callback(_done)

    def _set_offboard(self, enable: bool):
        if not self.cli_offboard.service_is_ready():
            self.get_logger().info("offboard 서비스 대기 중...")
            if not self.cli_offboard.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn("offboard 서비스 준비 안 됨")
                return

        from std_srvs.srv import SetBool  # 재확인용

        req = SetBool.Request()
        req.data = enable
        fut = self.cli_offboard.call_async(req)

        def _done(_):
            try:
                resp = fut.result()
                if resp is None:
                    self.get_logger().warn("offboard 응답 없음")
                    return
                state = "ON" if (enable and resp.success) else "OFF"
                self.get_logger().warning(f"Offboard → {state} ({resp.message})")
            except Exception as e:
                self.get_logger().error(f"offboard 설정 오류: {e!r}")

        fut.add_done_callback(_done)

    # ---------- MoveBy 발행 ----------
    def _publish_moveby(self, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0):
        msg = MoveByCommand()
        msg.dx = float(dx)
        msg.dy = float(dy)
        msg.dz = float(dz)
        msg.dyaw = float(dyaw)
        self.pub_moveby.publish(msg)
        self.get_logger().info(
            f"MoveBy PUB → dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, dyaw={dyaw:.3f} rad"
        )

    # ---------- 키보드 폴링 ----------
    def _keyboard_tick(self):
        ch = self._kb.getch()
        if ch is None:
            return

        turbo = ch.isupper()
        lin = self.lin_step * (self.turbo_multiplier if turbo else 1.0)
        yaw_deg = self.yaw_step_deg * (self.turbo_multiplier if turbo else 1.0)
        yaw_rad = yaw_deg * 3.141592653589793 / 180.0

        # --- 이동 키 (MoveBy) ---
        if ch in ('w', 'W'):
            # 전진 (+dx)
            self._publish_moveby(dx=lin)
        elif ch in ('s', 'S'):
            # 후진 (-dx)
            self._publish_moveby(dx=-lin)
        elif ch in ('d', 'D'):
            # 오른쪽 (dy>0 로 사용, TestFlight 예제에 맞춰)
            self._publish_moveby(dy=lin)
        elif ch in ('a', 'A'):
            # 왼쪽 (dy<0)
            self._publish_moveby(dy=-lin)
        elif ch in ('r', 'R'):
            # 위로 (dz<0, down axis 기준)
            self._publish_moveby(dz=-lin)
        elif ch in ('f', 'F'):
            # 아래로 (dz>0)
            self._publish_moveby(dz=lin)
        elif ch in ('q', 'Q'):
            # 좌회전 (+yaw)
            self._publish_moveby(dyaw=yaw_rad)
        elif ch in ('e', 'E'):
            # 우회전 (-yaw)
            self._publish_moveby(dyaw=-yaw_rad)

        # --- 스텝 크기 조절 ---
        elif ch == 'z':
            self.lin_step = max(0.1, self.lin_step * 0.8)
            self.yaw_step_deg = max(1.0, self.yaw_step_deg * 0.8)
            self.get_logger().info(
                f"스텝 축소: lin_step={self.lin_step:.2f} m, yaw_step={self.yaw_step_deg:.1f} deg"
            )
        elif ch == 'x':
            self.lin_step = min(5.0, self.lin_step * 1.25)
            self.yaw_step_deg = min(90.0, self.yaw_step_deg * 1.25)
            self.get_logger().info(
                f"스텝 확대: lin_step={self.lin_step:.2f} m, yaw_step={self.yaw_step_deg:.1f} deg"
            )

        # --- 드론 동작 키 ---
        elif ch == 't':
            self.get_logger().warning("[키보드] 이륙 요청 (t)")
            self._call_trigger_async(self.cli_takeoff, 'takeoff')
        elif ch == 'l':
            self.get_logger().warning("[키보드] 착륙 요청 (l)")
            self._call_trigger_async(self.cli_land, 'land')
        elif ch == 'h':
            self.get_logger().warning("[키보드] RTH 요청 (h)")
            self._call_trigger_async(self.cli_rth, 'rth')
        elif ch == 'k':
            self.get_logger().warning("[키보드] HALT 강제 정지 요청 (k)")
            self._call_trigger_async(self.cli_halt, 'halt')
        elif ch == 'o':
            self.get_logger().warning("[키보드] offboard 토글 요청 (o)")
            # 간단하게 토글만, 실제 상태는 응답 로그로 확인
            self._set_offboard(True)  # 필요하면 토글 상태를 멤버로 들고가도 됨
        elif ch == '?':
            self.get_logger().info(HELP_TEXT)
        # 그 외 키는 무시


def main():
    rclpy.init()
    node = AnafiMoveByKeyboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, "_kb") and node._kb:
            try:
                node._kb.restore()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

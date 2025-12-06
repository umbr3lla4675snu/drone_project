#!/usr/bin/env python3
"""
Person Counter Node

/camera/image 토픽을 구독하여 YOLO로 person 감지 후
감지된 person 수를 /person_count 토픽으로 퍼블리시

추가: 키보드 't'를 누르면 /cf/hl/takeoff (Float32) 퍼블리시하여 이륙

Usage:
    ros2 run anafi_ai person_counter
    
    # 다른 이미지 토픽 사용
    ros2 run anafi_ai person_counter --ros-args -p image_topic:=/other/image
"""

import sys
import termios
import tty
import select

import cv2
import time
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float32
from std_srvs.srv import Trigger

from ultralytics import YOLO


class _Keyboard:
    """터미널 비차단 단일 문자 입력"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def getch(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            try:
                return sys.stdin.read(1)
            except Exception:
                return None
        return None

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class PersonCounter(Node):
    def __init__(self):
        super().__init__('person_counter')

        # ---------- 파라미터 ----------
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('model', 'yolo11n.pt')  # 가벼운 모델 사용
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('threshold', 0.3)
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('takeoff_height', 1.0)  # m
        self.declare_parameter('stabilize_sec', 5.0)  # 감지 변화 확정 대기 시간

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model').value
        self.device = self.get_parameter('device').value
        self.threshold = self.get_parameter('threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.takeoff_height = float(self.get_parameter('takeoff_height').value)
        self.stabilize_sec = float(self.get_parameter('stabilize_sec').value)

        # ---------- YOLO 모델 로드 ----------
        self.get_logger().info(f"YOLO 모델 로드 중: {self.model_path}")
        self.yolo = YOLO(self.model_path)
        self.yolo.fuse()
        self.get_logger().info("YOLO 모델 로드 완료")

        # ---------- CV Bridge ----------
        self.cv_bridge = CvBridge()

        # ---------- 상태 ----------
        self.latest_image = None
        self.person_count = 0
        self.stable_count = 0  # 내부 확정 카운트
        self._pending_count = 0
        self._pending_since = None

        # ---------- QoS ----------
        qos_image = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        qos_pub = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ---------- Subscriber ----------
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_image
        )

        # ---------- Publisher ----------
        self.pub_count = self.create_publisher(Int32, '/person_count', qos_pub)  # 실시간 감지 수
        self.pub_count_stable = self.create_publisher(Int32, '/person_count_stable', qos_pub)  # 확정 카운트
        self.pub_takeoff = self.create_publisher(Float32, '/cf/hl/takeoff', qos_pub)
        self.pub_land = self.create_publisher(Float32, '/cf/hl/land', qos_pub)

        # ---------- Service Clients ----------
        self.cli_stop = self.create_client(Trigger, '/cf/stop')

        # ---------- 처리 타이머 ----------
        self.timer = self.create_timer(1.0 / self.publish_rate, self._process_tick)

        # ---------- 키보드 ----------
        self._kb = None
        try:
            self._kb = _Keyboard()
        except Exception as e:
            self.get_logger().warn(f"키보드 초기화 실패: {e}")

        if self._kb:
            self.timer_kb = self.create_timer(0.02, self._keyboard_tick)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Person Counter Node 시작")
        self.get_logger().info(f"  이미지 토픽: {self.image_topic}")
        self.get_logger().info(f"  모델: {self.model_path}")
        self.get_logger().info(f"  퍼블리시: /person_count (실시간), /person_count_stable (확정)")
        self.get_logger().info("  t : takeoff (/cf/hl/takeoff)")
        self.get_logger().info("  l : land (/cf/hl/land)")
        self.get_logger().info("  k : emergency stop (/cf/stop)")
        self.get_logger().info("=" * 50)

    def _on_image(self, msg: Image):
        """이미지 수신"""
        self.latest_image = msg

    def _process_tick(self):
        """주기적으로 YOLO 처리 및 person 수 퍼블리시"""
        if self.latest_image is None:
            return

        try:
            # 이미지 변환
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                self.latest_image, desired_encoding='passthrough'
            )

            # mono8이면 BGR로 변환
            if len(cv_image.shape) == 2 or (len(cv_image.shape) == 3 and cv_image.shape[2] == 1):
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            # YOLO 추론
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device,
                classes=[0],  # person class만
            )

            # person 수 카운트
            count = 0
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    count = len(boxes)

            self.person_count = count

            # 실시간 퍼블리시
            msg = Int32(); msg.data = count
            self.pub_count.publish(msg)

            # 안정화 로직: 감지 수 변화가 5초 유지되면 내부 카운트 변경
            now = time.time()
            if count != self._pending_count:
                self._pending_count = count
                self._pending_since = now
            else:
                if self._pending_since is not None and (now - self._pending_since) >= self.stabilize_sec:
                    diff = self._pending_count - self.stable_count
                    if diff == 1:
                        self.stable_count += 1
                        self.get_logger().warning(f"확정 카운트 +1 → {self.stable_count}")
                    elif diff == -1:
                        self.stable_count -= 1
                        self.get_logger().warning(f"확정 카운트 -1 → {self.stable_count}")
                    # 변화 반영 후 pending 초기화
                    self._pending_since = None

            # 확정 카운트 퍼블리시
            msg_stable = Int32(); msg_stable.data = self.stable_count
            self.pub_count_stable.publish(msg_stable)

            self.get_logger().info(
                f"Person count: {count} (stable={self.stable_count})",
                throttle_duration_sec=1.0,
            )

        except Exception as e:
            self.get_logger().error(f"처리 오류: {e}")

    # ---------- 키보드 처리 ----------
    def _keyboard_tick(self):
        ch = self._kb.getch()
        if ch is None:
            return

        if ch == 't':
            msg = Float32()
            msg.data = float(self.takeoff_height)
            self.pub_takeoff.publish(msg)
            self.get_logger().warning(f"takeoff 명령 발행: {msg.data:.2f} m")
        elif ch == 'l':
            msg = Float32()
            msg.data = 0.0
            self.pub_land.publish(msg)
            self.get_logger().warning("land 명령 발행")
        elif ch == 'k':
            if not self.cli_stop.service_is_ready():
                self.get_logger().warn("/cf/stop 서비스 대기 중...")
                if not self.cli_stop.wait_for_service(timeout_sec=1.0):
                    self.get_logger().error("/cf/stop 서비스 없음")
                    return
            fut = self.cli_stop.call_async(Trigger.Request())

            def _done(_):
                try:
                    resp = fut.result()
                    if resp and resp.success:
                        self.get_logger().warning("EMERGENCY STOP 실행")
                    else:
                        self.get_logger().error("EMERGENCY STOP 실패")
                except Exception as e:
                    self.get_logger().error(f"EMERGENCY STOP 오류: {e}")

            fut.add_done_callback(_done)


def main():
    rclpy.init()
    node = PersonCounter()
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

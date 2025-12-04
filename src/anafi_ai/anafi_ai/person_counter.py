#!/usr/bin/env python3
"""
Person Counter Node

/camera/image 토픽을 구독하여 YOLO로 person 감지 후
감지된 person 수를 /person_count 토픽으로 퍼블리시

Usage:
    ros2 run anafi_ai person_counter
    
    # 다른 이미지 토픽 사용
    ros2 run anafi_ai person_counter --ros-args -p image_topic:=/other/image
"""

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Int32

from ultralytics import YOLO


class PersonCounter(Node):
    def __init__(self):
        super().__init__('person_counter')

        # ---------- 파라미터 ----------
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('model', 'yolo11n.pt')  # 가벼운 모델 사용
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('publish_rate', 1.0)  # Hz

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model').value
        self.device = self.get_parameter('device').value
        self.threshold = self.get_parameter('threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value

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
        self.pub_count = self.create_publisher(Int32, '/person_count', qos_pub)

        # ---------- 처리 타이머 ----------
        self.timer = self.create_timer(1.0 / self.publish_rate, self._process_tick)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Person Counter Node 시작")
        self.get_logger().info(f"  이미지 토픽: {self.image_topic}")
        self.get_logger().info(f"  모델: {self.model_path}")
        self.get_logger().info(f"  퍼블리시: /person_count")
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

            # 퍼블리시
            msg = Int32()
            msg.data = count
            self.pub_count.publish(msg)

            self.get_logger().info(f"Person count: {count}", throttle_duration_sec=1.0)

        except Exception as e:
            self.get_logger().error(f"처리 오류: {e}")


def main():
    rclpy.init()
    node = PersonCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

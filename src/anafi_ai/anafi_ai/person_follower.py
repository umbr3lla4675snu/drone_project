#!/usr/bin/env python3
"""
Person Follower Node for Parrot Anafi

키 매핑:
  t : 이륙 및 추적 시작
  l : 착륙
  k : 긴급 정지
  스페이스 : 추적 일시정지/재개
  r : 녹화 시작/중단
  d : 녹화 파일 다운로드
"""

import sys
import termios
import tty
import select
import time
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool
from std_srvs.srv import Trigger, SetBool
from anafi_ros_interfaces.msg import MoveByCommand, GimbalCommand, CameraCommand
from anafi_ros_interfaces.srv import Recording
from yolo_msgs.msg import DetectionArray, Detection, KeyPoint2DArray


def _make_qos(depth=10, reliable=True):
    return QoSProfile(
        depth=depth,
        reliability=(ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT),
        history=HistoryPolicy.KEEP_LAST,
    )


class _Keyboard:
    """터미널 비차단 단일 문자 입력"""

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


class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower', namespace='/anafi')

        # ---------- 파라미터 ----------
        self.declare_parameter('image_width', 1920)  # 카메라 이미지 가로 해상도
        self.declare_parameter('image_height', 1080)  # 카메라 이미지 세로 해상도
        self.declare_parameter('center_deadzone', 60)  # 중심 허용 오차 (픽셀)
        self.declare_parameter('move_step', 0.13)  # Y축 이동 기본 스텝 (m)
        self.declare_parameter('move_step_max', 1.0)  # Y축 이동 최대 스텝 (m)
        self.declare_parameter('control_rate', 4.0)  # 제어 주기 (Hz)
        self.declare_parameter('no_target_timeout', 5.0)  # 복귀 대기 시간 (초)
        self.declare_parameter('gimbal_pitch_gain', 0.02)  # 짐벌 피치 게인 (deg/pixel)
        self.declare_parameter('gimbal_deadzone', 20)  # 짐벌 제어 데드존 (픽셀)
        # self.declare_parameter('tracking_zoom', 1.2)  # 추적 중 줌 배율
        # self.declare_parameter('default_zoom', 1.0)  # 기본 줌 배율

        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.center_deadzone = self.get_parameter('center_deadzone').value
        self.move_step = self.get_parameter('move_step').value
        self.move_step_max = self.get_parameter('move_step_max').value
        self.control_rate = self.get_parameter('control_rate').value
        self.no_target_timeout = self.get_parameter('no_target_timeout').value
        self.gimbal_pitch_gain = self.get_parameter('gimbal_pitch_gain').value
        self.gimbal_deadzone = self.get_parameter('gimbal_deadzone').value
        # self.tracking_zoom = self.get_parameter('tracking_zoom').value
        # self.default_zoom = self.get_parameter('default_zoom').value

        # 이미지 중심 좌표
        self.image_center_x = self.image_width / 2.0
        self.image_center_y = self.image_height / 2.0
        
        # 현재 짐벌 피치 각도 (마지막 전송값)
        self.current_gimbal_pitch = 0.0
        
        # 현재 줌 상태
        # self.current_zoom = self.default_zoom
        self.is_zoomed_in = False

        # ---------- 상태 변수 ----------
        self.is_flying = False
        self.tracking_enabled = False
        self.target_person: Optional[Detection] = None
        self.last_detections: List[Detection] = []

        # 이동 누적 추적 (시작 위치 복귀용)
        self.total_dy = 0.0  # 이륙 이후 총 Y축 이동량

        # 타겟 없음 타이머
        self.last_target_seen_time: Optional[float] = None
        self.returning_home = False  # 복귀 중 플래그
        self.waiting_at_home = False  # 홈에서 대기 중

        # 현재 추적 중인 사람의 ID
        self.tracking_id: Optional[str] = None
        
        # 이륙 후 초기 상승 대기 플래그
        self.waiting_for_initial_ascent = False

        # ---------- 토픽/서비스 ----------
        qos_ctrl = _make_qos(depth=10, reliable=True)
        qos_yolo = _make_qos(depth=5, reliable=False)

        # YOLO tracking 구독
        self.sub_tracking = self.create_subscription(
            DetectionArray, '/yolo/tracking', self._on_tracking, qos_yolo
        )

        # MoveBy 퍼블리셔
        self.pub_moveby = self.create_publisher(MoveByCommand, 'drone/moveby', qos_ctrl)

        # Gimbal 퍼블리셔
        self.pub_gimbal = self.create_publisher(GimbalCommand, 'gimbal/command', qos_ctrl)

        # Camera 퍼블리셔 (줌)
        self.pub_camera = self.create_publisher(CameraCommand, 'camera/command', qos_ctrl)

        # MoveBy 완료 구독
        self.sub_moveby_done = self.create_subscription(
            Bool, 'drone/moveby_done', self._on_moveby_done, qos_ctrl
        )

        # 서비스 클라이언트
        self.cli_takeoff = self.create_client(Trigger, 'drone/takeoff')
        self.cli_land = self.create_client(Trigger, 'drone/land')
        self.cli_halt = self.create_client(Trigger, 'drone/halt')
        self.cli_offboard = self.create_client(SetBool, 'skycontroller/offboard')
        self.cli_record_start = self.create_client(Recording, 'camera/recording/start')
        self.cli_record_stop = self.create_client(Recording, 'camera/recording/stop')
        self.cli_download = self.create_client(SetBool, 'storage/download')

        # 녹화 상태
        self.is_recording = False

        # ---------- 키보드 ----------
        self._kb = None
        try:
            self._kb = _Keyboard()
        except Exception as e:
            self.get_logger().warn(f"키보드 초기화 실패: {e}")

        if self._kb:
            self.timer_kb = self.create_timer(0.02, self._keyboard_tick)

        # ---------- 제어 루프 타이머 ----------
        self.timer_control = self.create_timer(1.0 / self.control_rate, self._control_tick)

        # 이동 중 플래그
        self.is_moving = False
        self.move_start_time: Optional[float] = None
        self.move_timeout = 5.0  # 이동 타임아웃 (초)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Person Follower Node 시작")
        self.get_logger().info("  t : 이륙 및 추적 시작")
        self.get_logger().info("  l : 착륙")
        self.get_logger().info("  k : 긴급 정지")
        self.get_logger().info("  스페이스 : 추적 일시정지/재개")
        # self.get_logger().info("  r : 녹화 시작/중단")
        # self.get_logger().info("  d : 녹화 파일 다운로드")
        self.get_logger().info(f"  정면 타겟 없으면 {self.no_target_timeout}초 후 이륙지점 복귀")
        self.get_logger().info("=" * 50)

    # ---------- YOLO Tracking 콜백 ----------
    def _on_tracking(self, msg: DetectionArray):
        """YOLO tracking 결과 수신"""
        self.last_detections = msg.detections

    # ---------- 정면 판단 ----------
    def _is_facing_front(self, detection: Detection) -> bool:
        """
        키포인트로 사람이 정면을 보고 있는지 판단
        COCO keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        """
        if not detection.keypoints or not detection.keypoints.data:
            return False

        keypoints = detection.keypoints.data

        # 키포인트 ID별 confidence 추출
        conf_map = {}
        for kp in keypoints:
            conf_map[kp.id] = kp.score

        # 얼굴 키포인트 (1=nose, 2=left_eye, 3=right_eye, 4=left_ear, 5=right_ear)
        # yolo_node에서 id = 1부터 시작
        nose_conf = conf_map.get(1, 0.0)
        left_eye_conf = conf_map.get(2, 0.0)
        right_eye_conf = conf_map.get(3, 0.0)
        left_ear_conf = conf_map.get(4, 0.0)
        right_ear_conf = conf_map.get(5, 0.0)

        face_score = nose_conf + left_eye_conf + right_eye_conf

        # 앞모습 조건: 얼굴(코, 눈)이 보임
        if face_score > 1.5:
            return True

        # 뒷모습: 양쪽 귀가 모두 보이고 얼굴이 안 보임
        if left_ear_conf > 0.5 and right_ear_conf > 0.5 and face_score < 0.5:
            return False

        return face_score > 0.8

    # ---------- 타겟 선택 ----------
    def _find_front_facing_person(self) -> Optional[Detection]:
        """정면을 보고 있는 사람 중 가장 큰 사람 선택"""
        candidates = []

        for det in self.last_detections:
            if det.class_name != 'person':
                continue
            if self._is_facing_front(det):
                size = det.bbox.size.x * det.bbox.size.y
                candidates.append((det, size))

        if not candidates:
            return None

        # 가장 큰 사람 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _find_tracked_person(self) -> Optional[Detection]:
        """현재 추적 중인 ID의 사람 찾기 (정면/후면 무관)"""
        if self.tracking_id is None:
            return None

        for det in self.last_detections:
            if det.class_name != 'person':
                continue
            if det.id == self.tracking_id:
                return det

        return None

    # ---------- 제어 루프 ----------
    def _control_tick(self):
        if not self.is_flying or not self.tracking_enabled:
            return

        # 이동 타임아웃 체크
        if self.is_moving:
            if self.move_start_time and (time.time() - self.move_start_time) > self.move_timeout:
                self.get_logger().warn("이동 타임아웃 - 강제 해제")
                self.is_moving = False
                self.move_start_time = None
            else:
                # 이전 이동 완료 대기
                return

        # 홈에서 대기 중이면 정면 타겟만 찾기
        if self.waiting_at_home:
            target = self._find_front_facing_person()
            if target is not None:
                self.get_logger().warning(f"정면 타겟 다시 감지! (ID={target.id}) 추적 재개")
                self.waiting_at_home = False
                self.last_target_seen_time = time.time()
                self.tracking_id = target.id  # 새 타겟 ID 설정
            else:
                self.get_logger().info("홈에서 대기 중...", throttle_duration_sec=2.0)
            return

        # 복귀 중이면 복귀 완료 대기
        if self.returning_home:
            return

        current_time = time.time()
        target = None

        # 1. 현재 추적 중인 사람이 있으면 그 사람 계속 추적 (정면/후면 무관)
        if self.tracking_id is not None:
            target = self._find_tracked_person()
            if target is not None:
                facing = "정면" if self._is_facing_front(target) else "후면"
                self.get_logger().debug(f"추적 중: ID={self.tracking_id} ({facing})")

        # 2. 추적 중인 사람이 화면에서 사라졌으면 새로운 정면 타겟 찾기
        if target is None:
            if self.tracking_id is not None:
                self.get_logger().info(f"추적 대상 (ID={self.tracking_id}) 화면에서 사라짐")
                self.tracking_id = None
                # self._zoom_out_to_default()

            target = self._find_front_facing_person()
            if target is not None:
                self.tracking_id = target.id
                self.get_logger().warning(f"새 타겟 감지! ID={target.id}")
                # self._zoom_in_for_tracking()

        # 타겟 없음 처리
        if target is None:
            # self._zoom_out_to_default()
            
            if self.last_target_seen_time is None:
                self.last_target_seen_time = current_time

            elapsed = current_time - self.last_target_seen_time
            remaining = self.no_target_timeout - elapsed

            if elapsed >= self.no_target_timeout:
                # 5초 이상 타깃 없음 → 시작 위치로 Y축 복귀
                self._return_to_home()
            else:
                self.get_logger().info(
                    f"타겟 없음 (복귀까지 {remaining:.1f}초)",
                    throttle_duration_sec=1.0
                )

            self.target_person = None
            return

        # 타겟 발견 → 줌 인
        # self._zoom_in_for_tracking()

        # 타겟 발견 → 타이머 리셋
        self.last_target_seen_time = current_time
        self.target_person = target

        # 짐벌로 얼굴 추적 (Y축)
        self._control_gimbal(target)

        # 타겟의 X 좌표 (이미지에서)
        target_x = target.bbox.center.position.x

        # 중심과의 차이
        error_x = target_x - self.image_center_x

        facing = "정면" if self._is_facing_front(target) else "후면"
        self.get_logger().info(
            f"타겟 ID={target.id} ({facing}): x={target_x:.0f}, 오차={error_x:.0f}px"
        )

        # deadzone 내면 이동 안 함
        if abs(error_x) < self.center_deadzone:
            self.get_logger().info("타겟이 중심에 있음 ✓")
            return

        # Y축 이동 방향 및 크기 결정 (오차 클수록 스텝 증가, 최대 move_step_max)
        error_mag = abs(error_x)
        ratio = error_mag / self.center_deadzone if self.center_deadzone > 0 else 1.0
        scale = max(1.0, ratio)
        dy_mag = min(self.move_step_max, self.move_step * scale)

        if error_x > 0:
            dy = dy_mag
            direction = "오른쪽"
        else:
            dy = -dy_mag
            direction = "왼쪽"

        if(self.total_dy + dy > 5 or self.total_dy + dy < -5):
            self.get_logger().warn("최대 이동 한도 도달")
            if(self.total_dy + dy > 5):
                dy = 5 - self.total_dy
            else:
                dy = -5 - self.total_dy
            if abs(dy) < 0.01:
                self.get_logger().info("더 이상 이동 불가")
                return

        self.get_logger().info(f"이동: {direction} (dy={dy:.2f}m)")
        self._publish_moveby(dy=dy)
        self.total_dy += dy  # 이동량 누적
        self.get_logger().info(f"누적 Y 이동량: {self.total_dy:.2f}m")
        self.is_moving = True
        self.move_start_time = time.time()

    # ---------- 시작 위치로 Y축 복귀 ----------
    def _return_to_home(self):
        """시작 위치로 복귀 (공중에서 Y축 이동만 되돌림)"""
        if abs(self.total_dy) < 0.1:
            self.get_logger().warning("이미 시작 위치 근처, 대기 모드 진입")
            self.waiting_at_home = True
            self.returning_home = False
            return

        # 반대 방향으로 Y축만 이동 (공중에서)
        return_dy = -self.total_dy
        self.get_logger().warning(f"시작 위치로 Y축 복귀 (dy={return_dy:.2f}m, 공중 유지)")

        self.returning_home = True
        self._publish_moveby(dy=return_dy)
        self.is_moving = True
        self.move_start_time = time.time()

    # ---------- MoveBy ----------
    def _publish_moveby(self, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0):
        msg = MoveByCommand()
        msg.dx = float(dx)
        msg.dy = float(dy)
        msg.dz = float(dz)
        # msg.dyaw = float(dyaw)
        msg.dyaw = 0.0
        self.get_logger().info(
            f"MoveBy PUB → dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, dyaw={dyaw:.3f} rad"
        )
        self.pub_moveby.publish(msg)

    # ---------- Gimbal 제어 ----------
    def _get_face_center_y(self, detection: Detection) -> Optional[float]:
        """얼굴 중심 Y좌표 추출 (코 또는 눈 위치 사용)"""
        if not detection.keypoints or not detection.keypoints.data:
            return detection.bbox.center.position.y - detection.bbox.size.y * 0.2
        
        keypoints = detection.keypoints.data
        
        face_y_points = []
        for kp in keypoints:
            if kp.id in [1, 2, 3] and kp.score > 0.5:  # nose, left_eye, right_eye
                face_y_points.append(kp.point.y)
        
        if face_y_points:
            return sum(face_y_points) / len(face_y_points)
        
        # 얼굴 키포인트 없으면 bbox 상단 사용
        return detection.bbox.center.position.y - detection.bbox.size.y * 0.2

    def _control_gimbal(self, target: Detection):
        """타겟의 얼굴 위치에 맞게 짐벌 피치 조정"""
        face_y = self._get_face_center_y(target)
        if face_y is None:
            return
        
        # 이미지 중심과의 Y 오차 (위로 가면 음수, 아래로 가면 양수)
        error_y = face_y - self.image_center_y + 100
        
        self.get_logger().info(f"얼굴Y={face_y:.0f}, 중심Y={self.image_center_y:.0f}, 오차={error_y:.0f}px")
        
        # 데드존 내면 조정 안 함
        if abs(error_y) < self.gimbal_deadzone:
            self.get_logger().info(f"짐벌 데드존 내 (오차 {error_y:.0f} < {self.gimbal_deadzone})")
            return
        
        target_pitch = error_y * self.gimbal_pitch_gain
        
        # 범위 제한 (-90 ~ +30)
        target_pitch = max(-90.0, min(30.0, target_pitch))
        
        self.get_logger().info(f"짐벌 피치: {target_pitch:.1f}° (오차 {error_y:.0f}px → 위쪽" if error_y < 0 else f"짐벌 피치: {target_pitch:.1f}° (오차 {error_y:.0f}px → 아래쪽)")
        
        if abs(target_pitch - self.current_gimbal_pitch) > 0.5:
            self.current_gimbal_pitch = target_pitch
            self._publish_gimbal(pitch=target_pitch)
            self.get_logger().warning(f">>> 짐벌 명령 전송: pitch={target_pitch:.1f}°")

    def _publish_gimbal(self, roll=0.0, pitch=0.0, yaw=0.0):
        """짐벌 명령 퍼블리시 (상대 각도 - 드론 기준)"""
        msg = GimbalCommand()
        msg.mode = 0  # position mode
        msg.frame = 1  # relative frame
        msg.roll = float(roll)
        msg.pitch = float(pitch)
        msg.yaw = 0.0  # yaw는 0으로 고정
        self.pub_gimbal.publish(msg)

    # ---------- 줌 제어 ----------
    def _set_zoom(self, zoom_level: float):
        return
        # """카메라 줌 설정"""
        # if abs(self.current_zoom - zoom_level) < 0.05:
        #     return  # 변화 없으면 무시
        
        # self.current_zoom = zoom_level
        # msg = CameraCommand()
        # msg.mode = 0  # level mode (절대값)
        # msg.zoom = float(zoom_level)
        # self.pub_camera.publish(msg)
        # self.get_logger().info(f"줌: {zoom_level:.1f}x")

    def _zoom_in_for_tracking(self):
        return
        # if not self.is_zoomed_in:
        #     self._set_zoom(self.tracking_zoom)
        #     self.is_zoomed_in = True

    def _zoom_out_to_default(self):
        return
        # if self.is_zoomed_in:
        #     self._set_zoom(self.default_zoom)
        #     self.is_zoomed_in = False

    def _on_moveby_done(self, msg: Bool):
        self.is_moving = False
        if self.move_start_time is not None:
            self.get_logger().info(f"Move time : {time.time() - self.move_start_time}")
        self.move_start_time = None
        
        # 초기 상승 완료 후 추적 시작
        if self.waiting_for_initial_ascent:
            if msg.data:
                self.get_logger().warning("상승 완료! 추적 시작 ✓")
                self.tracking_enabled = True
                self.waiting_for_initial_ascent = False
            else:
                self.get_logger().error("초기 상승 실패")
                self.tracking_enabled = True
                self.waiting_for_initial_ascent = False
            return

        if self.returning_home:
            # 복귀 완료
            self.returning_home = False
            self.total_dy = 0.0
            self.waiting_at_home = True
            self.get_logger().warning("시작 위치 복귀 완료! 정면 타깃 대기 중...")
            return

        if msg.data:
            self.get_logger().debug("이동 완료")
        else:
            self.get_logger().warn("이동 실패")

    # ---------- 서비스 헬퍼 ----------
    def _call_trigger(self, client, name: str):
        if not client.service_is_ready():
            self.get_logger().info(f"{name} 서비스 대기 중...")
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f"{name} 서비스 없음")
                return False

        fut = client.call_async(Trigger.Request())

        def _done(_):
            try:
                resp = fut.result()
                if resp and resp.success:
                    self.get_logger().info(f"{name}: {resp.message}")
                else:
                    self.get_logger().warn(f"{name} 실패")
            except Exception as e:
                self.get_logger().error(f"{name} 오류: {e}")

        fut.add_done_callback(_done)
        return True

    def _set_offboard(self, enable: bool):
        if not self.cli_offboard.service_is_ready():
            if not self.cli_offboard.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn("offboard 서비스 없음")
                return

        req = SetBool.Request()
        req.data = enable
        fut = self.cli_offboard.call_async(req)

        def _done(_):
            try:
                resp = fut.result()
                if resp and resp.success:
                    self.get_logger().info(f"Offboard: {'ON' if enable else 'OFF'}")
            except Exception:
                pass

        fut.add_done_callback(_done)

    # ---------- 키보드 ----------
    def _keyboard_tick(self):
        ch = self._kb.getch()
        if ch is None:
            return

        if ch == 't':
            self.get_logger().warning("=" * 30)
            self.get_logger().warning("이륙 요청...")
            self.get_logger().warning("=" * 30)
            self._set_offboard(True)

            # 이륙 성공 전까지 제어 비활성화
            self.is_flying = False
            self.tracking_enabled = False

            # takeoff 서비스 확인
            if not self.cli_takeoff.service_is_ready():
                self.get_logger().info("takeoff 서비스 대기 중...")
                if not self.cli_takeoff.wait_for_service(timeout_sec=5.0):
                    self.get_logger().error("takeoff 서비스 없음")
                    return

            req = Trigger.Request()
            fut = self.cli_takeoff.call_async(req)

            def _takeoff_done(_):
                try:
                    resp = fut.result()
                    if resp and resp.success:
                        self.get_logger().warning("=" * 30)
                        self.get_logger().warning("✅ 이륙 성공! +0.6m 상승 후 추적 시작")
                        self.get_logger().warning("=" * 30)
                        # 이륙 성공 후 is_flying만 활성화 (추적은 아직 비활성화)
                        self.is_flying = True
                        self.tracking_enabled = False  # moveby 완료 후 활성화
                        self.total_dy = 0.0
                        self.last_target_seen_time = time.time()
                        self.returning_home = False
                        self.waiting_at_home = False
                        self.tracking_id = None
                        self.current_gimbal_pitch = 0.0
                        self._publish_gimbal(pitch=0.0)
                        self.is_zoomed_in = False
                        # self._set_zoom(self.default_zoom)
                        
                        # 이륙 후 녹화 시작
                        if not self.is_recording:
                            self.is_recording = True
                            self._start_recording()
                        
                        # +0.6m 상승 명령 발행 및 대기 플래그 설정
                        
                        time.sleep(5)
                        self.get_logger().info("상승 명령 (dz=-0.6m) 발행")
                        self.waiting_for_initial_ascent = True
                        self._publish_moveby(dz=-0.6)
                    else:
                        self.get_logger().error(f"이륙 실패: {resp.message if resp else 'unknown error'}")
                        self.is_flying = False
                        self.tracking_enabled = False
                except Exception as e:
                    self.get_logger().error(f"이륙 오류: {e}")
                    self.is_flying = False
                    self.tracking_enabled = False

            fut.add_done_callback(_takeoff_done)

        elif ch == 'l':
            self.get_logger().warning("착륙 요청")
            self.tracking_enabled = False
            # self._zoom_out_to_default()  # 착륙 시 줌 아웃
            
            # 착륙 전 녹화 중단
            if self.is_recording:
                self.is_recording = False
                self._stop_recording()
            
            self._call_trigger(self.cli_land, 'land')
            self.is_flying = False
            self._download_media()

        elif ch == 'k':
            self.get_logger().error("긴급 정지!")
            self.tracking_enabled = False
            # self._zoom_out_to_default()  # 정지 시 줌 아웃
            self._call_trigger(self.cli_halt, 'halt')
            self.is_flying = False

        elif ch == ' ':
            self.tracking_enabled = not self.tracking_enabled
            status = "재개" if self.tracking_enabled else "일시정지"
            self.get_logger().warning(f"추적 {status}")

        elif ch == 'm':
            self.get_logger().info("수동으로 위로 이동")
            self._publish_moveby(dz=-0.2)
        
        elif ch == 'n':
            self.get_logger().info("수동으로 아래로 이동")
            self._publish_moveby(dz=0.2)

        # elif ch == 'r':
        #     self.is_recording = not self.is_recording
        #     if self.is_recording:
        #         self._start_recording()
        #     else:
        #         self._stop_recording()

        elif ch == 'd':
            self._download_media()


    # ---------- 녹화 제어 ----------
    def _start_recording(self):
        """녹화 시작"""
        if not self.cli_record_start.service_is_ready():
            self.get_logger().info("camera/recording/start 서비스 대기 중...")
            if not self.cli_record_start.wait_for_service(timeout_sec=3.0):
                self.get_logger().error("camera/recording/start 서비스 없음")
                self.is_recording = False
                return

        try:
            from anafi_ros_interfaces.srv import Recording
            req = Recording.Request()
            req.mode = 0         # standard
            req.resolution = 3   # 1920x1080 (Full HD)
            req.framerate = 2    # 30 fps
            req.hyperlapse = 1   # 1/30 (기본값)
            
            fut = self.cli_record_start.call_async(req)

            def _done(_):
                try:
                    resp = fut.result()
                    self.get_logger().warning("✅ 녹화 시작")
                except Exception as e:
                    self.get_logger().error(f"녹화 시작 오류: {e}")
                    self.is_recording = False

            fut.add_done_callback(_done)
        except Exception as e:
            self.get_logger().error(f"녹화 시작 요청 오류: {e}")
            self.is_recording = False

    def _stop_recording(self):
        """녹화 중단"""
        if not self.cli_record_stop.service_is_ready():
            self.get_logger().info("camera/recording/stop 서비스 대기 중...")
            if not self.cli_record_stop.wait_for_service(timeout_sec=3.0):
                self.get_logger().error("camera/recording/stop 서비스 없음")
                self.is_recording = True
                return

        try:
            from anafi_ros_interfaces.srv import Recording
            req = Recording.Request()
            
            fut = self.cli_record_stop.call_async(req)

            def _done(_):
                try:
                    resp = fut.result()
                    self.get_logger().warning("⏹️ 녹화 중단")
                except Exception as e:
                    self.get_logger().error(f"녹화 중단 오류: {e}")
                    self.is_recording = True

            fut.add_done_callback(_done)
        except Exception as e:
            self.get_logger().error(f"녹화 중단 요청 오류: {e}")
            self.is_recording = True

    # ---------- 미디어 다운로드 ----------
    def _download_media(self):
        """녹화된 미디어 파일 다운로드"""
        if not self.cli_download.service_is_ready():
            self.get_logger().info("다운로드 서비스 대기 중...")
            if not self.cli_download.wait_for_service(timeout_sec=3.0):
                self.get_logger().error("다운로드 서비스 없음")
                return

        try:
            fut = self.cli_download.call_async(SetBool.Request())

            def _done(_):
                try:
                    resp = fut.result()
                    if resp and resp.success:
                        self.get_logger().warning(f"✅ 다운로드 완료: {resp.message}")
                        self.get_logger().warning("파일 위치: ~/Pictures/Anafi")
                    else:
                        pass
                        # self.get_logger().error(f"다운로드 실패: {resp.message if resp else 'unknown error'}")
                except Exception as e:
                    self.get_logger().error(f"다운로드 오류: {e}")

            fut.add_done_callback(_done)
        except Exception as e:
            self.get_logger().error(f"다운로드 요청 오류: {e}")


def main():
    rclpy.init()
    node = PersonFollower()
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

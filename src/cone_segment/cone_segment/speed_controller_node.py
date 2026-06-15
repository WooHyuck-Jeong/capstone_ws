#!/usr/bin/env python3
"""
speed_controller_node.py
========================
cone_detector_node 의 /cone_pairs 토픽을 구독하여
PX4 Offboard velocity 제어 명령을 발행하는 노드.

데이터 흐름:
    /cone_pairs   (bearing_deg, distance_m, disappeared)
    /fmu/out/vehicle_local_position  (heading [rad, NED])
            ↓
    yaw_ned = heading + radians(bearing_deg)
    speed   = clip(distance_m × speed_k, u_min, u_max)
            ↓
    /fmu/in/offboard_control_mode   (velocity=True)
    /fmu/in/trajectory_setpoint     (velocity=[speed, 0, 0], yaw=yaw_ned)

게이트가 안 보이면 (pairs 없음 / disappeared > 0 / 토픽 타임아웃):
    velocity=False 로 publish → Atan2 guidance 복귀

선택 기준:
    disappeared == 0 인 쌍 중 midpoint.y 가 가장 큰 것 (= 가장 가까운 쌍)
"""

import math
import json
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition


class SpeedControllerNode(Node):

    # ── 토픽 상수 ─────────────────────────────────────────────────────────────
    TOPIC_CONE_PAIRS            = '/cone_pairs'
    TOPIC_LOCAL_POSITION        = '/fmu/out/vehicle_local_position'
    TOPIC_OFFBOARD_CONTROL_MODE = '/fmu/in/offboard_control_mode'
    TOPIC_TRAJECTORY_SETPOINT   = '/fmu/in/trajectory_setpoint'

    def __init__(self):
        super().__init__('speed_controller_node')

        # ── 파라미터 선언 ──────────────────────────────────────────────────────
        self.declare_parameter('timer_hz',       50.0)   # 제어 루프 주기 [Hz]
        self.declare_parameter('u_min',          1.5)    # 최소 전진 속도 [m/s]
        self.declare_parameter('u_max',          2.0)    # 최대 전진 속도 [m/s]
        self.declare_parameter('speed_k',        1.0)    # distance → speed 비례계수
        self.declare_parameter('cmd_timeout_sec', 1.0)   # /cone_pairs 타임아웃 [s]
        self.declare_parameter('use_distance',   True)   # False 면 항상 u_max 고정

        timer_hz              = self.get_parameter('timer_hz').value
        self.u_min            = self.get_parameter('u_min').value
        self.u_max            = self.get_parameter('u_max').value
        self.speed_k          = self.get_parameter('speed_k').value
        self.cmd_timeout_sec  = self.get_parameter('cmd_timeout_sec').value
        self.use_distance     = self.get_parameter('use_distance').value

        # ── QoS ───────────────────────────────────────────────────────────────
        px4_pub_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability  = DurabilityPolicy.TRANSIENT_LOCAL,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 1,
        )
        px4_sub_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability  = DurabilityPolicy.VOLATILE,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 1,
        )

        # ── 상태 변수 ─────────────────────────────────────────────────────────
        self.lock = Lock()

        # /fmu/out/vehicle_local_position
        self.heading        = 0.0    # NED heading [rad]
        self.heading_valid  = False

        # /cone_pairs 에서 파싱한 최근 유효 값
        self.bearing_deg    = 0.0
        self.distance_m     = None   # None 이면 distance 정보 없음
        self.gate_visible   = False  # disappeared==0 인 유효 쌍 존재 여부
        self.pairs_stamp_ns = 0      # 마지막 /cone_pairs 수신 시각

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            String,
            self.TOPIC_CONE_PAIRS,
            self._cone_pairs_callback,
            10,
        )
        self.create_subscription(
            VehicleLocalPosition,
            self.TOPIC_LOCAL_POSITION,
            self._local_position_callback,
            px4_sub_qos,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_offboard_mode = self.create_publisher(
            OffboardControlMode,
            self.TOPIC_OFFBOARD_CONTROL_MODE,
            px4_pub_qos,
        )
        self.pub_trajectory = self.create_publisher(
            TrajectorySetpoint,
            self.TOPIC_TRAJECTORY_SETPOINT,
            px4_pub_qos,
        )

        # ── 제어 루프 타이머 ──────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / timer_hz, self._timer_callback)

        self.get_logger().info(
            f'SpeedControllerNode started | '
            f'{timer_hz:.0f}Hz | '
            f'u=[{self.u_min}, {self.u_max}] m/s | '
            f'speed_k={self.speed_k} | '
            f'use_distance={self.use_distance}'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Subscriber callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def _cone_pairs_callback(self, msg: String):
        """/cone_pairs JSON 파싱 → 가장 가까운 유효 쌍 선택."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(
                '/cone_pairs 파싱 실패', throttle_duration_sec=5.0)
            return

        pairs = data.get('pairs', [])

        # disappeared == 0 인 쌍만 유효 후보로 사용
        active = [p for p in pairs if p.get('disappeared', 1) == 0]

        with self.lock:
            self.pairs_stamp_ns = self.get_clock().now().nanoseconds

            if not active:
                # 검출된 유효 쌍 없음
                self.gate_visible = False
                return

            # midpoint.y 가 가장 큰 쌍 = 화면 아래 = 가장 가까운 쌍
            nearest = max(active, key=lambda p: p['midpoint']['y'])

            self.bearing_deg  = float(nearest['bearing_deg'])
            self.distance_m   = (
                float(nearest['distance_m'])
                if nearest.get('distance_m') is not None
                else None
            )
            self.gate_visible = True

        self.get_logger().debug(
            f'[PAIRS] bearing={self.bearing_deg:.2f}deg  '
            f'dist={self.distance_m}m  '
            f'active={len(active)}/{len(pairs)}'
        )

    def _local_position_callback(self, msg: VehicleLocalPosition):
        """PX4 NED heading 수신."""
        with self.lock:
            if math.isfinite(msg.heading):
                self.heading       = float(msg.heading)
                self.heading_valid = True

    # ══════════════════════════════════════════════════════════════════════════
    # 제어 루프
    # ══════════════════════════════════════════════════════════════════════════

    def _timer_callback(self):
        now_ns = self.get_clock().now().nanoseconds
        now_us = int(now_ns / 1000)

        with self.lock:
            gate_visible   = self.gate_visible
            bearing_deg    = self.bearing_deg
            distance_m     = self.distance_m
            heading        = self.heading
            heading_valid  = self.heading_valid
            pairs_stamp_ns = self.pairs_stamp_ns

        # /cone_pairs 신선도 확인
        if pairs_stamp_ns > 0:
            age_sec   = (now_ns - pairs_stamp_ns) / 1e9
            pairs_fresh = age_sec <= self.cmd_timeout_sec
        else:
            pairs_fresh = False

        # 제어 활성화 조건: 유효 쌍 존재 + 토픽 신선 + heading 유효
        can_control = gate_visible and pairs_fresh and heading_valid

        if can_control:
            # ── 콘 추종 모드 ──────────────────────────────────────────────────
            yaw_ned = heading + math.radians(bearing_deg)
            yaw_ned = math.atan2(math.sin(yaw_ned), math.cos(yaw_ned))  # [-π, π] 정규화

            speed = self._compute_speed(distance_m)

            self._publish_offboard_mode(now_us, velocity=True)
            self._publish_trajectory(now_us, speed, yaw_ned)

            self.get_logger().info(
                f'[CTRL] bearing={bearing_deg:+.1f}deg | '
                f'dist={distance_m:.2f}m | '
                f'speed={speed:.2f}m/s | '
                f'yaw_ned={math.degrees(yaw_ned):.1f}deg',
                throttle_duration_sec=1.0,
            )
        else:
            # ── Fallback: velocity=False → Atan2 guidance 복귀 ───────────────
            self._publish_offboard_mode(now_us, velocity=False)

            reason = []
            if not gate_visible:
                reason.append('no valid cone pair')
            if not pairs_fresh:
                reason.append(f'topic stale ({(now_ns - pairs_stamp_ns)/1e9:.1f}s)')
            if not heading_valid:
                reason.append('heading invalid')

            self.get_logger().info(
                f'[FALLBACK] {", ".join(reason)} → velocity=False',
                throttle_duration_sec=2.0,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 속도 계산
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_speed(self, distance_m) -> float:
        """
        거리 기반 가변 속도 계산.

        use_distance=False 이거나 distance_m 이 None 이면 u_max 고정.
        그 외: speed = clip(distance_m × speed_k, u_min, u_max)
        """
        if not self.use_distance or distance_m is None:
            return self.u_max
        speed = distance_m * self.speed_k
        return max(self.u_min, min(self.u_max, speed))

    # ══════════════════════════════════════════════════════════════════════════
    # Publish helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _publish_offboard_mode(self, timestamp_us: int, velocity: bool):
        msg = OffboardControlMode()
        msg.timestamp    = timestamp_us
        msg.position     = False
        msg.velocity     = velocity
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self.pub_offboard_mode.publish(msg)

    def _publish_trajectory(self, timestamp_us: int, speed: float, yaw_ned: float):
        msg = TrajectorySetpoint()
        msg.timestamp = timestamp_us
        msg.position  = [float('nan'), float('nan'), float('nan')]
        msg.velocity  = [float(speed), 0.0, 0.0]
        msg.yaw       = float(yaw_ned)
        self.pub_trajectory.publish(msg)


# ══════════════════════════════════════════════════════════════════════════════
# 엔트리포인트
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SpeedControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
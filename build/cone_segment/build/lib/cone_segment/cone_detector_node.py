"""
cone_detector_node.py - torch_env Python으로 직접 실행되는 ROS2 노드
ROS2 패키지 경로를 sys.path에 직접 추가해서 rclpy 등을 사용 가능하게 함

[추가 기능]
- CameraOnlyConeTracker : Hungarian Algorithm 기반 꼬깔쌍 영구 ID 추적
  · ID는 단방향 증가, 절대 재사용 안 함
  · pair0가 사라져도 pair1 → pair0로 재번호 되지 않음
  · max_disappeared 프레임 동안 소실 허용 (카메라 각도 변경 대응)
  · bbox 픽셀거리 + bbox 크기(거리 근사) + HSV 색상 히스토그램 3중 비용으로 매칭
- Bearing angle 히스토리 기반 이동평균 스무싱 (이상값 억제)
"""

import sys
import os

# ── ROS2 jazzy Python 경로 추가 (Python 3.12) ────────────────────────────────
ROS2_PYTHON_PATHS = [
    '/opt/ros/jazzy/lib/python3.12/site-packages',
    '/opt/ros/jazzy/local/lib/python3.12/dist-packages',
]
for p in ROS2_PYTHON_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── capstone_ws install 경로 추가 ─────────────────────────────────────────────
WS_INSTALL = '/home/woohyuck/capstone_ws/install'
for pkg_dir in os.listdir(WS_INSTALL):
    lib_path = os.path.join(WS_INSTALL, pkg_dir, 'lib', 'python3.12', 'site-packages')
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

import cv2
import numpy as np
import math
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 구조
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Cone:
    index: int
    color: str
    cx: float
    cy: float
    bbox: Tuple
    confidence: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class ConePair:
    pair_index: int          # 영구 추적 ID (절대 재사용 안 함)
    red: Cone
    blue: Cone
    midpoint_x: float
    midpoint_y: float
    bearing_deg: float
    disappeared: int = 0     # 현재 소실 프레임 수


# ══════════════════════════════════════════════════════════════════════════════
# 색상 설정
# ══════════════════════════════════════════════════════════════════════════════

CLASS_COLOR_MAP = {"red_cone": "red", "blue_cone": "blue"}
MASK_COLORS  = {"red": (0, 50, 255), "blue": (255, 80, 0)}
BBOX_COLORS  = {"red": (0, 0, 220),  "blue": (220, 60, 0)}
MID_COLOR    = (0, 255, 255)
AXIS_COLOR   = (0, 255, 0)
LOST_COLOR   = (0, 165, 255)   # 소실 중 표시 색상 (주황)


# ══════════════════════════════════════════════════════════════════════════════
# TrackedPair : 추적기 내부 상태 객체
# ══════════════════════════════════════════════════════════════════════════════

class TrackedPair:
    """
    꼬깔쌍 하나의 추적 상태를 유지하는 클래스.
    EMA(지수이동평균)로 중심점·크기를 안정화하고,
    bearing 히스토리로 이상값을 억제한다.
    """

    # EMA 평활화 계수 (0에 가까울수록 부드럽고 느림)
    ALPHA = 0.4
    # bearing 이동평균 윈도우
    BEARING_WINDOW = 8

    def __init__(self, pair_id: int, mid_x: float, mid_y: float,
                 size: float, histogram, bearing: float):
        self.id            = pair_id
        self.mid_x         = mid_x
        self.mid_y         = mid_y
        self.size          = size          # bbox 넓이 (거리 근사)
        self.histogram     = histogram     # HSV 히스토그램
        self.bearing_history: List[float] = [bearing]
        self.smooth_bearing = bearing
        # 마지막으로 매칭된 Cone 정보 (시각화 및 publish 용)
        self.last_red:  Optional[Cone] = None
        self.last_blue: Optional[Cone] = None

    # ── 업데이트 ─────────────────────────────────────────────────────────────
    def update(self, mid_x: float, mid_y: float, size: float,
               histogram, bearing: float,
               red: Cone, blue: Cone):
        # [매칭 안정화용] 위치·크기만 EMA 평활화 → ID 매칭 비용 계산에 사용
        a = self.ALPHA
        self.mid_x = a * mid_x + (1 - a) * self.mid_x
        self.mid_y = a * mid_y + (1 - a) * self.mid_y
        self.size  = a * size  + (1 - a) * self.size
        if histogram is not None:
            self.histogram = histogram

        # [Bearing] raw 픽셀값 그대로 → 카메라 회전에 즉각 반응
        # 3-sigma 필터 제거: 카메라 각도 변경 시 큰 변화를 이상값으로 거부하지 않도록
        self.bearing_history.append(bearing)
        if len(self.bearing_history) > self.BEARING_WINDOW * 2:
            self.bearing_history.pop(0)
        # 경량 이동평균(3프레임)으로 노이즈만 제거, 반응속도는 유지
        recent = self.bearing_history[-3:]
        self.smooth_bearing = float(np.mean(recent))

        self.last_red  = red
        self.last_blue = blue


# ══════════════════════════════════════════════════════════════════════════════
# CameraOnlyConeTracker : Hungarian Algorithm 기반 영구 ID 추적기
# ══════════════════════════════════════════════════════════════════════════════

class CameraOnlyConeTracker:
    """
    카메라만 사용하는 꼬깔쌍 영구 추적기.

    매칭 비용 = bbox 크기(거리 근사, 45%) + HSV 색상(40%) + 픽셀거리(15%)
    카메라 헤딩이 바뀌면 픽셀 위치는 크게 달라지지만
    꼬깔의 크기(거리)와 색 조합은 유지되므로 이를 주 매칭 기준으로 사용.

    ID 규칙:
      - next_id 는 단방향 증가, 절대 재사용 안 함
      - pair0 가 화면에서 사라져도 pair1 은 절대 pair0 가 되지 않음
      - tracked + lost 전체를 매칭 대상으로 삼아 유령 pair 방지
      - 새 탐지는 기존 전체(tracked+lost)와 매칭 실패한 경우만 신규 등록
    """

    def __init__(self,
                 max_disappeared: int = 45,   # 소실 허용 프레임 (30fps → 1.5초)
                 max_cost: float = 150.0,      # 매칭 거부 비용 임계값 (낮출수록 엄격)
                 pixel_w: float = 0.15,        # 픽셀거리 가중치 (헤딩 변화 시 불안정)
                 size_w:  float = 0.45,        # bbox 크기 가중치 (거리 근사 → 안정)
                 color_w: float = 0.40):       # 색상 가중치 (red+blue 조합 → 안정)

        self.next_id      = 0
        self.tracked:     Dict[int, TrackedPair] = {}  # 활성 객체
        self.lost:        Dict[int, TrackedPair] = {}  # 소실 중 객체
        self.disappeared: Dict[int, int]          = {} # {id: 소실 프레임 수}

        self.max_disappeared = max_disappeared
        self.max_cost        = max_cost
        self.pixel_w         = pixel_w
        self.size_w          = size_w
        self.color_w         = color_w

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────
    def update(self, detections: List[dict], frame: np.ndarray
               ) -> Dict[int, TrackedPair]:
        """
        detections 원소: {
            'mid_x', 'mid_y', 'size', 'bearing',
            'red': Cone, 'blue': Cone,
            'bbox_union': [x1,y1,x2,y2]
        }
        """
        # 히스토그램 추출
        for det in detections:
            det['histogram'] = self._get_histogram(frame, det['bbox_union'])

        # ── 탐지 없음 → 전체 소실 카운트 ────────────────────────────────────
        if not detections:
            self._increment_disappeared(list(self.tracked.keys()))
            return {**self.tracked, **self.lost}

        # ── 기존 객체 전혀 없음 → 모두 신규 등록 ────────────────────────────
        all_pairs = {**self.tracked, **self.lost}
        if not all_pairs:
            for det in detections:
                self._register(det)
            return {**self.tracked, **self.lost}

        # ── Hungarian Matching: tracked + lost 전체를 대상으로 ───────────────
        matched, new_det_idxs, lost_obj_ids = self._match(detections, all_pairs)

        # 매칭 성공 → 업데이트 + lost이면 tracked로 복귀
        for det_idx, obj_id in matched:
            d  = detections[det_idx]
            tp = all_pairs[obj_id]
            tp.update(d['mid_x'], d['mid_y'], d['size'],
                      d['histogram'], d['bearing'],
                      d['red'], d['blue'])
            self.disappeared[obj_id] = 0
            # lost 버킷에 있었으면 tracked로 복귀
            if obj_id in self.lost:
                self.tracked[obj_id] = self.lost.pop(obj_id)

        # 매칭 실패한 기존 객체 → 소실 카운트
        self._increment_disappeared(lost_obj_ids)

        # 매칭 실패한 신규 탐지 → 신규 등록
        for det_idx in new_det_idxs:
            self._register(detections[det_idx])

        return {**self.tracked, **self.lost}

    # ── 내부 메서드 ───────────────────────────────────────────────────────────
    def _match(self, detections, all_pairs: Dict[int, 'TrackedPair']):
        """tracked + lost 전체를 매칭 대상으로 삼는다."""
        obj_ids = list(all_pairs.keys())
        n_obj   = len(obj_ids)
        n_det   = len(detections)

        cost = np.zeros((n_obj, n_det), dtype=np.float64)
        for i, oid in enumerate(obj_ids):
            for j, det in enumerate(detections):
                cost[i, j] = self._cost(all_pairs[oid], det)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched     = []
        unmatched_d = set(range(n_det))
        unmatched_o = set(range(n_obj))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.max_cost:
                continue
            matched.append((c, obj_ids[r]))
            unmatched_d.discard(c)
            unmatched_o.discard(r)

        lost_obj_ids = [obj_ids[i] for i in unmatched_o]
        return matched, list(unmatched_d), lost_obj_ids

    def _cost(self, tp: 'TrackedPair', det: dict) -> float:
        # 1. bbox 크기 비율 차이 (거리 근사 → 헤딩 무관하게 안정)
        size_ratio = abs(tp.size - det['size']) / (tp.size + 1e-6)
        size_cost  = min(size_ratio * 120.0, 120.0)   # 최대 120 cap

        # 2. HSV 히스토그램 거리 (red+blue 색 조합 → 헤딩 무관하게 안정)
        color_cost = 0.0
        if tp.histogram is not None and det['histogram'] is not None:
            color_cost = cv2.compareHist(
                tp.histogram, det['histogram'],
                cv2.HISTCMP_BHATTACHARYYA
            ) * 80.0

        # 3. 픽셀 중심 거리 (보조 — 헤딩 변화 시 불안정하므로 가중치 낮춤)
        pixel_dist = math.hypot(tp.mid_x - det['mid_x'],
                                tp.mid_y - det['mid_y'])

        return (size_cost  * self.size_w
                + color_cost * self.color_w
                + pixel_dist * self.pixel_w)

    def _get_histogram(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _register(self, det: dict):
        tp = TrackedPair(
            pair_id  = self.next_id,
            mid_x    = det['mid_x'],
            mid_y    = det['mid_y'],
            size     = det['size'],
            histogram= det.get('histogram'),
            bearing  = det['bearing']
        )
        tp.last_red  = det['red']
        tp.last_blue = det['blue']
        self.tracked[self.next_id]     = tp
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _increment_disappeared(self, obj_ids: List[int]):
        for oid in obj_ids:
            self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
            if self.disappeared[oid] > self.max_disappeared:
                # 완전 소멸 (ID 재사용 안 함)
                self.tracked.pop(oid, None)
                self.lost.pop(oid, None)
                self.disappeared.pop(oid, None)
            else:
                # 소실 → lost 버킷으로 이동
                if oid in self.tracked:
                    self.lost[oid] = self.tracked.pop(oid)


# ══════════════════════════════════════════════════════════════════════════════
# ROS2 노드
# ══════════════════════════════════════════════════════════════════════════════

class ConeDetectorNode(Node):

    def __init__(self):
        super().__init__('cone_detector_node')

        # ── 파라미터 선언 ──────────────────────────────────────────────────────
        self.declare_parameter('camera_index',   0)
        self.declare_parameter('model_path',     'best.pt')
        self.declare_parameter('hfov_deg',       46.0)
        self.declare_parameter('conf_thresh',    0.88)
        self.declare_parameter('min_area',       0.003)
        self.declare_parameter('show_image',     True)
        self.declare_parameter('img_width',      640)
        self.declare_parameter('img_height',     480)
        # 추적기 파라미터
        self.declare_parameter('max_disappeared', 45)   # 소실 허용 프레임
        self.declare_parameter('max_cost',        220.0) # 매칭 거부 임계값

        cam_idx            = self.get_parameter('camera_index').value
        model_path         = self.get_parameter('model_path').value
        hfov_deg           = self.get_parameter('hfov_deg').value
        self.conf_thresh   = self.get_parameter('conf_thresh').value
        self.min_area      = self.get_parameter('min_area').value
        self.show_image    = self.get_parameter('show_image').value
        self.img_w         = self.get_parameter('img_width').value
        self.img_h         = self.get_parameter('img_height').value
        max_disappeared    = self.get_parameter('max_disappeared').value
        max_cost           = self.get_parameter('max_cost').value

        # ── 카메라 핀홀 파라미터 ───────────────────────────────────────────────
        self.cx_p  = self.img_w / 2.0
        self.f_eq  = self.img_w / math.radians(hfov_deg)
        self.get_logger().info(f'Camera | hfov={hfov_deg}deg  f_eq={self.f_eq:.1f}')

        # ── YOLO ──────────────────────────────────────────────────────────────
        self.model = YOLO(model_path)
        self.get_logger().info(f'YOLO | {model_path}  classes={self.model.names}')

        # ── 추적기 ────────────────────────────────────────────────────────────
        self.tracker = CameraOnlyConeTracker(
            max_disappeared = max_disappeared,
            max_cost        = max_cost,
        )
        self.get_logger().info(
            f'Tracker | max_disappeared={max_disappeared}  max_cost={max_cost}'
        )

        # ── 카메라 캡처 ───────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera opened | {actual_w}x{actual_h}')

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_pairs   = self.create_publisher(String,            '/cone_pairs',   10)
        self.pub_bearing = self.create_publisher(Float32MultiArray, '/cone_bearing', 10)
        self.pub_image   = self.create_publisher(Image,             '/cone_image',   10)

        if self.show_image:
            cv2.namedWindow('Cone Detector', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Cone Detector', self.img_w, self.img_h)

        self._pair_colors: Dict[int, Tuple] = {}  # ID별 고정 색상

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('ConeDetectorNode started (with persistent tracking)')

    # ── 메인 콜백 ─────────────────────────────────────────────────────────────
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Frame read failed')
            return

        # 1. YOLO 탐지 → 꼬깔쌍 후보 생성
        det_pairs, all_cones = self._detect(frame)

        # 2. 추적기 업데이트 → 영구 ID 부여
        tracked = self.tracker.update(det_pairs, frame)

        # 3. TrackedPair → ConePair 변환 (publish / 시각화 공용)
        cone_pairs = self._build_cone_pairs(tracked)

        # 4. 시각화
        vis = self._draw(frame.copy(), cone_pairs, all_cones)
        self._publish(cone_pairs, vis)

        if self.show_image:
            cv2.imshow('Cone Detector', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

    # ── YOLO 탐지 ─────────────────────────────────────────────────────────────
    def _detect(self, frame):
        """
        Returns
        -------
        det_pairs : tracker.update() 에 전달할 후보 리스트
        all_cones : 시각화용 {'red': [...], 'blue': [...]}
        """
        results  = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        red_cones, blue_cones = [], []
        img_h, img_w = frame.shape[:2]
        img_area = img_h * img_w

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()
            for i, box in enumerate(results.boxes):
                cls_name = self.model.names[int(box.cls[0])]
                color    = CLASS_COLOR_MAP.get(cls_name)
                if color is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area / img_area < self.min_area:
                    continue
                mask_full = cv2.resize(masks_data[i], (img_w, img_h))
                cone = Cone(
                    index=-1, color=color,
                    cx=(x1+x2)/2.0, cy=(y1+y2)/2.0,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    mask=mask_full
                )
                (red_cones if color == 'red' else blue_cones).append(cone)

        for lst in (red_cones, blue_cones):
            lst.sort(key=lambda c: c.cx)
            for idx, c in enumerate(lst):
                c.index = idx

        # ── 꼬깔쌍 매칭 (기존 로직 유지) ─────────────────────────────────────
        raw_pairs = self._match_raw_pairs(red_cones, blue_cones)

        # ── tracker 입력 형태로 변환 ──────────────────────────────────────────
        det_pairs = []
        for r, b in raw_pairs:
            mid_x   = (r.cx + b.cx) / 2.0
            mid_y   = (r.cy + b.cy) / 2.0
            bearing = math.degrees((mid_x - self.cx_p) / self.f_eq)
            # 합친 bbox (히스토그램용)
            x1 = min(r.bbox[0], b.bbox[0]); y1 = min(r.bbox[1], b.bbox[1])
            x2 = max(r.bbox[2], b.bbox[2]); y2 = max(r.bbox[3], b.bbox[3])
            size = (x2 - x1) * (y2 - y1)
            det_pairs.append({
                'mid_x':     mid_x,
                'mid_y':     mid_y,
                'size':      size,
                'bearing':   bearing,
                'red':       r,
                'blue':      b,
                'bbox_union': [x1, y1, x2, y2],
                'histogram': None,   # tracker.update() 내에서 채움
            })

        return det_pairs, {'red': red_cones, 'blue': blue_cones}

    def _match_raw_pairs(self, reds, blues):
        """기존 greedy 매칭 → (red, blue) 튜플 리스트 반환"""
        if not reds or not blues:
            return []
        used_blue = set()
        matched   = []
        for r in reds:
            best_dist = float('inf')
            best_b    = None
            for j, b in enumerate(blues):
                if j in used_blue:
                    continue
                d = math.hypot(r.cx - b.cx, r.cy - b.cy)
                if d < best_dist:
                    best_dist = d
                    best_b    = (j, b)
            if best_b is not None:
                used_blue.add(best_b[0])
                matched.append((r, best_b[1]))
        return matched

    # ── TrackedPair → ConePair 변환 ───────────────────────────────────────────
    def _build_cone_pairs(self, tracked: Dict[int, TrackedPair]) -> List[ConePair]:
        pairs = []
        for tp in tracked.values():
            if tp.last_red is None or tp.last_blue is None:
                continue

            # ── midpoint · bearing 은 현재 프레임 raw 픽셀 기준 ──────────────
            # EMA mid_x/mid_y 는 Hungarian 매칭 비용 계산에만 쓰이고,
            # 화면 표시·publish 에는 실제 꼬깔 픽셀 위치를 그대로 사용한다.
            raw_mid_x = (tp.last_red.cx + tp.last_blue.cx) / 2.0
            raw_mid_y = (tp.last_red.cy + tp.last_blue.cy) / 2.0
            # bearing 도 현재 raw midpoint 로 계산 → 카메라 각도에 즉각 반응
            raw_bearing = math.degrees((raw_mid_x - self.cx_p) / self.f_eq)

            cp = ConePair(
                pair_index = tp.id,
                red        = tp.last_red,
                blue       = tp.last_blue,
                midpoint_x = raw_mid_x,
                midpoint_y = raw_mid_y,
                bearing_deg= tp.smooth_bearing,   # 3프레임 평균으로 미세 노이즈만 제거
                disappeared= self.tracker.disappeared.get(tp.id, 0),
            )
            # smooth_bearing 은 TrackedPair.update() 에서 raw_bearing 기반으로
            # 이미 갱신됐으므로 별도 재계산 불필요
            # (raw_bearing 변수는 디버깅·로그 용도로 유지)
            _ = raw_bearing
            pairs.append(cp)

        # 화면 아래(가까운 쪽)부터 정렬해서 표시
        pairs.sort(key=lambda p: p.midpoint_y, reverse=True)
        return pairs

    # ── Publish ───────────────────────────────────────────────────────────────
    def _publish(self, pairs: List[ConePair], vis: np.ndarray):
        # /cone_pairs (JSON) ── pair_index가 영구 ID
        pairs_data = {
            'pairs': [{
                'pair_index':  p.pair_index,
                'disappeared': p.disappeared,
                'red':  {'index': p.red.index,  'cx': round(p.red.cx,  1),
                         'cy': round(p.red.cy,  1), 'conf': round(p.red.confidence,  3)},
                'blue': {'index': p.blue.index, 'cx': round(p.blue.cx, 1),
                         'cy': round(p.blue.cy, 1), 'conf': round(p.blue.confidence, 3)},
                'midpoint':    {'x': round(p.midpoint_x, 1), 'y': round(p.midpoint_y, 1)},
                'bearing_deg': round(p.bearing_deg, 3)
            } for p in pairs],
            'timestamp': time.time()
        }
        msg_str      = String()
        msg_str.data = json.dumps(pairs_data)
        self.pub_pairs.publish(msg_str)

        # /cone_bearing ── [pair_id, bearing, pair_id, bearing, ...] 형식
        bearing_flat = []
        for p in pairs:
            bearing_flat.extend([float(p.pair_index), float(p.bearing_deg)])
        msg_bearing      = Float32MultiArray()
        msg_bearing.data = bearing_flat
        self.pub_bearing.publish(msg_bearing)

        # /cone_image
        img_msg                 = Image()
        img_msg.header.stamp    = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera'
        img_msg.height          = vis.shape[0]
        img_msg.width           = vis.shape[1]
        img_msg.encoding        = 'bgr8'
        img_msg.is_bigendian    = 0
        img_msg.step            = vis.shape[1] * 3
        img_msg.data            = vis.tobytes()
        self.pub_image.publish(img_msg)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    def _get_pair_color(self, pair_id: int) -> Tuple:
        """ID마다 고정 BGR 색상 반환 (매 프레임 일관성 유지)"""
        if pair_id not in self._pair_colors:
            rng = np.random.default_rng(pair_id * 7 + 13)
            self._pair_colors[pair_id] = tuple(
                int(v) for v in rng.integers(60, 240, 3)
            )
        return self._pair_colors[pair_id]

    def _draw(self, vis: np.ndarray,
              pairs: List[ConePair],
              all_cones: dict) -> np.ndarray:
        cx_img = self.img_w / 2

        # 마스크 오버레이
        for color, cones in all_cones.items():
            bgr = MASK_COLORS[color]
            for cone in cones:
                if cone.mask is None:
                    continue
                overlay = vis.copy()
                overlay[cone.mask > 0.5] = bgr
                vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

        # 개별 꼬깔 bbox
        for color, cones in all_cones.items():
            b_col = BBOX_COLORS[color]
            for cone in cones:
                x1, y1, x2, y2 = cone.bbox
                cx, cy = int(cone.cx), int(cone.cy)
                cv2.rectangle(vis, (x1, y1), (x2, y2), b_col, 2)
                cv2.circle(vis, (cx, cy), 5, b_col, -1)
                cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 1)
                label = f"{color[0].upper()}{cone.index} {cone.confidence:.2f}"
                cv2.putText(vis, label, (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_col, 2)

        # 중심축
        cv2.line(vis, (int(cx_img), 0), (int(cx_img), vis.shape[0]),
                 AXIS_COLOR, 1, cv2.LINE_AA)

        # 꼬깔쌍 시각화 ── 영구 ID 색상 사용
        for pair in pairs:
            p_color  = self._get_pair_color(pair.pair_index)
            is_lost  = pair.disappeared > 0

            rx, ry = int(pair.red.cx),  int(pair.red.cy)
            bx, by = int(pair.blue.cx), int(pair.blue.cy)
            mx, my = int(pair.midpoint_x), int(pair.midpoint_y)

            line_color = LOST_COLOR if is_lost else p_color
            cv2.line(vis, (rx, ry), (bx, by), line_color, 2, cv2.LINE_AA)
            cv2.circle(vis, (mx, my), 8,  line_color, -1)
            cv2.circle(vis, (mx, my), 10, (255, 255, 255), 1)

            d   = 'R' if pair.bearing_deg > 0 else ('L' if pair.bearing_deg < 0 else 'C')
            # 소실 중이면 [LOST Nf] 표시
            lost_tag = f" [LOST {pair.disappeared}f]" if is_lost else ""
            txt = f"Pair{pair.pair_index}: {pair.bearing_deg:+.1f}deg {d}{lost_tag}"

            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx = mx - tw // 2
            ty = max(my - 20, th + 8)
            cv2.rectangle(vis, (tx - 4, ty - th - 4),
                          (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(vis, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)

        # 좌상단 상태 요약
        y_off = 20
        for pair in pairs:
            lost_tag = f" | LOST:{pair.disappeared}f" if pair.disappeared > 0 else ""
            s = (f"[Pair{pair.pair_index}] "
                 f"R({pair.red.cx:.0f},{pair.red.cy:.0f}) "
                 f"<-> B({pair.blue.cx:.0f},{pair.blue.cy:.0f}) "
                 f"| bearing={pair.bearing_deg:+.2f}deg{lost_tag}")
            cv2.putText(vis, s, (8, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_off += 18

        # 우상단 추적기 통계
        total   = len(self.tracker.tracked) + len(self.tracker.lost)
        stat_txt = f"Tracking: {len(self.tracker.tracked)} active / {len(self.tracker.lost)} lost | total IDs: {self.tracker.next_id}"
        cv2.putText(vis, stat_txt, (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return vis

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# 엔트리포인트
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

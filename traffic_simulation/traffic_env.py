import pygame
import numpy as np
import json
import math
import random
from typing import Tuple, Optional, List

# ============================================================
# 상수
# ============================================================
SENSOR_ANGLES = [i * (math.pi / 4) for i in range(8)]  # 0°,45°,90°,...,315° (차량 헤딩 기준 오프셋)
FORWARD_SENSOR_IDX = [0, 1, 7]                           # 전방 센서: 정면(0), 우전방(1), 좌전방(7)
TL_ENCODE = {'green': 1.0, 'yellow': 0.5, 'red': -1.0}  # 신호등 인코딩

# state 크기: 센서8 + 신호등1 + 도로정보3 + 속도2 + 헤딩2 = 16
STATE_SIZE = 16


# ============================================================
# Car
# ============================================================
class Car:
    """
    실제같은 차량 물리 시뮬레이션
    - 가속/브레이크
    - 조향 (속도에 따른 민감도)
    - 드리프트
    - 마찰과 관성
    - 속도에 비례한 동적 friction
    """
    def __init__(self, x, y, angle=0, car_info=None):
        self.x = x
        self.y = y
        self.angle = angle

        self.velocity_x = 0
        self.velocity_y = 0
        self.speed = 0

        self.max_speed          = car_info['max_speed']
        self.acceleration_force = car_info['acceleration_force']
        self.brake_force        = car_info.get('brake_force', 400)

        self.base_friction         = car_info['friction']
        self.base_lateral_friction = car_info['lateral_friction']
        self.base_drift_friction   = car_info['drift_friction']
        self.base_brake_factor     = car_info['brake_factor']

        self.friction         = self.base_friction
        self.lateral_friction = self.base_lateral_friction
        self.drift_friction   = self.base_drift_friction
        self.brake_factor     = self.base_brake_factor

        self.turn_speed   = car_info['turn_speed']
        self.sensor_range = car_info['sensor_range']

        self._original_lateral_friction = car_info['lateral_friction']

        self.width  = car_info.get('width',  10)
        self.length = car_info.get('length', 20)

        self.is_drifting = False
        self.drift_factor = 0

        self.speed_friction_reduction       = car_info.get('speed_friction_reduction',       0.05)
        self.drift_extra_friction_reduction = car_info.get('drift_extra_friction_reduction', 0.08)
        self.brake_speed_penalty            = car_info.get('brake_speed_penalty',            0.15)

    def _calculate_speed_ratio(self):
        return min(self.speed / self.max_speed, 1.0) if self.max_speed > 0 else 0.0

    def _update_dynamic_friction(self, is_drifting=False):
        speed_ratio = self._calculate_speed_ratio()
        speed_friction_adjustment = speed_ratio * self.speed_friction_reduction
        self.friction = self.base_friction + (1.0 - self.base_friction) * speed_friction_adjustment
        if is_drifting:
            drift_adjustment = speed_ratio * self.drift_extra_friction_reduction
            self.lateral_friction = self.base_drift_friction * (1.0 - drift_adjustment)
            self.friction = self.friction + (1.0 - self.friction) * drift_adjustment
        else:
            lateral_adjustment = speed_ratio * self.speed_friction_reduction * 0.5
            self.lateral_friction = self._original_lateral_friction + (1.0 - self._original_lateral_friction) * lateral_adjustment

    def _calculate_dynamic_brake_factor(self):
        speed_ratio = self._calculate_speed_ratio()
        speed_penalty = speed_ratio * speed_ratio * self.brake_speed_penalty
        return min(self.base_brake_factor + (1.0 - self.base_brake_factor) * speed_penalty, 0.99)

    def update(self, dt, controls):
        self.speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        is_drifting = controls.get('brake', False) and (controls.get('left', False) or controls.get('right', False))
        self._update_dynamic_friction(is_drifting)

        forward_dir_x = math.cos(self.angle)
        forward_dir_y = math.sin(self.angle)

        if controls.get('forward', False):
            self.velocity_x += forward_dir_x * self.acceleration_force * dt
            self.velocity_y += forward_dir_y * self.acceleration_force * dt

        if controls.get('backward', False):
            self.velocity_x -= forward_dir_x * self.acceleration_force * dt * 0.5
            self.velocity_y -= forward_dir_y * self.acceleration_force * dt * 0.5

        if self.speed > 1:
            turn_factor = min(self.speed / 100, 1.0)
            if controls.get('left',  False): self.angle -= self.turn_speed * turn_factor * dt
            if controls.get('right', False): self.angle += self.turn_speed * turn_factor * dt

        if controls.get('brake', False):
            self.is_drifting = True
            dynamic_brake_factor = self._calculate_dynamic_brake_factor()
            if is_drifting:
                speed_ratio = self._calculate_speed_ratio()
                dynamic_brake_factor = min(dynamic_brake_factor + speed_ratio * 0.03, 0.99)
            self.velocity_x *= dynamic_brake_factor
            self.velocity_y *= dynamic_brake_factor
        else:
            self.is_drifting = False

        if self.speed > 10:
            move_angle = math.atan2(self.velocity_y, self.velocity_x)
            angle_diff = move_angle - self.angle
            lateral_velocity_x = -math.sin(self.angle) * self.speed * math.sin(angle_diff)
            lateral_velocity_y =  math.cos(self.angle) * self.speed * math.sin(angle_diff)
            self.velocity_x -= lateral_velocity_x * (1 - self.lateral_friction)
            self.velocity_y -= lateral_velocity_y * (1 - self.lateral_friction)

        self.velocity_x *= self.friction
        self.velocity_y *= self.friction

        current_speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if current_speed > self.max_speed:
            scale = self.max_speed / current_speed
            self.velocity_x *= scale
            self.velocity_y *= scale

        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        self.speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)

    def get_corners(self):
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        hw, hl = self.width / 2, self.length / 2
        corners = [(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)]
        return [(self.x + dx*cos_a - dy*sin_a, self.y + dx*sin_a + dy*cos_a) for dx,dy in corners]

    def draw(self, surface, camera_offset=(0, 0)):
        corners = self.get_corners()
        screen_corners = [(x-camera_offset[0], y-camera_offset[1]) for x,y in corners]
        color = (255, 100, 100) if self.is_drifting else (0, 100, 255)
        pygame.draw.polygon(surface, color, screen_corners)
        pygame.draw.polygon(surface, (0,0,0), screen_corners, 2)

        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        for wx, wy in [(-self.length/2+8,-self.width/2+3),(self.length/2-8,-self.width/2+3),
                       (self.length/2-8, self.width/2-3),(-self.length/2+8, self.width/2-3)]:
            wheel_x = self.x + wx*cos_a - wy*sin_a
            wheel_y = self.y + wx*sin_a + wy*cos_a
            wc, ws = math.cos(self.angle), math.sin(self.angle)
            hl, hw = 4, 2
            rotated = [(wheel_x+dx*wc-dy*ws-camera_offset[0],
                        wheel_y+dx*ws+dy*wc-camera_offset[1])
                       for dx,dy in [(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)]]
            pygame.draw.polygon(surface, (30,30,30), rotated)

        front_x = self.x + math.cos(self.angle)*self.length/2
        front_y = self.y + math.sin(self.angle)*self.length/2
        pygame.draw.line(surface, (255,255,0),
                         (self.x-camera_offset[0], self.y-camera_offset[1]),
                         (front_x-camera_offset[0], front_y-camera_offset[1]), 3)


# ============================================================
# RacingGame
# ============================================================
class RacingGame:
    """
    2D 탑뷰 레이싱 게임
    - 커스텀 트랙 로드 (track_data.json / track.json 호환)
    - 실제같은 차량 물리
    - 충돌 감지
    - 강화학습 인터페이스 + get_state()
    - headless 모드 지원
    """

    def __init__(self, track_file="track_data.json", width=1200, height=800,
                 car_json_path="racing_car.json", headless=False):
        self.headless = headless

        if headless:
            pygame.init()
            self.screen = pygame.Surface((width, height))
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("2D 레이싱 게임")

        self.width  = width
        self.height = height
        self.clock  = pygame.time.Clock()
        self.fps    = 60
        self.car_json = json.load(open(car_json_path, 'r', encoding='utf-8'))

        self.WHITE = (255, 255, 255)
        self.BLACK = (0,   0,   0)
        self.GREEN = (0,   200, 0)
        self.RED   = (200, 0,   0)
        self.GRAY  = (100, 100, 100)

        # 트랙 로드 (direction_grid, lane_data, traffic_lights 포함)
        self.track_surface  = None
        self.track_mask     = None
        self.direction_grid = {}
        self.lane_data      = []
        self.traffic_lights = []
        self.lane_segments: List[Tuple] = []   # 레이캐스팅용 차선 세그먼트 목록
        self.start_positions = []
        self.start_pos       = None
        self.end_pos         = None
        self.checkpoints     = []

        # 순차 신호등 상태
        self.tl_green_ms  = 7000
        self.tl_yellow_ms = 3000
        self.tl_seq_idx   = 0
        self.tl_seq_timer = 0
        self.tl_seq_phase = 'green'

        self._load_track(track_file)

        # car_json 의 start_point 번호로 시작 위치 선택
        sp_idx = self.car_json.get('start_point', 0)
        if self.start_positions:
            sp_idx = sp_idx if sp_idx < len(self.start_positions) else 0
            self.start_pos = self.start_positions[sp_idx]
        sp = self.start_pos or [width//2, height//2]
        self.car = Car(sp[0], sp[1], angle=0, car_info=self.car_json)

        self.camera_x = 0
        self.camera_y = 0
        self.camera_smooth = 0.1

        self.collision         = False
        self.goal_reached      = False
        self.total_distance    = 0
        self.start_time        = pygame.time.get_ticks()
        self.end_time          = None
        self.current_time      = None
        self.checkpoints_reached = []

        if not headless:
            try:
                self.font     = pygame.font.SysFont('arial,sans-serif', 24)
                self.big_font = pygame.font.SysFont('arial,sans-serif', 48)
            except:
                self.font     = pygame.font.Font(None, 28)
                self.big_font = pygame.font.Font(None, 48)
        else:
            self.font = self.big_font = None

    # ----------------------------------------------------------
    # 트랙 로드
    # ----------------------------------------------------------
    def _load_track(self, filename):
        """track_data.json (신버전) 및 track.json (구버전) 모두 지원"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # ── 구버전: track_mask 픽셀 데이터 ──────────────────
            if 'track_mask' in data:
                track_mask = np.array(data['track_mask'], dtype=np.uint8)
                th, tw = track_mask.shape
                surf = pygame.Surface((tw, th))
                surf.fill(self.WHITE)
                for y in range(th):
                    for x in range(tw):
                        if track_mask[y, x] > 0:
                            surf.set_at((x, y), self.GRAY)
                self.track_surface = surf
                self.track_mask    = track_mask

            # ── 신버전: direction_grid + lane_data ───────────────
            self.direction_grid = data.get('direction_grid', {})
            self.lane_data      = data.get('lane_data', [])

            # lane_data 가 있으면 트랙 서피스 + 마스크 재생성
            if self.lane_data:
                self.track_surface, self.track_mask = self._build_track_from_lane_data()

            # ── 신호등 로드 (timer 필드 추가) ────────────────────
            # 순차 신호등 전역 duration 로드
            self.tl_green_ms  = data.get('tl_green_ms',  7000)
            self.tl_yellow_ms = data.get('tl_yellow_ms', 3000)

            self.traffic_lights = []
            for i, tl in enumerate(data.get('traffic_lights', [])):
                entry = {
                    'pos':   tuple(tl['pos']),
                    'state': 'green' if i == 0 else 'red',
                }
                if 'dir' in tl:
                    entry['dir'] = tl['dir']
                self.traffic_lights.append(entry)

            # 다중 시작 위치 로드 (신버전: start_positions, 구버전: start_pos 호환)
            if 'start_positions' in data:
                self.start_positions = data['start_positions']
            elif 'start_pos' in data and data['start_pos'] is not None:
                self.start_positions = [data['start_pos']]
            else:
                self.start_positions = []

            self.end_pos     = data.get('end_pos')
            self.checkpoints = data.get('checkpoints', [])

            # 레이캐스팅용 차선 세그먼트 빌드
            self.lane_segments = self._build_lane_segments()

            if not self.headless:
                print(f"Track loaded: {filename}  |  "
                      f"grid_cells={len(self.direction_grid)}  "
                      f"lane_segs={len(self.lane_segments)}  "
                      f"traffic_lights={len(self.traffic_lights)}")

        except Exception as e:
            print(f"Track load failed: {e}")
            self.track_surface = pygame.Surface((self.width, self.height))
            self.track_surface.fill(self.WHITE)
            pygame.draw.rect(self.track_surface, self.GRAY, (200, 200, 800, 400))
            self.track_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            self.track_mask[200:600, 200:1000] = 255

    def _build_track_from_lane_data(self):
        """lane_data의 left/right 경계로 track_surface + track_mask 생성"""
        surf = pygame.Surface((self.width, self.height))
        surf.fill(self.WHITE)
        for ld in self.lane_data:
            left  = ld['left_lane']
            right = ld['right_lane']
            if len(left) < 2 or len(right) < 2:
                continue
            for i in range(len(left) - 1):
                poly = [
                    (int(left[i][0]),   int(left[i][1])),
                    (int(left[i+1][0]), int(left[i+1][1])),
                    (int(right[i+1][0]),int(right[i+1][1])),
                    (int(right[i][0]),  int(right[i][1])),
                ]
                pygame.draw.polygon(surf, self.GRAY, poly)
        arr  = pygame.surfarray.array3d(surf)
        gray = np.mean(arr, axis=2).T
        mask = (gray < 240).astype(np.uint8) * 255
        return surf, mask

    # ----------------------------------------------------------
    # 레이캐스팅용 차선 세그먼트 빌드
    # ----------------------------------------------------------
    def _build_lane_segments(self) -> List[Tuple]:
        """
        lane_data의 left_lane, center_lane, right_lane을
        ((x1,y1),(x2,y2)) 형태의 세그먼트 리스트로 변환.
        """
        segs = []
        for ld in self.lane_data:
            for key in ('left_lane', 'center_lane', 'right_lane'):
                pts = ld[key]
                for i in range(len(pts) - 1):
                    segs.append(((pts[i][0],   pts[i][1]),
                                 (pts[i+1][0], pts[i+1][1])))
        return segs

    # ----------------------------------------------------------
    # 신호등 업데이트
    # ----------------------------------------------------------
    def _update_traffic_lights(self, dt: float):
        """dt: 밀리초. 순차 신호등: 0번→1번→... 순으로 초록→노랑→(다음 번호 초록)"""
        n = len(self.traffic_lights)
        if n == 0:
            return

        self.tl_seq_timer += dt
        duration = self.tl_green_ms if self.tl_seq_phase == 'green' else self.tl_yellow_ms

        if self.tl_seq_timer >= duration:
            self.tl_seq_timer -= duration
            if self.tl_seq_phase == 'green':
                self.tl_seq_phase = 'yellow'
            else:
                self.tl_seq_idx   = (self.tl_seq_idx + 1) % n
                self.tl_seq_phase = 'green'

        for i, tl in enumerate(self.traffic_lights):
            tl['state'] = self.tl_seq_phase if i == self.tl_seq_idx else 'red'

    # ----------------------------------------------------------
    # ── STATE 생성 관련 함수들 ──────────────────────────────
    # ----------------------------------------------------------

    @staticmethod
    def _ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by) -> Optional[float]:
        """
        반직선 (ox,oy) + t*(dx,dy) 와 선분 (ax,ay)-(bx,by) 의 교점까지 거리 t 반환.
        교점이 없으면 None.
        """
        sx, sy = bx - ax, by - ay
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None
        t = ((ax - ox) * sy - (ay - oy) * sx) / denom
        s = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t >= 0 and 0.0 <= s <= 1.0:
            return t
        return None

    def _cast_ray(self, ox: float, oy: float, angle: float) -> float:
        """
        (ox, oy) 에서 angle 방향으로 레이를 쏘아 가장 가까운 차선까지의
        정규화된 거리 [0, 1] 반환 (닿지 않으면 1.0).
        """
        max_dist = self.car.sensor_range
        dx, dy   = math.cos(angle), math.sin(angle)
        min_t    = max_dist

        for (p1, p2) in self.lane_segments:
            t = self._ray_segment_intersect(ox, oy, dx, dy,
                                            p1[0], p1[1], p2[0], p2[1])
            if t is not None and t < min_t:
                min_t = t

        return min_t / max_dist  # 정규화

    def _get_sensor_data(self) -> List[float]:
        """
        차량 헤딩 기준 8방향 레이캐스트.
        각 센서: 가장 가까운 차선(left/center/right)까지 정규화 거리 [0, 1].

        반환: 길이 8 리스트
        """
        ox, oy = self.car.x, self.car.y
        heading = self.car.angle
        return [self._cast_ray(ox, oy, heading + offset) for offset in SENSOR_ANGLES]

    def _get_traffic_light_state(self) -> float:
        """
        전방 센서(인덱스 0,1,7) 범위 안에 신호등이 있으면 그 상태를 인코딩해 반환.
        - 없음    : 0.0
        - 초록    : 1.0
        - 노란    : 0.5
        - 빨강    : -1.0

        여러 신호등이 있으면 가장 가까운 것을 사용.
        반환: float (1개 값)
        """
        ox, oy     = self.car.x, self.car.y
        heading    = self.car.angle
        max_dist   = self.car.sensor_range

        # 전방 센서들의 각도 범위
        fwd_angles = [heading + SENSOR_ANGLES[i] for i in FORWARD_SENSOR_IDX]
        half_span  = math.pi / 8  # ±22.5° 허용

        nearest_dist = max_dist
        nearest_tl   = None

        for tl in self.traffic_lights:
            tx, ty = tl['pos']
            dx, dy = tx - ox, ty - oy
            dist   = math.hypot(dx, dy)
            if dist > max_dist:
                continue

            angle_to_tl = math.atan2(dy, dx)
            for fwd_a in fwd_angles:
                diff = angle_to_tl - fwd_a
                # [-π, π] 정규화
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) <= half_span and dist < nearest_dist:
                    nearest_dist = dist
                    nearest_tl   = tl
                    break

        if nearest_tl is None:
            return 0.0
        return TL_ENCODE.get(nearest_tl['state'], 0.0)

    def _get_traffic_light_info(self) -> tuple:
        """
        전방 센서 범위 내 가장 가까운 신호등 정보 반환.
        신호등의 dir 벡터와 차량 진행 방향이 일치하거나 우회전 중인 경우만 감지.

        반환: (tl_exists: int, tl_state: int, right_turnable: int)
          tl_exists     : 0 = 없음, 1 = 있음
          tl_state      : 0 = 빨강, 1 = 노랑, 2 = 초록
          right_turnable: 0 = 직진/좌회전, 1 = 우회전 중 (빨강이라도 통과 가능)
        """
        ox, oy   = self.car.x, self.car.y
        heading  = self.car.angle
        max_dist = self.car.sensor_range
        fwd_angles = [heading + SENSOR_ANGLES[i] for i in FORWARD_SENSOR_IDX]
        half_span  = math.pi / 8

        # 차량 진행 방향 단위벡터 (정지 시 heading 기준)
        spd = self.car.speed
        if spd > 1:
            car_dx = self.car.velocity_x / spd
            car_dy = self.car.velocity_y / spd
        else:
            car_dx = math.cos(heading)
            car_dy = math.sin(heading)

        nearest_dist     = max_dist
        nearest_tl       = None
        nearest_is_right = False

        for tl in self.traffic_lights:
            tx, ty = tl['pos']
            dx, dy = tx - ox, ty - oy
            dist   = math.hypot(dx, dy)
            if dist > max_dist:
                continue

            tl_dir = tl.get('dir')
            is_right_turn = False
            if tl_dir is not None:
                # 직진 정렬 (신호등 방향과 차량 진행 방향의 일치도)
                fwd_align = car_dx * tl_dir[0] + car_dy * tl_dir[1]

                # 우회전 방향: 스크린 좌표에서 시계방향 90° = [-dy, dx]
                right_dx = -tl_dir[1]
                right_dy =  tl_dir[0]
                right_align = car_dx * right_dx + car_dy * right_dy
                is_right_turn = right_align > 0.7

                # 직진도 아니고 우회전도 아니면 이 신호등은 무시
                if fwd_align < 0.3 and not is_right_turn:
                    continue

            angle_to_tl = math.atan2(dy, dx)
            for fwd_a in fwd_angles:
                diff = (angle_to_tl - fwd_a + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) <= half_span and dist < nearest_dist:
                    nearest_dist     = dist
                    nearest_tl       = tl
                    nearest_is_right = is_right_turn
                    break

        if nearest_tl is None:
            return 0, 0, 0
        state_map = {'red': 0, 'yellow': 1, 'green': 2}
        return 1, state_map.get(nearest_tl['state'], 0), int(nearest_is_right)

    def _get_road_info(self) -> List[float]:
        """
        현재 차량 위치의 direction_grid 셀 조회.

        반환: [is_intersection, dir_x, dir_y]
        - 도로 위    : [0.0, dx, dy]   (단위 방향벡터)
        - 교차로     : [1.0, 0.0, 0.0]
        - 도로 밖    : [0.0, 0.0, 0.0]
        """
        key  = f"{int(self.car.x / 10)}_{int(self.car.y / 10)}"
        cell = self.direction_grid.get(key)
        if cell is None:
            return [0.0, 0.0, 0.0]
        return [float(cell['is_intersection']), cell['dir'][0], cell['dir'][1]]

    def get_state(self) -> np.ndarray:
        """
        에이전트 관측 벡터 생성. 항상 고정 크기 (STATE_SIZE = 16).

        구성:
          [0:8]   센서 거리 × 8          (정규화, 0=바로 닿음, 1=최대 거리)
          [8]     신호등 상태             (0=없음, 1=초록, 0.5=노랑, -1=빨강)
          [9]     is_intersection        (0 or 1)
          [10:12] 도로 방향벡터 (dir_x, dir_y)
          [12:14] 정규화 속도 (vx, vy)   (÷ max_speed)
          [14:16] 헤딩 (cos, sin)

        반환: np.ndarray shape=(16,) dtype=float32
        """
        sensors   = self._get_sensor_data()          # 8
        tl_state  = self._get_traffic_light_state()  # 1
        road_info = self._get_road_info()             # 3

        vx_n = self.car.velocity_x / self.car.max_speed
        vy_n = self.car.velocity_y / self.car.max_speed
        cos_h = math.cos(self.car.angle)
        sin_h = math.sin(self.car.angle)

        state = sensors + [tl_state] + road_info + [vx_n, vy_n, cos_h, sin_h]
        return np.array(state, dtype=np.float32)

    # ----------------------------------------------------------
    # 기존 게임 로직
    # ----------------------------------------------------------
    def _update_camera(self):
        target_x = self.car.x - self.width  // 2
        target_y = self.car.y - self.height // 2
        self.camera_x += (target_x - self.camera_x) * self.camera_smooth
        self.camera_y += (target_y - self.camera_y) * self.camera_smooth

    def _check_collision(self) -> bool:
        corners = self.car.get_corners()
        if self.track_mask is not None:
            for x, y in corners:
                ix, iy = int(x), int(y)
                if (ix < 0 or iy < 0
                        or ix >= self.track_mask.shape[1]
                        or iy >= self.track_mask.shape[0]):
                    return True
                if self.track_mask[iy, ix] == 0:
                    return True
        return False

    def _check_goal(self) -> bool:
        if self.end_pos is None:
            return False
        return math.hypot(self.car.x - self.end_pos[0],
                          self.car.y - self.end_pos[1]) < 30

    def _check_checkpoints(self) -> int:
        for i, cp in enumerate(self.checkpoints):
            if i not in self.checkpoints_reached:
                if math.hypot(self.car.x - cp[0], self.car.y - cp[1]) < 30:
                    return i
        return -1

    def _draw_traffic_lights(self):
        TL_R = (255,30,30);  TL_Y = (255,220,0);  TL_G = (0,220,0)
        OFF_R = (80,0,0);    OFF_Y = (80,70,0);   OFF_G = (0,70,0)
        for tl in self.traffic_lights:
            x = int(tl['pos'][0] - self.camera_x)
            y = int(tl['pos'][1] - self.camera_y)
            pygame.draw.rect(self.screen, self.BLACK, (x-14, y-38, 28, 76), border_radius=5)
            pygame.draw.circle(self.screen, TL_R if tl['state']=='red'    else OFF_R, (x, y-25), 9)
            pygame.draw.circle(self.screen, TL_Y if tl['state']=='yellow' else OFF_Y, (x, y),    9)
            pygame.draw.circle(self.screen, TL_G if tl['state']=='green'  else OFF_G, (x, y+25), 9)

    def _draw_sensors(self):
        """센서 레이를 화면에 시각화 (디버그용)"""
        ox, oy  = self.car.x, self.car.y
        heading = self.car.angle
        max_d   = self.car.sensor_range
        for i, offset in enumerate(SENSOR_ANGLES):
            angle  = heading + offset
            dist_n = self._cast_ray(ox, oy, angle)
            dist   = dist_n * max_d
            ex = ox + math.cos(angle) * dist
            ey = oy + math.sin(angle) * dist
            color = (255, 80, 80) if dist_n < 0.3 else (80, 200, 255)
            pygame.draw.line(self.screen, color,
                             (int(ox - self.camera_x), int(oy - self.camera_y)),
                             (int(ex - self.camera_x), int(ey - self.camera_y)), 1)
            pygame.draw.circle(self.screen, color,
                               (int(ex - self.camera_x), int(ey - self.camera_y)), 3)

    def _draw(self):
        if self.headless:
            return
        self.screen.fill(self.WHITE)
        self.screen.blit(self.track_surface, (-self.camera_x, -self.camera_y))

        if self.start_pos:
            sx, sy = int(self.start_pos[0]-self.camera_x), int(self.start_pos[1]-self.camera_y)
            pygame.draw.circle(self.screen, self.GREEN, (sx, sy), 20)
            self.screen.blit(self.font.render("START", True, self.WHITE),
                             self.font.render("START", True, self.WHITE).get_rect(center=(sx, sy)))

        if self.end_pos:
            ex, ey = int(self.end_pos[0]-self.camera_x), int(self.end_pos[1]-self.camera_y)
            pygame.draw.circle(self.screen, self.RED, (ex, ey), 20)
            self.screen.blit(self.font.render("GOAL", True, self.WHITE),
                             self.font.render("GOAL", True, self.WHITE).get_rect(center=(ex, ey)))

        for i, cp in enumerate(self.checkpoints):
            cx, cy = int(cp[0]-self.camera_x), int(cp[1]-self.camera_y)
            color  = self.GRAY if i in self.checkpoints_reached else (255,255,0)
            pygame.draw.circle(self.screen, color, (cx, cy), 15)
            self.screen.blit(self.font.render(str(i+1), True, self.BLACK),
                             self.font.render(str(i+1), True, self.BLACK).get_rect(center=(cx, cy)))

        self._draw_traffic_lights()
        self._draw_sensors()
        self.car.draw(self.screen, (self.camera_x, self.camera_y))

        # HUD
        speed_kmh = self.car.speed * 0.36
        elapsed   = ((self.end_time or pygame.time.get_ticks()) - self.start_time) / 1000
        state     = self.get_state()
        hud = [
            f"Speed: {speed_kmh:.1f} km/h",
            f"Time: {elapsed:.1f}s",
            f"TL: {state[8]:+.1f}  Intersection: {int(state[9])}",
            f"RoadDir: ({state[10]:.2f}, {state[11]:.2f})",
        ]
        if self.car.is_drifting: hud.append("DRIFT!")
        for j, t in enumerate(hud):
            self.screen.blit(self.font.render(t, True, self.BLACK), (10, 10+j*28))

        if self.collision:
            self.screen.blit(
                self.big_font.render("CRASHED! R to Reset", True, self.RED),
                self.big_font.render("CRASHED! R to Reset", True, self.RED)
                    .get_rect(center=(self.width//2, self.height//2)))
        if self.goal_reached:
            self.screen.blit(
                self.big_font.render(f"GOAL! {elapsed:.2f}s", True, self.GREEN),
                self.big_font.render(f"GOAL! {elapsed:.2f}s", True, self.GREEN)
                    .get_rect(center=(self.width//2, self.height//2)))

        pygame.display.flip()

    # ----------------------------------------------------------
    # RL 인터페이스
    # ----------------------------------------------------------
    def reset(self):
        sp = self.start_pos or [self.width//2, self.height//2]
        self.car = Car(sp[0], sp[1], angle=0, car_info=self.car_json)

        self.collision = self.goal_reached = False
        self.total_distance = 0
        self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.camera_x = self.camera_y = 0
        self.checkpoints_reached = []
        # 신호등 시퀀스는 리셋하지 않음 — 에피소드 간 연속 작동으로 다양한 신호 상황 노출

    def step(self, controls):
        """
        강화학습 스텝.
        반환: (state, reward, done, info)
        """
        dt = 1 / self.fps
        dt_ms = dt * 1000
        self.current_time = pygame.time.get_ticks() - self.start_time

        prev_x, prev_y = self.car.x, self.car.y
        self.car.update(dt, controls)
        self.total_distance += math.hypot(self.car.x-prev_x, self.car.y-prev_y)

        self._update_camera()
        self._update_traffic_lights(dt_ms)

        self.collision = self._check_collision()

        cp_idx = self._check_checkpoints()
        if cp_idx != -1:
            self.checkpoints_reached.append(cp_idx)

        self.goal_reached = self._check_goal()

        if (self.collision or self.goal_reached) and self.end_time is None:
            self.end_time = pygame.time.get_ticks()

        # 리워드
        reward = self.car.speed * 0.01
        if self.collision:   reward -= 100
        if self.goal_reached: reward += 1000

        done  = self.collision or self.goal_reached
        info  = {
            'speed':        self.car.speed,
            'distance':     self.total_distance,
            'collision':    self.collision,
            'goal_reached': self.goal_reached,
            'time':         self.current_time / 1000,
        }
        return reward, done, info

    def run(self):
        """수동 플레이"""
        if self.headless:
            print("headless 모드에서는 수동 플레이 불가")
            return

        print("=" * 50)
        print("Controls: Arrow Keys / Space(Drift) / R(Reset)")
        print(f"State size: {STATE_SIZE}")
        print("=" * 50)

        running = True
        while running:
            controls = dict(forward=False, backward=False,
                            left=False,    right=False, brake=False)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            controls['forward']  = bool(keys[pygame.K_UP])
            controls['backward'] = bool(keys[pygame.K_DOWN])
            controls['left']     = bool(keys[pygame.K_LEFT])
            controls['right']    = bool(keys[pygame.K_RIGHT])
            controls['brake']    = bool(keys[pygame.K_SPACE])
            if keys[pygame.K_r]: self.reset()

            if not self.collision and not self.goal_reached:
                state, reward, done, info = self.step(controls)
                if done:
                    print("GOAL!" if self.goal_reached else "CRASHED!")

            self._draw()
            self.clock.tick(self.fps)

        pygame.quit()


# ============================================================
if __name__ == "__main__":
    import os
    track_file = "track_data.json" if os.path.exists("track_data.json") else "track.json"
    if not os.path.exists(track_file):
        print("트랙 파일이 없습니다. track_editor2 copy.py 로 먼저 트랙을 만들어주세요.")
    else:
        RacingGame(track_file).run()

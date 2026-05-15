import pygame
import numpy as np
import json
import math
import random
from typing import Tuple, Optional, List

# ============================================================
# 상수
# ============================================================
SENSOR_ANGLES      = [i * (math.pi / 4) for i in range(8)]
FORWARD_SENSOR_IDX = [0, 1, 7]
TL_ENCODE          = {'green': 1.0, 'yellow': 0.5, 'red': -1.0}
STATE_SIZE         = 16

# 에이전트별 차량 색상 (최대 20대)
AGENT_COLORS = [
    (0,100,255),(255,80,0),(0,180,0),(180,0,180),(200,170,0),
    (0,180,180),(180,80,0),(80,0,180),(220,0,100),(0,200,80),
    (120,180,255),(255,160,80),(80,220,80),(220,80,220),(220,220,80),
    (80,220,220),(220,160,80),(160,80,220),(220,80,160),(80,180,80),
]


# ============================================================
# Car
# ============================================================
class Car:
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
        speed_ratio  = self._calculate_speed_ratio()
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

    def draw(self, surface, camera_offset=(0, 0), color=None):
        corners = self.get_corners()
        screen_corners = [(x-camera_offset[0], y-camera_offset[1]) for x,y in corners]
        draw_color = color if color else ((255,100,100) if self.is_drifting else (0,100,255))
        pygame.draw.polygon(surface, draw_color, screen_corners)
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
    2D 탑뷰 레이싱 게임 — 다중 에이전트 지원
    - vehicle_configs: 단일 경로(str) 또는 경로 리스트(list[str])
    - 각 차량 JSON의 checkpoints 마지막 항목 = 그 차량의 GOAL
    - step(controls_list) → 에이전트별 result dict 리스트 반환
    """

    def __init__(self, track_file="track_data.json", width=1200, height=800,
                 vehicle_configs=None, car_json_path=None, headless=False):
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

        # vehicle_configs 정규화 (하위 호환: car_json_path)
        if vehicle_configs is None and car_json_path is not None:
            vehicle_configs = [car_json_path]
        elif vehicle_configs is None:
            vehicle_configs = ["./vehicle_config.json"]
        elif isinstance(vehicle_configs, str):
            vehicle_configs = [vehicle_configs]

        self.n_agents  = len(vehicle_configs)
        self.car_jsons = []
        for p in vehicle_configs:
            with open(p, 'r', encoding='utf-8') as f:
                self.car_jsons.append(json.load(f))

        self.WHITE = (255, 255, 255)
        self.BLACK = (0,   0,   0)
        self.GREEN = (0,   200, 0)
        self.RED   = (200, 0,   0)
        self.GRAY  = (100, 100, 100)

        self.track_surface   = None
        self.track_mask      = None
        self.direction_grid  = {}
        self.lane_data       = []
        self.traffic_lights  = []
        self.lane_segments: List[Tuple] = []
        self.start_positions = []
        self.start_angles_deg = []
        self.end_pos         = None
        self.checkpoints     = []    # 트랙 전체 체크포인트 좌표 목록

        self.tl_green_ms  = 12000
        self.tl_yellow_ms = 2000
        self.tl_seq_idx   = 0
        self.tl_seq_timer = 0
        self.tl_seq_phase = 'green'

        self._load_track(track_file)

        # 다중 차량 상태
        self.cars             = []
        self.car_checkpoints  = []   # 에이전트별 nav waypoint 좌표 리스트 (goal 제외)
        self.car_goals        = []   # 에이전트별 goal 좌표
        self.car_cp_reached   = []   # 에이전트별 도달한 nav cp 인덱스 리스트
        self.car_collisions   = []
        self.car_goal_reached = []
        self.car_end_times    = []
        self.tl_prev_dots     = {}   # {(car_idx, tl_idx): prev_dot}

        self._init_cars()

        # 하위 호환: self.car, self.start_pos (car 0 기준)
        self.car       = self.cars[0] if self.cars else None
        self.start_pos = self.start_positions[0] if self.start_positions else None

        self.camera_x = self.camera_y = 0
        self.camera_smooth = 0.1

        self.collision          = False
        self.goal_reached       = False
        self.total_distance     = 0
        self.start_time         = pygame.time.get_ticks()
        self.end_time           = None
        self.current_time       = 0.0
        self.sim_time_ms        = 0.0
        self.checkpoints_reached = []   # 하위 호환 (car 0)

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
    # 차량 초기화
    # ----------------------------------------------------------
    def _init_cars(self):
        self.cars             = []
        self.car_checkpoints  = []
        self.car_goals        = []
        self.car_cp_reached   = []
        self.car_collisions   = []
        self.car_goal_reached = []
        self.car_end_times    = []

        for i, cj in enumerate(self.car_jsons):
            sp_idx = cj.get('start_point', i % max(len(self.start_positions), 1))
            if self.start_positions:
                sp = self.start_positions[sp_idx % len(self.start_positions)]
            else:
                sp = [self.width // 2, self.height // 2]

            if self.start_angles_deg:
                start_angle_deg = self.start_angles_deg[sp_idx % len(self.start_angles_deg)]
            else:
                start_angle_deg = cj.get('start_angle_deg', 0.0)
            start_angle = math.radians(start_angle_deg)
            self.cars.append(Car(sp[0], sp[1], angle=start_angle, car_info=cj))

            # 체크포인트 로드: 마지막 = GOAL, 나머지 = nav waypoints
            cp_indices = cj.get('checkpoints', [])
            if cp_indices and self.checkpoints:
                cp_coords = [self.checkpoints[j] for j in cp_indices if j < len(self.checkpoints)]
                if cp_coords:
                    goal    = cp_coords[-1]
                    nav_cps = cp_coords[:-1]
                else:
                    goal    = self.end_pos
                    nav_cps = []
            else:
                nav_cps = self.checkpoints[:]
                goal    = self.end_pos

            self.car_checkpoints.append(nav_cps)
            self.car_goals.append(goal)
            self.car_cp_reached.append([])
            self.car_collisions.append(False)
            self.car_goal_reached.append(False)
            self.car_end_times.append(None)

    # ----------------------------------------------------------
    # 트랙 로드
    # ----------------------------------------------------------
    def _load_track(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

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

            self.direction_grid = data.get('direction_grid', {})
            self.lane_data      = data.get('lane_data', [])

            if self.lane_data:
                self.track_surface, self.track_mask = self._build_track_from_lane_data()

            self.tl_green_ms  = data.get('tl_green_ms',  12000)
            self.tl_yellow_ms = data.get('tl_yellow_ms', 2000)

            self.traffic_lights = []
            for i, tl in enumerate(data.get('traffic_lights', [])):
                entry = {
                    'pos':   tuple(tl['pos']),
                    'state': 'green' if i == 0 else 'red',
                }
                if 'dir' in tl:
                    entry['dir'] = tl['dir']
                self.traffic_lights.append(entry)

            if 'start_positions' in data:
                self.start_positions = data['start_positions']
            elif 'start_pos' in data and data['start_pos'] is not None:
                self.start_positions = [data['start_pos']]
            else:
                self.start_positions = []

            self.start_angles_deg = data.get('start_angles_deg', [])

            self.end_pos     = data.get('end_pos')
            self.checkpoints = data.get('checkpoints', [])
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
        surf = pygame.Surface((self.width, self.height))
        surf.fill(self.WHITE)
        for ld in self.lane_data:
            left  = ld['left_lane']
            right = ld['right_lane']
            if len(left) < 2 or len(right) < 2:
                continue
            for i in range(len(left) - 1):
                poly = [
                    (int(left[i][0]),    int(left[i][1])),
                    (int(left[i+1][0]),  int(left[i+1][1])),
                    (int(right[i+1][0]), int(right[i+1][1])),
                    (int(right[i][0]),   int(right[i][1])),
                ]
                pygame.draw.polygon(surf, self.GRAY, poly)
        arr  = pygame.surfarray.array3d(surf)
        gray = np.mean(arr, axis=2).T
        mask = (gray < 240).astype(np.uint8) * 255
        return surf, mask

    def _build_lane_segments(self) -> List[Tuple]:
        segs = []
        for ld in self.lane_data:
            for key in ('left_lane', 'center_lane', 'right_lane'):
                pts = ld[key]
                for i in range(len(pts) - 1):
                    segs.append(((pts[i][0],   pts[i][1]),
                                 (pts[i+1][0], pts[i+1][1])))
        # NumPy 벡터화 + 공간 인덱스 (속도 최적화용 캐시)
        # _seg_a / _seg_b : (N, 2) 끝점 배열, _seg_grid : 셀 -> seg index 리스트
        self._seg_cell_size = 100
        if segs:
            self._seg_a = np.asarray([s[0] for s in segs], dtype=np.float32)
            self._seg_b = np.asarray([s[1] for s in segs], dtype=np.float32)
        else:
            self._seg_a = np.zeros((0, 2), dtype=np.float32)
            self._seg_b = np.zeros((0, 2), dtype=np.float32)
        self._seg_grid = self._build_seg_spatial_index(
            self._seg_a, self._seg_b, self._seg_cell_size)
        return segs

    @staticmethod
    def _build_seg_spatial_index(seg_a: np.ndarray, seg_b: np.ndarray, cell: int) -> dict:
        """차선 세그먼트의 (셀좌표) -> [seg index list] 인덱스를 미리 만든다.
        한 세그먼트가 걸친 모든 셀에 등록 (Bresenham 비슷한 cell-by-cell)."""
        grid: dict = {}
        if seg_a.shape[0] == 0:
            return grid
        for i in range(seg_a.shape[0]):
            ax, ay = float(seg_a[i, 0]), float(seg_a[i, 1])
            bx, by = float(seg_b[i, 0]), float(seg_b[i, 1])
            x0, y0 = int(ax // cell), int(ay // cell)
            x1, y1 = int(bx // cell), int(by // cell)
            if x0 == x1 and y0 == y1:
                grid.setdefault((x0, y0), []).append(i)
                continue
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy))
            for s in range(steps + 1):
                t = s / steps
                cx = int(round(x0 + dx * t))
                cy = int(round(y0 + dy * t))
                grid.setdefault((cx, cy), []).append(i)
        for k in grid:
            grid[k] = np.asarray(sorted(set(grid[k])), dtype=np.int32)
        return grid

    def _candidate_seg_indices(self, ox: float, oy: float, max_dist: float) -> np.ndarray:
        """차량 주변 (반경 max_dist)에서 광선 후보가 될 수 있는 세그먼트 인덱스."""
        grid = getattr(self, '_seg_grid', None)
        if not grid:
            return np.arange(self._seg_a.shape[0], dtype=np.int32)
        cell = self._seg_cell_size
        cx = int(ox // cell)
        cy = int(oy // cell)
        rng = int(math.ceil(max_dist / cell))
        out = []
        for dx in range(-rng, rng + 1):
            for dy in range(-rng, rng + 1):
                arr = grid.get((cx + dx, cy + dy))
                if arr is not None:
                    out.append(arr)
        if not out:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(out))

    # ----------------------------------------------------------
    # 신호등 업데이트
    # ----------------------------------------------------------
    def _update_traffic_lights(self, dt: float):
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
    # 레이캐스팅
    # ----------------------------------------------------------
    @staticmethod
    def _ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by) -> Optional[float]:
        sx, sy = bx - ax, by - ay
        denom  = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None
        t = ((ax - ox) * sy - (ay - oy) * sx) / denom
        s = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t >= 0 and 0.0 <= s <= 1.0:
            return t
        return None

    @staticmethod
    def _ray_min_t_vec(ox: float, oy: float, dx: float, dy: float,
                       ax: np.ndarray, ay: np.ndarray,
                       bx: np.ndarray, by: np.ndarray,
                       max_t: float) -> float:
        """벡터화 광선-세그먼트 교차: N개 세그먼트 중 가장 가까운 t (없으면 max_t).
        (수식은 _ray_segment_intersect와 동일)"""
        if ax.shape[0] == 0:
            return max_t
        sx = bx - ax
        sy = by - ay
        denom = dx * sy - dy * sx
        # denom == 0이면 평행 -> 무시
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((ax - ox) * sy - (ay - oy) * sx) / denom
            s = ((ax - ox) * dy - (ay - oy) * dx) / denom
        valid = (np.abs(denom) > 1e-10) & (t >= 0.0) & (s >= 0.0) & (s <= 1.0) & (t < max_t)
        if not np.any(valid):
            return max_t
        return float(t[valid].min())

    def _cast_ray_for_car(self, car_idx: int, ox: float, oy: float, angle: float) -> float:
        """
        car_idx 차량 기준 레이캐스팅 (NumPy 벡터화 + 공간 인덱스).
        차선 세그먼트 + 다른 차량의 bounding box 4변을 모두 장애물로 감지.
        반환: 정규화 거리 [0, 1]
        """
        car      = self.cars[car_idx]
        max_dist = car.sensor_range
        dx, dy   = math.cos(angle), math.sin(angle)

        # 차선 세그먼트: 격자에서 후보만 추려 벡터화 교차
        cand = self._candidate_seg_indices(ox, oy, max_dist)
        if cand.size > 0:
            ax = self._seg_a[cand, 0]
            ay = self._seg_a[cand, 1]
            bx = self._seg_b[cand, 0]
            by = self._seg_b[cand, 1]
            min_t = self._ray_min_t_vec(ox, oy, dx, dy, ax, ay, bx, by, max_dist)
        else:
            min_t = max_dist

        # 다른 차량의 4변 (개수 적어 NumPy 한 번으로 묶음)
        if len(self.cars) > 1:
            edges_a = []
            edges_b = []
            for j, other in enumerate(self.cars):
                if j == car_idx:
                    continue
                corners = other.get_corners()
                for k in range(4):
                    edges_a.append(corners[k])
                    edges_b.append(corners[(k + 1) % 4])
            if edges_a:
                eA = np.asarray(edges_a, dtype=np.float32)
                eB = np.asarray(edges_b, dtype=np.float32)
                t_other = self._ray_min_t_vec(
                    ox, oy, dx, dy, eA[:, 0], eA[:, 1], eB[:, 0], eB[:, 1], min_t)
                if t_other < min_t:
                    min_t = t_other

        return min_t / max_dist

    def _cast_rays_for_car(self, car_idx: int, ox: float, oy: float, angles) -> list:
        """
        car_idx 차량 기준 여러 방향 레이캐스팅.
        같은 위치에서 쏘는 ray들은 차선 후보와 다른 차량 bbox를 공유하므로
        한 번만 준비하고 방향별 교차만 반복한다.
        """
        car      = self.cars[car_idx]
        max_dist = car.sensor_range

        cand = self._candidate_seg_indices(ox, oy, max_dist)
        if cand.size > 0:
            seg_ax = self._seg_a[cand, 0]
            seg_ay = self._seg_a[cand, 1]
            seg_bx = self._seg_b[cand, 0]
            seg_by = self._seg_b[cand, 1]
        else:
            seg_ax = seg_ay = seg_bx = seg_by = None

        edge_ax = edge_ay = edge_bx = edge_by = None
        if len(self.cars) > 1:
            edges_a = []
            edges_b = []
            for j, other in enumerate(self.cars):
                if j == car_idx:
                    continue
                corners = other.get_corners()
                for k in range(4):
                    edges_a.append(corners[k])
                    edges_b.append(corners[(k + 1) % 4])
            if edges_a:
                eA = np.asarray(edges_a, dtype=np.float32)
                eB = np.asarray(edges_b, dtype=np.float32)
                edge_ax, edge_ay = eA[:, 0], eA[:, 1]
                edge_bx, edge_by = eB[:, 0], eB[:, 1]

        out = []
        for angle in angles:
            dx, dy = math.cos(angle), math.sin(angle)
            if seg_ax is not None:
                min_t = self._ray_min_t_vec(
                    ox, oy, dx, dy, seg_ax, seg_ay, seg_bx, seg_by, max_dist)
            else:
                min_t = max_dist

            if edge_ax is not None:
                t_other = self._ray_min_t_vec(
                    ox, oy, dx, dy, edge_ax, edge_ay, edge_bx, edge_by, min_t)
                if t_other < min_t:
                    min_t = t_other
            out.append(min_t / max_dist)
        return out

    def _cast_ray(self, ox: float, oy: float, angle: float) -> float:
        """하위 호환: car 0 기준"""
        return self._cast_ray_for_car(0, ox, oy, angle)

    # ----------------------------------------------------------
    # 도로 정보
    # ----------------------------------------------------------
    def _get_road_info_for_car(self, car_idx: int) -> List[float]:
        car  = self.cars[car_idx]
        key  = f"{int(car.x / 10)}_{int(car.y / 10)}"
        cell = self.direction_grid.get(key)
        if cell is None:
            return [0.0, 0.0, 0.0]
        return [float(cell['is_intersection']), cell['dir'][0], cell['dir'][1]]

    def _get_road_info(self) -> List[float]:
        return self._get_road_info_for_car(0)

    # ----------------------------------------------------------
    # 신호등 정보
    # ----------------------------------------------------------
    def _get_traffic_light_info_for_car(self, car_idx: int) -> tuple:
        """
        car_idx 차량 기준 신호등 감지.
        반환: (tl_exists, tl_state, right_turnable)
        """
        car      = self.cars[car_idx]
        ox, oy   = car.x, car.y
        heading  = car.angle
        max_dist = car.sensor_range
        fwd_angles = [heading + SENSOR_ANGLES[i] for i in FORWARD_SENSOR_IDX]
        half_span  = math.pi / 8

        spd = car.speed
        if spd > 1:
            car_dx = car.velocity_x / spd
            car_dy = car.velocity_y / spd
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
                fwd_align   = car_dx * tl_dir[0] + car_dy * tl_dir[1]
                right_dx    = -tl_dir[1]
                right_dy    =  tl_dir[0]
                right_align = car_dx * right_dx + car_dy * right_dy
                is_right_turn = right_align > 0.7
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

    def _get_traffic_light_info(self) -> tuple:
        return self._get_traffic_light_info_for_car(0)

    # ----------------------------------------------------------
    # 정지선 위반 감지
    # ----------------------------------------------------------
    def _check_red_light_crossing_for_car(self, car_idx: int,
                                           prev_x: float, prev_y: float) -> tuple:
        """
        신호등 방향벡터에 수직인 정지선을 빨간불 중에 넘으면 위반.
        반환: (crossed: bool, is_right_turn: bool)
        """
        car      = self.cars[car_idx]
        car_x, car_y = car.x, car.y
        spd      = car.speed
        heading  = car.angle
        if spd > 1:
            car_dx = car.velocity_x / spd
            car_dy = car.velocity_y / spd
        else:
            car_dx = math.cos(heading)
            car_dy = math.sin(heading)

        for tl_idx, tl in enumerate(self.traffic_lights):
            tl_dir = tl.get('dir')
            if tl_dir is None:
                continue
            if tl['state'] != 'red':
                # prev_dot을 삭제하지 않고 유지 — 초록→빨강 전환 직후에도
                # 정지선 통과 여부를 연속으로 감지할 수 있도록 함
                continue

            dir_x, dir_y = tl_dir[0], tl_dir[1]
            tx, ty = tl['pos']

            fwd_align   = car_dx * dir_x + car_dy * dir_y
            right_align = car_dx * (-dir_y) + car_dy * dir_x
            is_right    = right_align > 0.7
            if fwd_align < 0.3 and not is_right:
                self.tl_prev_dots.pop((car_idx, tl_idx), None)
                continue

            curr_dot = (car_x - tx) * dir_x + (car_y - ty) * dir_y
            key      = (car_idx, tl_idx)
            prev_dot = self.tl_prev_dots.get(key)
            self.tl_prev_dots[key] = curr_dot

            if prev_dot is not None and prev_dot < 0 and curr_dot >= 0:
                return True, is_right

        return False, False

    def _check_red_light_crossing(self, prev_x: float, prev_y: float) -> tuple:
        return self._check_red_light_crossing_for_car(0, prev_x, prev_y)

    # ----------------------------------------------------------
    # 충돌 감지
    # ----------------------------------------------------------
    def _check_collision_for_car(self, car_idx: int) -> bool:
        car     = self.cars[car_idx]
        corners = car.get_corners()

        # 트랙 경계
        if self.track_mask is not None:
            for x, y in corners:
                ix, iy = int(x), int(y)
                if (ix < 0 or iy < 0
                        or ix >= self.track_mask.shape[1]
                        or iy >= self.track_mask.shape[0]):
                    return True
                if self.track_mask[iy, ix] == 0:
                    return True

        # 차량 간 충돌 (근사: 중심 간 거리)
        for j, other in enumerate(self.cars):
            if j == car_idx:
                continue
            dist     = math.hypot(car.x - other.x, car.y - other.y)
            min_dist = (car.length + other.length) * 0.5
            if dist < min_dist:
                return True

        return False

    def _check_collision(self) -> bool:
        return self._check_collision_for_car(0)

    # ----------------------------------------------------------
    # 체크포인트 / 골
    # ----------------------------------------------------------
    def _check_checkpoints_for_car(self, car_idx: int) -> int:
        """순서대로 다음 nav 체크포인트에 도달하면 해당 인덱스 반환.
        순서 외 CP(등록되지 않은 CP 포함)는 무시. 없으면 -1."""
        car      = self.cars[car_idx]
        nav_cps  = self.car_checkpoints[car_idx]
        reached  = self.car_cp_reached[car_idx]
        next_idx = len(reached)   # 다음에 도달해야 할 CP 인덱스 (순서 강제)
        if next_idx < len(nav_cps):
            cp = nav_cps[next_idx]
            if math.hypot(car.x - cp[0], car.y - cp[1]) < 30:
                return next_idx
        return -1

    def _check_goal_for_car(self, car_idx: int) -> bool:
        car  = self.cars[car_idx]
        goal = self.car_goals[car_idx]
        if goal is None:
            return False
        return math.hypot(car.x - goal[0], car.y - goal[1]) < 30

    def _check_checkpoints(self) -> int:
        return self._check_checkpoints_for_car(0)

    def _check_goal(self) -> bool:
        return self._check_goal_for_car(0)

    # ----------------------------------------------------------
    # 센서 / 상태 (하위 호환, car 0)
    # ----------------------------------------------------------
    def _get_sensor_data(self) -> List[float]:
        car = self.cars[0]
        return [self._cast_ray_for_car(0, car.x, car.y, car.angle + offset)
                for offset in SENSOR_ANGLES]

    def _get_traffic_light_state(self) -> float:
        ox, oy   = self.car.x, self.car.y
        heading  = self.car.angle
        max_dist = self.car.sensor_range
        fwd_angles = [heading + SENSOR_ANGLES[i] for i in FORWARD_SENSOR_IDX]
        half_span  = math.pi / 8
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
                diff = (angle_to_tl - fwd_a + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) <= half_span and dist < nearest_dist:
                    nearest_dist = dist
                    nearest_tl   = tl
                    break
        if nearest_tl is None:
            return 0.0
        return TL_ENCODE.get(nearest_tl['state'], 0.0)

    def get_state(self) -> np.ndarray:
        sensors   = self._get_sensor_data()
        tl_state  = self._get_traffic_light_state()
        road_info = self._get_road_info()
        vx_n = self.car.velocity_x / self.car.max_speed
        vy_n = self.car.velocity_y / self.car.max_speed
        cos_h = math.cos(self.car.angle)
        sin_h = math.sin(self.car.angle)
        state = sensors + [tl_state] + road_info + [vx_n, vy_n, cos_h, sin_h]
        return np.array(state, dtype=np.float32)

    # ----------------------------------------------------------
    # 카메라
    # ----------------------------------------------------------
    def _update_camera(self):
        if not self.cars:
            return
        target_x = self.cars[0].x - self.width  // 2
        target_y = self.cars[0].y - self.height // 2
        self.camera_x += (target_x - self.camera_x) * self.camera_smooth
        self.camera_y += (target_y - self.camera_y) * self.camera_smooth

    # ----------------------------------------------------------
    # 렌더링
    # ----------------------------------------------------------
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
        if not self.cars:
            return
        car     = self.cars[0]
        ox, oy  = car.x, car.y
        heading = car.angle
        max_d   = car.sensor_range
        for i, offset in enumerate(SENSOR_ANGLES):
            angle  = heading + offset
            dist_n = self._cast_ray_for_car(0, ox, oy, angle)
            dist   = dist_n * max_d
            ex = ox + math.cos(angle) * dist
            ey = oy + math.sin(angle) * dist
            color = (255, 80, 80) if dist_n < 0.3 else (80, 200, 255)
            pygame.draw.line(self.screen, color,
                             (int(ox-self.camera_x), int(oy-self.camera_y)),
                             (int(ex-self.camera_x), int(ey-self.camera_y)), 1)
            pygame.draw.circle(self.screen, color,
                               (int(ex-self.camera_x), int(ey-self.camera_y)), 3)

    def _draw(self):
        if self.headless:
            return
        self.screen.fill(self.WHITE)
        self.screen.blit(self.track_surface, (-self.camera_x, -self.camera_y))

        # 시작 위치들
        for idx, sp in enumerate(self.start_positions):
            sx, sy = int(sp[0]-self.camera_x), int(sp[1]-self.camera_y)
            pygame.draw.circle(self.screen, self.GREEN, (sx, sy), 15)
            lbl = self.font.render(f"S{idx}", True, self.WHITE)
            self.screen.blit(lbl, lbl.get_rect(center=(sx, sy)))

        # 체크포인트 (track 전체)
        for i, cp in enumerate(self.checkpoints):
            cx, cy = int(cp[0]-self.camera_x), int(cp[1]-self.camera_y)
            pygame.draw.circle(self.screen, (255,255,0), (cx, cy), 10)
            lbl = self.font.render(str(i), True, self.BLACK)
            self.screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

        # 에이전트별 goal 표시
        for i, goal in enumerate(self.car_goals):
            if goal:
                gx, gy = int(goal[0]-self.camera_x), int(goal[1]-self.camera_y)
                color  = AGENT_COLORS[i % len(AGENT_COLORS)]
                pygame.draw.circle(self.screen, color, (gx, gy), 14, 3)

        self._draw_traffic_lights()
        self._draw_sensors()

        # 모든 차량 그리기 (에이전트 번호 + 색상)
        for i, car in enumerate(self.cars):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            car.draw(self.screen, (self.camera_x, self.camera_y), color=color)
            # 번호 표시
            lx = int(car.x - self.camera_x)
            ly = int(car.y - self.camera_y - car.length)
            lbl = self.font.render(str(i+1), True, color)
            self.screen.blit(lbl, lbl.get_rect(center=(lx, ly)))

        # HUD (car 0 기준)
        if self.cars:
            car = self.cars[0]
            speed_kmh = car.speed * 0.36
            elapsed   = ((self.car_end_times[0] or pygame.time.get_ticks()) - self.start_time) / 1000
            hud = [
                f"Agents: {self.n_agents}",
                f"Car0 Speed: {speed_kmh:.1f} km/h",
                f"Time: {elapsed:.1f}s",
            ]
            for j, t in enumerate(hud):
                self.screen.blit(self.font.render(t, True, self.BLACK), (10, 10+j*28))

        pygame.display.flip()

    # ----------------------------------------------------------
    # RL 인터페이스
    # ----------------------------------------------------------
    def reset(self):
        """모든 차량 및 에피소드 상태 초기화"""
        for i, cj in enumerate(self.car_jsons):
            sp_idx = cj.get('start_point', i % max(len(self.start_positions), 1))
            if self.start_positions:
                sp = self.start_positions[sp_idx % len(self.start_positions)]
            else:
                sp = [self.width // 2, self.height // 2]
            if self.start_angles_deg:
                start_angle_deg = self.start_angles_deg[sp_idx % len(self.start_angles_deg)]
            else:
                start_angle_deg = cj.get('start_angle_deg', 0.0)
            start_angle = math.radians(start_angle_deg)
            self.cars[i]              = Car(sp[0], sp[1], angle=start_angle, car_info=cj)
            self.car_collisions[i]    = False
            self.car_goal_reached[i]  = False
            self.car_cp_reached[i]    = []
            self.car_end_times[i]     = None

        self.car       = self.cars[0] if self.cars else None
        self.start_pos = self.start_positions[0] if self.start_positions else None

        self.collision          = False
        self.goal_reached       = False
        self.total_distance     = 0
        self.start_time         = pygame.time.get_ticks()
        self.end_time           = None
        self.current_time       = 0.0
        self.sim_time_ms        = 0.0
        self.camera_x           = self.camera_y = 0
        self.checkpoints_reached = []
        self.tl_prev_dots        = {}
        # 신호등 시퀀스 유지

    def step(self, controls_list) -> list:
        """
        다중 에이전트 스텝.

        controls_list: list of control dicts, 길이 = n_agents
                       (단일 에이전트 호환: dict 1개를 list로 감싸도 됨)

        반환: list of result dicts (에이전트 수만큼)
          각 dict:
            collision            : bool
            goal_reached         : bool
            cp_reached           : bool  — 이 프레임에 nav cp 도달
            cp_idx               : int   — 도달한 cp 인덱스 (-1이면 없음)
            red_light_crossed    : bool
            red_light_right_turn : bool
            done                 : bool
        """
        # 단일 에이전트 호환: dict가 직접 넘어오면 리스트로 감쌈
        if isinstance(controls_list, dict):
            controls_list = [controls_list]

        dt    = 1 / self.fps
        dt_ms = dt * 1000
        self.sim_time_ms += dt_ms
        self.current_time = self.sim_time_ms

        prev_positions = [(car.x, car.y) for car in self.cars]

        # 모든 차 이동
        for i, car in enumerate(self.cars):
            if not self.car_collisions[i] and not self.car_goal_reached[i]:
                ctrl = controls_list[i] if i < len(controls_list) else {}
                car.update(dt, ctrl)

        self._update_camera()
        self._update_traffic_lights(dt_ms)

        results = []
        for i, car in enumerate(self.cars):
            # 이미 종료된 에이전트
            if self.car_collisions[i] or self.car_goal_reached[i]:
                results.append({
                    'collision':            self.car_collisions[i],
                    'goal_reached':         self.car_goal_reached[i],
                    'cp_reached':           False,
                    'cp_idx':               -1,
                    'red_light_crossed':    False,
                    'red_light_right_turn': False,
                    'done':                 True,
                })
                continue

            prev_x, prev_y = prev_positions[i]

            collision = self._check_collision_for_car(i)
            self.car_collisions[i] = collision

            cp_idx = self._check_checkpoints_for_car(i)
            if cp_idx != -1:
                self.car_cp_reached[i].append(cp_idx)

            goal = self._check_goal_for_car(i)
            self.car_goal_reached[i] = goal

            red_crossed, rl_right = self._check_red_light_crossing_for_car(i, prev_x, prev_y)

            if (collision or goal) and self.car_end_times[i] is None:
                self.car_end_times[i] = pygame.time.get_ticks()

            results.append({
                'collision':            collision,
                'goal_reached':         goal,
                'cp_reached':           cp_idx != -1,
                'cp_idx':               cp_idx,
                'red_light_crossed':    red_crossed,
                'red_light_right_turn': rl_right,
                'done':                 collision or goal,
            })

        # 하위 호환 (car 0)
        self.collision    = self.car_collisions[0]
        self.goal_reached = self.car_goal_reached[0]
        if self.cars:
            self.checkpoints_reached = self.car_cp_reached[0]

        return results

    # ----------------------------------------------------------
    # 수동 플레이 (car 0)
    # ----------------------------------------------------------
    def run(self):
        if self.headless:
            print("headless 모드에서는 수동 플레이 불가")
            return

        print("=" * 50)
        print("Controls: Arrow Keys / Space(Drift) / R(Reset)")
        print(f"Agents: {self.n_agents}")
        print("=" * 50)

        running = True
        while running:
            ctrl = dict(forward=False, backward=False, left=False, right=False, brake=False)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            ctrl['forward']  = bool(keys[pygame.K_UP])
            ctrl['backward'] = bool(keys[pygame.K_DOWN])
            ctrl['left']     = bool(keys[pygame.K_LEFT])
            ctrl['right']    = bool(keys[pygame.K_RIGHT])
            ctrl['brake']    = bool(keys[pygame.K_SPACE])
            if keys[pygame.K_r]:
                self.reset()

            if not self.car_collisions[0] and not self.car_goal_reached[0]:
                # car 0만 수동 조종, 나머지는 정지
                controls_list = [ctrl] + [{}] * (self.n_agents - 1)
                results = self.step(controls_list)
                if results[0]['done']:
                    print("GOAL!" if results[0]['goal_reached'] else "CRASHED!")

            self._draw()
            self.clock.tick(self.fps)

        pygame.quit()


# ============================================================
if __name__ == "__main__":
    import os
    track_file = "track_data.json" if os.path.exists("track_data.json") else "track.json"
    if not os.path.exists(track_file):
        print("트랙 파일이 없습니다. map_editor.py 로 먼저 트랙을 만들어주세요.")
    else:
        RacingGame(track_file,
                   vehicle_configs=["./vehicles/vehicle_1.json"]).run()

import pygame
import numpy as np
import json
import math
from typing import Tuple, Optional

class Car:
    """
    이니셜D 스타일 드리프트 물리 시뮬레이션
    
    특징:
    - 극적인 테일 슬라이드
    - 카운터 스티어링 메카닉
    - 드리프트 관성 (한번 시작하면 쭉 밀림)
    - 드리프트 중 속도 유지
    - 드리프트 앵글 시각화
    """
    
    def __init__(self, x, y, angle=0, car_info=None):
        # 위치와 방향
        self.x = x
        self.y = y
        self.angle = angle  # 차량이 바라보는 방향 (라디안)
        
        # 속도 (벡터)
        self.velocity_x = 0
        self.velocity_y = 0
        self.speed = 0
        
        # 차량 속성
        self.max_speed = car_info['max_speed']
        self.acceleration_force = car_info['acceleration_force']
        self.brake_force = 400
        
        # 기본 물리 값
        self.base_friction = car_info['friction']
        self.base_lateral_friction = car_info['lateral_friction']
        self.turn_speed = car_info['turn_speed']
        self.sensor_range = car_info['sensor_range']
        
        # 현재 적용되는 동적 값
        self.friction = self.base_friction
        self.lateral_friction = self.base_lateral_friction
        
        # 차량 크기
        self.width = 10
        self.length = 20
        
        # ========== 이니셜D 드리프트 시스템 ==========
        
        # 드리프트 상태
        self.is_drifting = False
        self.drift_direction = 0  # -1: 왼쪽 드리프트, 0: 없음, 1: 오른쪽 드리프트
        
        # 드리프트 앵글 (차량 방향과 이동 방향의 차이)
        self.drift_angle = 0  # 라디안
        self.target_drift_angle = 0
        
        # 드리프트 모멘텀 (관성) - 0~1, 높을수록 드리프트 유지
        self.drift_momentum = 0
        self.drift_momentum_decay = 0.98  # 드리프트 관성이 서서히 줄어드는 비율
        self.drift_momentum_build = 0.15  # 드리프트 관성이 쌓이는 속도
        
        # 드리프트 파라미터
        self.drift_entry_speed = 150  # 드리프트 진입 최소 속도
        self.drift_max_angle = math.radians(75)  # 최대 드리프트 각도 (75도)
        self.drift_angle_speed = 4.5  # 드리프트 각도 변화 속도
        self.drift_recovery_speed = 2.0  # 드리프트 회복 속도
        
        # 드리프트 중 물리
        self.drift_lateral_friction = car_info.get('drift_friction', 0.15)  # 드리프트 중 측면 마찰
        self.drift_speed_retention = 0.998  # 드리프트 중 속도 유지율 (높을수록 속도 유지)
        self.drift_acceleration_boost = 1.3  # 드리프트 중 가속 부스트
        
        # 카운터 스티어링
        self.counter_steer_bonus = 1.8  # 카운터 스티어 시 회전 보너스
        self.counter_steer_stability = 0.3  # 카운터 스티어 시 안정성 보너스
        
        # 드리프트 부스트 (드리프트 후 가속)
        self.drift_boost = 0
        self.drift_boost_max = 200  # 최대 부스트 속도 추가
        self.drift_boost_decay = 0.95
        
        # 타이어 스키드 마크용
        self.skid_marks = []  # [(x, y, intensity), ...]
        self.max_skid_marks = 500
        
        # 드리프트 사운드/이펙트용 상태
        self.drift_intensity = 0  # 0~1, 드리프트 강도
        
    def _get_movement_angle(self):
        """현재 이동 방향 각도 반환"""
        if self.speed < 1:
            return self.angle
        return math.atan2(self.velocity_y, self.velocity_x)
    
    def _calculate_slip_angle(self):
        """슬립 앵글 계산 (차량 방향과 이동 방향 차이)"""
        if self.speed < 5:
            return 0
        
        move_angle = self._get_movement_angle()
        slip = move_angle - self.angle
        
        # -π ~ π 범위로 정규화
        while slip > math.pi:
            slip -= 2 * math.pi
        while slip < -math.pi:
            slip += 2 * math.pi
        
        return slip
    
    def _initiate_drift(self, direction):
        """드리프트 시작"""
        if self.speed < self.drift_entry_speed:
            return
        
        self.is_drifting = True
        self.drift_direction = direction
        
        # 초기 드리프트 각도 설정
        initial_angle = math.radians(25) * direction
        self.target_drift_angle = initial_angle
        
    def _update_drift(self, dt, controls):
        """드리프트 상태 업데이트"""
        is_brake = controls.get('brake', False)
        is_left = controls.get('left', False)
        is_right = controls.get('right', False)
        is_forward = controls.get('forward', False)
        
        # 드리프트 진입 조건
        if is_brake and self.speed > self.drift_entry_speed:
            if is_left and not self.is_drifting:
                self._initiate_drift(-1)  # 왼쪽 드리프트 (차량 뒷부분이 오른쪽으로)
            elif is_right and not self.is_drifting:
                self._initiate_drift(1)   # 오른쪽 드리프트 (차량 뒷부분이 왼쪽으로)
        
        if self.is_drifting:
            # 드리프트 모멘텀 증가
            if is_brake:
                self.drift_momentum = min(1.0, self.drift_momentum + self.drift_momentum_build * dt * 10)
            
            # 드리프트 방향에 따른 목표 각도 조절
            if self.drift_direction == -1:  # 왼쪽 드리프트
                if is_left:
                    # 같은 방향 = 드리프트 각도 증가
                    self.target_drift_angle = -self.drift_max_angle * 0.9
                elif is_right:
                    # 카운터 스티어 = 드리프트 각도 감소 (컨트롤)
                    self.target_drift_angle = -self.drift_max_angle * 0.4
                else:
                    self.target_drift_angle = -self.drift_max_angle * 0.6
                    
            elif self.drift_direction == 1:  # 오른쪽 드리프트
                if is_right:
                    self.target_drift_angle = self.drift_max_angle * 0.9
                elif is_left:
                    # 카운터 스티어
                    self.target_drift_angle = self.drift_max_angle * 0.4
                else:
                    self.target_drift_angle = self.drift_max_angle * 0.6
            
            # 드리프트 각도를 목표를 향해 부드럽게 변경
            angle_diff = self.target_drift_angle - self.drift_angle
            self.drift_angle += angle_diff * self.drift_angle_speed * dt
            
            # 드리프트 강도 계산
            self.drift_intensity = min(1.0, abs(self.drift_angle) / self.drift_max_angle)
            
            # 드리프트 부스트 축적 (드리프트 유지 시간에 비례)
            if abs(self.drift_angle) > math.radians(20):
                self.drift_boost = min(self.drift_boost_max, 
                                       self.drift_boost + self.drift_intensity * 100 * dt)
            
            # 드리프트 종료 조건
            if not is_brake and self.drift_momentum < 0.3:
                self._end_drift()
            elif self.speed < self.drift_entry_speed * 0.5:
                self._end_drift()
                
            # 드리프트 모멘텀 감소
            if not is_brake:
                self.drift_momentum *= self.drift_momentum_decay
                
        else:
            # 드리프트 아닐 때 각도 복구
            self.drift_angle *= 0.9
            if abs(self.drift_angle) < 0.01:
                self.drift_angle = 0
            
            self.drift_intensity *= 0.95
            self.drift_momentum *= 0.95
    
    def _end_drift(self):
        """드리프트 종료"""
        self.is_drifting = False
        self.drift_direction = 0
        self.target_drift_angle = 0
        # 드리프트 부스트 적용 (다음 프레임부터)
        
    def update(self, dt, controls):
        """차량 업데이트 - 이니셜D 스타일"""
        
        # 현재 속도 계산
        self.speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
        # 드리프트 상태 업데이트
        self._update_drift(dt, controls)
        
        # ========== 1. 조향 ==========
        if self.speed > 5:
            base_turn = self.turn_speed
            
            # 속도에 따른 조향 감도 (고속에서 약간 둔감)
            speed_factor = 1.0 - (self.speed / self.max_speed) * 0.3
            
            if self.is_drifting:
                # 드리프트 중 조향
                if self.drift_direction == -1:  # 왼쪽 드리프트
                    if controls.get('right', False):
                        # 카운터 스티어 - 보너스 회전
                        self.angle += base_turn * self.counter_steer_bonus * dt
                    elif controls.get('left', False):
                        # 같은 방향 - 드리프트 깊게
                        self.angle -= base_turn * 0.7 * dt
                        
                elif self.drift_direction == 1:  # 오른쪽 드리프트
                    if controls.get('left', False):
                        # 카운터 스티어
                        self.angle -= base_turn * self.counter_steer_bonus * dt
                    elif controls.get('right', False):
                        self.angle += base_turn * 0.7 * dt
            else:
                # 일반 조향
                turn_amount = base_turn * speed_factor * dt
                
                if controls.get('left', False):
                    self.angle -= turn_amount
                if controls.get('right', False):
                    self.angle += turn_amount
        
        # ========== 2. 가속/브레이크 ==========
        forward_dir_x = math.cos(self.angle)
        forward_dir_y = math.sin(self.angle)
        
        if controls.get('forward', False):
            accel = self.acceleration_force
            
            # 드리프트 중 가속 부스트
            if self.is_drifting:
                accel *= self.drift_acceleration_boost
            
            # 드리프트 부스트 적용
            if self.drift_boost > 0:
                accel += self.drift_boost * 2
                self.drift_boost *= self.drift_boost_decay
            
            self.velocity_x += forward_dir_x * accel * dt
            self.velocity_y += forward_dir_y * accel * dt
        else:
            # 드리프트 부스트 감소
            self.drift_boost *= self.drift_boost_decay
        
        if controls.get('backward', False):
            if not self.is_drifting:
                self.velocity_x -= forward_dir_x * self.acceleration_force * dt * 0.5
                self.velocity_y -= forward_dir_y * self.acceleration_force * dt * 0.5
        
        # ========== 3. 드리프트 물리 ==========
        if self.is_drifting and self.speed > 10:
            # 현재 이동 방향
            move_angle = self._get_movement_angle()
            
            # 드리프트 각도만큼 차량 회전 (뒷바퀴가 미끄러지는 효과)
            drift_rotation = self.drift_angle * self.drift_momentum * 3.0 * dt
            self.angle += drift_rotation
            
            # 드리프트 중 측면 마찰 대폭 감소 (미끄러짐!)
            current_lateral_friction = self.drift_lateral_friction * (1 - self.drift_momentum * 0.5)
            
            # 측면 속도 계산
            right_dir_x = -math.sin(self.angle)
            right_dir_y = math.cos(self.angle)
            
            lateral_speed = self.velocity_x * right_dir_x + self.velocity_y * right_dir_y
            
            # 측면 속도 감소 (아주 약하게 - 미끄러짐 유지)
            self.velocity_x -= right_dir_x * lateral_speed * current_lateral_friction
            self.velocity_y -= right_dir_y * lateral_speed * current_lateral_friction
            
            # 드리프트 중 속도 유지 (브레이크가 속도를 안 줄임!)
            if controls.get('brake', False):
                # 브레이크는 드리프트 유지에만 사용, 속도는 거의 안 줄임
                self.velocity_x *= self.drift_speed_retention
                self.velocity_y *= self.drift_speed_retention
            
            # 스키드 마크 추가
            if self.drift_intensity > 0.3:
                self._add_skid_mark()
                
        else:
            # 일반 주행 - 측면 마찰 높음 (그립)
            if self.speed > 10:
                move_angle = self._get_movement_angle()
                angle_diff = move_angle - self.angle
                
                # 측면 속도 제거 (그립)
                right_dir_x = -math.sin(self.angle)
                right_dir_y = math.cos(self.angle)
                
                lateral_speed = self.velocity_x * right_dir_x + self.velocity_y * right_dir_y
                
                # 일반 주행 시 측면 마찰 높음
                grip = self.base_lateral_friction
                self.velocity_x -= right_dir_x * lateral_speed * grip
                self.velocity_y -= right_dir_y * lateral_speed * grip
            
            # 일반 브레이크
            if controls.get('brake', False) and not self.is_drifting:
                brake_strength = 0.96
                self.velocity_x *= brake_strength
                self.velocity_y *= brake_strength
        
        # ========== 4. 공기저항/마찰 ==========
        self.velocity_x *= self.base_friction
        self.velocity_y *= self.base_friction
        
        # ========== 5. 최대 속도 제한 ==========
        current_speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        effective_max_speed = self.max_speed
        
        # 드리프트 부스트 중이면 최대 속도 약간 증가
        if self.drift_boost > 50:
            effective_max_speed *= 1.1
        
        if current_speed > effective_max_speed:
            scale = effective_max_speed / current_speed
            self.velocity_x *= scale
            self.velocity_y *= scale
        
        # ========== 6. 위치 업데이트 ==========
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        
        # 속도 업데이트
        self.speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    def _add_skid_mark(self):
        """스키드 마크 추가"""
        # 뒷바퀴 위치 계산
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        rear_offset = -self.length / 2 + 5
        
        # 왼쪽 뒷바퀴
        left_x = self.x + rear_offset * cos_a - (-self.width/2) * (-sin_a)
        left_y = self.y + rear_offset * sin_a + (-self.width/2) * cos_a
        
        # 오른쪽 뒷바퀴
        right_x = self.x + rear_offset * cos_a - (self.width/2) * (-sin_a)
        right_y = self.y + rear_offset * sin_a + (self.width/2) * cos_a
        
        intensity = self.drift_intensity
        
        self.skid_marks.append((left_x, left_y, intensity))
        self.skid_marks.append((right_x, right_y, intensity))
        
        # 최대 개수 제한
        while len(self.skid_marks) > self.max_skid_marks:
            self.skid_marks.pop(0)
    
    def get_corners(self):
        """차량의 4개 코너 좌표 반환"""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        half_width = self.width / 2
        half_length = self.length / 2
        
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        ]
        
        rotated_corners = []
        for dx, dy in corners:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rotated_corners.append((self.x + rx, self.y + ry))
        
        return rotated_corners
    
    def draw(self, surface, camera_offset=(0, 0)):
        """차량 그리기"""
        # 스키드 마크 그리기
        for sx, sy, intensity in self.skid_marks:
            screen_x = int(sx - camera_offset[0])
            screen_y = int(sy - camera_offset[1])
            
            # 강도에 따른 색상 (검은색 ~ 회색)
            color_val = int(50 + (1 - intensity) * 100)
            color = (color_val, color_val, color_val)
            
            pygame.draw.circle(surface, color, (screen_x, screen_y), 2)
        
        corners = self.get_corners()
        screen_corners = [(x - camera_offset[0], y - camera_offset[1]) for x, y in corners]
        
        # 차량 색상 (드리프트 강도에 따라 변화)
        if self.is_drifting:
            # 드리프트 중: 빨간색 계열
            red = min(255, 100 + int(self.drift_intensity * 155))
            color = (red, 50, 50)
        elif self.drift_boost > 50:
            # 부스트 중: 주황색
            color = (255, 150, 50)
        else:
            # 일반: 파란색
            color = (0, 100, 255)
        
        pygame.draw.polygon(surface, color, screen_corners)
        pygame.draw.polygon(surface, (0, 0, 0), screen_corners, 2)
        
        # 바퀴 그리기
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        wheel_positions = [
            (-self.length/2 + 8, -self.width/2 + 3),
            (self.length/2 - 8, -self.width/2 + 3),
            (self.length/2 - 8, self.width/2 - 3),
            (-self.length/2 + 8, self.width/2 - 3),
        ]
        
        wheel_width = 4
        wheel_length = 8
        
        for i, (wx, wy) in enumerate(wheel_positions):
            wheel_x = self.x + wx * cos_a - wy * sin_a
            wheel_y = self.y + wx * sin_a + wy * cos_a
            
            # 앞바퀴는 조향 각도 적용
            if i < 2:  # 앞바퀴
                wheel_angle = self.angle
            else:  # 뒷바퀴
                wheel_angle = self.angle
            
            wheel_cos = math.cos(wheel_angle)
            wheel_sin = math.sin(wheel_angle)
            
            half_length = wheel_length / 2
            half_width = wheel_width / 2
            
            wheel_corners = [
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width),
            ]
            
            rotated_wheel = []
            for dx, dy in wheel_corners:
                rx = dx * wheel_cos - dy * wheel_sin
                ry = dx * wheel_sin + dy * wheel_cos
                rotated_wheel.append((
                    wheel_x + rx - camera_offset[0],
                    wheel_y + ry - camera_offset[1]
                ))
            
            # 드리프트 중 뒷바퀴 색상 변경
            if self.is_drifting and i >= 2:
                wheel_color = (80, 30, 30)  # 빨간 계열
            else:
                wheel_color = (30, 30, 30)
            
            pygame.draw.polygon(surface, wheel_color, rotated_wheel)
        
        # 전방 표시
        front_x = self.x + math.cos(self.angle) * self.length / 2
        front_y = self.y + math.sin(self.angle) * self.length / 2
        
        screen_front = (front_x - camera_offset[0], front_y - camera_offset[1])
        screen_center = (self.x - camera_offset[0], self.y - camera_offset[1])
        
        pygame.draw.line(surface, (255, 255, 0), screen_center, screen_front, 3)
        
        # 이동 방향 표시 (드리프트 시 차이 보여줌)
        if self.speed > 30:
            move_angle = self._get_movement_angle()
            move_end_x = self.x + math.cos(move_angle) * 40
            move_end_y = self.y + math.sin(move_angle) * 40
            
            screen_move = (move_end_x - camera_offset[0], move_end_y - camera_offset[1])
            pygame.draw.line(surface, (0, 255, 0), screen_center, screen_move, 2)


class RacingGame:
    """
    이니셜D 스타일 2D 탑뷰 레이싱 게임
    """
    
    def __init__(self, track_file="track.json", width=1200, height=800, 
                 car_json_path="racing_car.json", headless=False):
        self.headless = headless
        
        if headless:
            pygame.init()
            self.screen = pygame.Surface((width, height))
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("이니셜D 스타일 드리프트 레이싱")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.car_json = json.load(open(car_json_path, 'r', encoding='utf-8'))
        
        # 색상
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        self.GRAY = (100, 100, 100)
        self.DARK_GRAY = (60, 60, 60)
        
        # 트랙 로드
        self.track_surface = None
        self.track_mask = None
        self.start_pos = None
        self.end_pos = None
        self.track_file = track_file
        self._load_track(track_file)
        
        # 차량
        if self.start_pos:
            self.car = Car(self.start_pos[0], self.start_pos[1], angle=0, car_info=self.car_json)
        else:
            self.car = Car(width // 2, height // 2, angle=0, car_info=self.car_json)
        
        # 카메라
        self.camera_x = 0
        self.camera_y = 0
        self.camera_smooth = 0.1
        
        # 게임 상태
        self.collision = False
        self.goal_reached = False
        self.total_distance = 0
        self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.current_time = None
        
        self.checkpoints_reached = []
        
        # 드리프트 통계
        self.total_drift_time = 0
        self.max_drift_angle = 0
        self.drift_count = 0
        self.was_drifting = False
        
        # 폰트
        if not headless:
            try:
                self.font = pygame.font.SysFont('arial,sans-serif', 24)
                self.big_font = pygame.font.SysFont('arial,sans-serif', 48)
                self.drift_font = pygame.font.SysFont('arial,sans-serif', 72)
            except:
                self.font = pygame.font.Font(None, 28)
                self.big_font = pygame.font.Font(None, 48)
                self.drift_font = pygame.font.Font(None, 72)
        else:
            self.font = None
            self.big_font = None
            self.drift_font = None
    
    def _load_track(self, filename):
        """트랙 데이터 로드"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            track_mask = np.array(data['track_mask'], dtype=np.uint8)
            
            track_height, track_width = track_mask.shape
            self.track_surface = pygame.Surface((track_width, track_height))
            self.track_surface.fill(self.DARK_GRAY)  # 더 어두운 배경
            
            for y in range(track_height):
                for x in range(track_width):
                    if track_mask[y, x] > 0:
                        self.track_surface.set_at((x, y), self.GRAY)
            
            self.track_mask = track_mask
            self.start_pos = data.get('start_pos')
            self.end_pos = data.get('end_pos')
            self.checkpoints = data.get('checkpoints', [])
            
            if not self.headless:
                print(f"Track loaded: {filename}")
                
        except Exception as e:
            print(f"Track load failed: {e}")
            self.track_surface = pygame.Surface((1200, 800))
            self.track_surface.fill(self.DARK_GRAY)
            pygame.draw.rect(self.track_surface, self.GRAY, (200, 200, 800, 400))
            
            self.track_mask = np.zeros((800, 1200), dtype=np.uint8)
            self.track_mask[200:600, 200:1000] = 255
    
    def _update_camera(self):
        """카메라를 차량 중심으로"""
        target_x = self.car.x - self.width // 2
        target_y = self.car.y - self.height // 2
        
        self.camera_x += (target_x - self.camera_x) * self.camera_smooth
        self.camera_y += (target_y - self.camera_y) * self.camera_smooth
    
    def _check_collision(self) -> bool:
        """충돌 체크"""
        corners = self.car.get_corners()
        
        for x, y in corners:
            ix, iy = int(x), int(y)
            
            if ix < 0 or iy < 0 or ix >= self.track_mask.shape[1] or iy >= self.track_mask.shape[0]:
                return True
            
            if self.track_mask[iy, ix] == 0:
                return True
        
        return False
    
    def _check_goal(self) -> bool:
        """골인 체크"""
        if self.end_pos is None:
            return False
        
        distance = math.sqrt((self.car.x - self.end_pos[0])**2 + (self.car.y - self.end_pos[1])**2)
        return distance < 30
    
    def _check_checkpoints(self) -> int:
        """체크포인트 체크"""
        for i, checkpoint in enumerate(self.checkpoints):
            if i not in self.checkpoints_reached:
                distance = math.sqrt((self.car.x - checkpoint[0])**2 + 
                                (self.car.y - checkpoint[1])**2)
                if distance < 30:
                    return i
        return -1
    
    def _draw(self):
        """화면 그리기"""
        if self.headless:
            return
        
        self.screen.fill(self.DARK_GRAY)
        
        # 트랙
        self.screen.blit(self.track_surface, (-self.camera_x, -self.camera_y))
        
        # 시작점
        if self.start_pos:
            screen_x = self.start_pos[0] - self.camera_x
            screen_y = self.start_pos[1] - self.camera_y
            pygame.draw.circle(self.screen, self.GREEN, (int(screen_x), int(screen_y)), 20)
            text = self.font.render("START", True, self.WHITE)
            text_rect = text.get_rect(center=(screen_x, screen_y))
            self.screen.blit(text, text_rect)
        
        # 골인점
        if self.end_pos:
            screen_x = self.end_pos[0] - self.camera_x
            screen_y = self.end_pos[1] - self.camera_y
            pygame.draw.circle(self.screen, self.RED, (int(screen_x), int(screen_y)), 20)
            text = self.font.render("GOAL", True, self.WHITE)
            text_rect = text.get_rect(center=(screen_x, screen_y))
            self.screen.blit(text, text_rect)
        
        # 체크포인트
        for i, checkpoint in enumerate(self.checkpoints):
            screen_x = checkpoint[0] - self.camera_x
            screen_y = checkpoint[1] - self.camera_y
            
            if i in self.checkpoints_reached:
                color = self.GRAY
            else:
                color = (255, 255, 0)
            
            pygame.draw.circle(self.screen, color, (int(screen_x), int(screen_y)), 15)
            text = self.font.render(str(i+1), True, self.BLACK)
            text_rect = text.get_rect(center=(screen_x, screen_y))
            self.screen.blit(text, text_rect)
        
        # 차량
        self.car.draw(self.screen, (self.camera_x, self.camera_y))
        
        # HUD
        speed_kmh = self.car.speed * 0.36
        
        if self.end_time is not None:
            elapsed_time = (self.end_time - self.start_time) / 1000
        else:
            elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000
        
        # 속도계 (타코미터 스타일)
        self._draw_speedometer(speed_kmh)
        
        # 드리프트 앵글 표시
        if self.car.is_drifting:
            drift_deg = abs(math.degrees(self.car.drift_angle))
            
            # 드리프트 앵글 텍스트 (크게!)
            angle_text = self.drift_font.render(f"{drift_deg:.0f}°", True, (255, 100, 100))
            angle_rect = angle_text.get_rect(center=(self.width // 2, 100))
            self.screen.blit(angle_text, angle_rect)
            
            # DRIFT! 텍스트
            drift_text = self.big_font.render("DRIFT!", True, (255, 50, 50))
            drift_rect = drift_text.get_rect(center=(self.width // 2, 160))
            self.screen.blit(drift_text, drift_rect)
            
            # 드리프트 모멘텀 바
            bar_width = 200
            bar_height = 10
            bar_x = self.width // 2 - bar_width // 2
            bar_y = 190
            
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            fill_width = int(bar_width * self.car.drift_momentum)
            pygame.draw.rect(self.screen, (255, 100, 50), (bar_x, bar_y, fill_width, bar_height))
        
        # 드리프트 부스트 게이지
        if self.car.drift_boost > 10:
            boost_text = self.font.render(f"BOOST: {self.car.drift_boost:.0f}", True, (255, 200, 50))
            self.screen.blit(boost_text, (self.width // 2 - 60, 220))
        
        # 일반 HUD
        hud_texts = [
            f"Speed: {speed_kmh:.0f} km/h",
            f"Time: {elapsed_time:.1f}s",
            f"Drift Count: {self.drift_count}",
            f"Total Drift: {self.total_drift_time:.1f}s",
        ]
        
        y_offset = 10
        for text_str in hud_texts:
            text = self.font.render(text_str, True, self.WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 28
        
        # 조작 안내
        controls_text = [
            "↑: Accelerate  ↓: Brake/Reverse",
            "←→: Steer  SPACE+←→: DRIFT!",
            "R: Reset",
            "",
            "Drift Tip: Space + Direction to initiate",
            "Counter-steer to control angle!"
        ]
        
        y_offset = self.height - 160
        for text_str in controls_text:
            text = self.font.render(text_str, True, (200, 200, 200))
            self.screen.blit(text, (10, y_offset))
            y_offset += 24
        
        # 충돌/골인 메시지
        if self.collision:
            text = self.big_font.render("CRASHED! Press R to Reset", True, self.RED)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text, text_rect)
        
        if self.goal_reached:
            text = self.big_font.render(f"GOAL! Time: {elapsed_time:.2f}s", True, self.GREEN)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def _draw_speedometer(self, speed_kmh):
        """속도계 그리기"""
        center_x = self.width - 100
        center_y = self.height - 100
        radius = 70
        
        # 배경 원
        pygame.draw.circle(self.screen, (30, 30, 30), (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (100, 100, 100), (center_x, center_y), radius, 3)
        
        # 속도에 따른 바늘 각도 (0~max_speed를 -135도~135도로)
        max_display_speed = self.car.max_speed * 0.36  # km/h
        angle_range = 270  # 도
        
        speed_ratio = min(speed_kmh / max_display_speed, 1.0)
        needle_angle = math.radians(-135 + speed_ratio * angle_range)
        
        needle_length = radius - 15
        needle_x = center_x + math.cos(needle_angle) * needle_length
        needle_y = center_y + math.sin(needle_angle) * needle_length
        
        # 바늘
        pygame.draw.line(self.screen, (255, 50, 50), (center_x, center_y), (needle_x, needle_y), 4)
        
        # 속도 텍스트
        speed_text = self.font.render(f"{speed_kmh:.0f}", True, self.WHITE)
        speed_rect = speed_text.get_rect(center=(center_x, center_y + 25))
        self.screen.blit(speed_text, speed_rect)
        
        unit_text = self.font.render("km/h", True, (150, 150, 150))
        unit_rect = unit_text.get_rect(center=(center_x, center_y + 45))
        self.screen.blit(unit_text, unit_rect)
    
    def reset(self):
        """게임 리셋"""
        if self.start_pos:
            self.car = Car(self.start_pos[0], self.start_pos[1], angle=0, car_info=self.car_json)
        else:
            self.car = Car(self.width // 2, self.height // 2, angle=0, car_info=self.car_json)
        
        self.collision = False
        self.goal_reached = False
        self.total_distance = 0
        self.start_time = pygame.time.get_ticks()
        self.end_time = None
        self.camera_x = 0
        self.camera_y = 0
        
        self.checkpoints_reached = []
        
        # 드리프트 통계 리셋
        self.total_drift_time = 0
        self.max_drift_angle = 0
        self.drift_count = 0
        self.was_drifting = False
    
    def step(self, controls):
        """게임 스텝"""
        dt = 1 / self.fps
        self.current_time = (pygame.time.get_ticks() - self.start_time)
        
        prev_x, prev_y = self.car.x, self.car.y
        self.car.update(dt, controls)
        
        # 드리프트 통계 업데이트
        if self.car.is_drifting:
            self.total_drift_time += dt
            current_drift_deg = abs(math.degrees(self.car.drift_angle))
            if current_drift_deg > self.max_drift_angle:
                self.max_drift_angle = current_drift_deg
            
            if not self.was_drifting:
                self.drift_count += 1
        
        self.was_drifting = self.car.is_drifting
        
        distance = math.sqrt((self.car.x - prev_x)**2 + (self.car.y - prev_y)**2)
        self.total_distance += distance
        
        self._update_camera()
        
        self.collision = self._check_collision()
        
        checkpoint_idx = self._check_checkpoints()
        if checkpoint_idx != -1:
            self.checkpoints_reached.append(checkpoint_idx)
        
        self.goal_reached = self._check_goal()
        
        if (self.collision or self.goal_reached) and self.end_time is None:
            self.end_time = pygame.time.get_ticks()
        
        # 보상 계산 (강화학습용)
        reward = 0
        reward += self.car.speed * 0.01
        
        # 드리프트 보상
        if self.car.is_drifting:
            reward += self.car.drift_intensity * 0.5
        
        if self.collision:
            reward -= 100
        
        if self.goal_reached:
            reward += 1000
        
        done = self.collision or self.goal_reached
        
        info = {
            'speed': self.car.speed,
            'distance': self.total_distance,
            'collision': self.collision,
            'goal_reached': self.goal_reached,
            'time': (pygame.time.get_ticks() - self.start_time) / 1000,
            'drift_time': self.total_drift_time,
            'drift_count': self.drift_count,
            'is_drifting': self.car.is_drifting,
            'drift_angle': math.degrees(self.car.drift_angle)
        }
        
        return reward, done, info
    
    def run(self):
        """게임 실행"""
        if self.headless:
            print("Cannot run manual play in headless mode!")
            return
        
        running = True
        
        print("=" * 60)
        print("이니셜D 스타일 드리프트 레이싱")
        print("=" * 60)
        print("조작법:")
        print("  ↑ : 가속")
        print("  ↓ : 후진/브레이크")
        print("  ←→ : 조향")
        print("  SPACE + ← 또는 → : 드리프트 시작!")
        print("  ")
        print("드리프트 팁:")
        print("  1. 속도를 충분히 올린다 (150+ 필요)")
        print("  2. SPACE + 방향키로 드리프트 진입")
        print("  3. 반대 방향키(카운터 스티어)로 각도 조절")
        print("  4. SPACE 떼면 드리프트 종료 + 부스트!")
        print("=" * 60)
        
        while running:
            controls = {
                'forward': False,
                'backward': False,
                'left': False,
                'right': False,
                'brake': False
            }
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            controls['forward'] = keys[pygame.K_UP]
            controls['backward'] = keys[pygame.K_DOWN]
            controls['left'] = keys[pygame.K_LEFT]
            controls['right'] = keys[pygame.K_RIGHT]
            controls['brake'] = keys[pygame.K_SPACE]
            
            if keys[pygame.K_r]:
                self.reset()
            
            if not self.collision and not self.goal_reached:
                reward, done, info = self.step(controls)
                
                if done:
                    if self.goal_reached:
                        print(f"GOAL! Time: {info.get('time', 0):.2f}s")
                        print(f"Total Drifts: {info.get('drift_count', 0)}")
                        print(f"Drift Time: {info.get('drift_time', 0):.1f}s")
                    elif self.collision:
                        print("CRASHED!")
            
            self._draw()
            self.clock.tick(self.fps)
        
        pygame.quit()


if __name__ == "__main__":
    import os
    if not os.path.exists("track.json"):
        print("=" * 60)
        print("Track file not found!")
        print("Please run track_editor.py first to create a track.")
        print("=" * 60)
    else:
        game = RacingGame("track.json")
        game.run()
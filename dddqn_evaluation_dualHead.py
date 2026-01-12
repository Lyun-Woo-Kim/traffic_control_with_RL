"""
Dueling Double DQN Dual Head 모델 평가 코드 - 시각적 렌더링 포함
Action + Duration 동시 예측 버전 (Dueling 구조)
"""

import pygame
import math
import numpy as np
from racing_game_2d import RacingGame
import torch
import torch.nn as nn
import copy
import time
import re

class DuelingDualHeadNetwork(nn.Module):
    """
    Dueling DQN + Dual Head 네트워크 구조
    
    Shared Features → 
        Action Branch (Dueling): V_a(s) + A_a(s,a) → Action Q values
        Duration Branch (Dueling): V_d(s) + A_d(s,d) → Duration Q values
    """
    
    def __init__(self, input_size, action_size, duration_size, layer_num, max_size):
        super(DuelingDualHeadNetwork, self).__init__()
        
        # 공유 레이어 (Feature Extraction)
        shared_layers = []
        shared_layers.append(nn.Linear(input_size, max_size))
        shared_layers.append(nn.ReLU())
        
        current_size = max_size
        for i in range(layer_num - 1):
            next_size = current_size // 2
            shared_layers.append(nn.Linear(current_size, next_size))
            shared_layers.append(nn.ReLU())
            current_size = next_size
        
        self.shared = nn.Sequential(*shared_layers)
        self.feature_size = current_size
        
        # ============== Action Branch (Dueling) ==============
        self.action_value_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, 1)
        )
        
        self.action_advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, action_size)
        )
        
        # ============== Duration Branch (Dueling) ==============
        self.duration_value_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, 1)
        )
        
        self.duration_advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, duration_size)
        )
    
    def forward(self, x):
        features = self.shared(x)
        
        # Action Q values (Dueling)
        action_value = self.action_value_stream(features)
        action_advantage = self.action_advantage_stream(features)
        action_q = action_value + (action_advantage - action_advantage.mean(dim=1, keepdim=True))
        
        # Duration Q values (Dueling)
        duration_value = self.duration_value_stream(features)
        duration_advantage = self.duration_advantage_stream(features)
        duration_q = duration_value + (duration_advantage - duration_advantage.mean(dim=1, keepdim=True))
        
        return action_q, duration_q
    
    def get_decomposed_values(self, x):
        """V(s)와 A(s,a) 분리 반환 (시각화용)"""
        features = self.shared(x)
        
        action_value = self.action_value_stream(features)
        action_advantage = self.action_advantage_stream(features)
        
        duration_value = self.duration_value_stream(features)
        duration_advantage = self.duration_advantage_stream(features)
        
        return {
            'action_value': action_value,
            'action_advantage': action_advantage,
            'duration_value': duration_value,
            'duration_advantage': duration_advantage
        }


class DuelingDoubleDQN_DualHead:
    """Dueling Double DQN Dual Head 에이전트 (평가용)"""
    
    def __init__(self, input_size=13, action_size=6, duration_size=10, layer_num=3, max_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.duration_size = duration_size
        self.layer_num = layer_num
        self.max_size = max_size
        
        self.model = DuelingDualHeadNetwork(input_size, action_size, duration_size, layer_num, max_size).to(self.device)
        
        # Duration index → 실제 frame 수 매핑 (1~10)
        self.duration_map = {i: (i+1) * 3 for i in range(duration_size)}
    
    def load_model(self, path):
        """학습된 모델 로드"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from: {path}")
    
    def get_real_action(self, action_index): 
        """6개의 단순화된 액션을 실제 컨트롤로 변환"""
        actions = {
            0: {'forward': True, 'backward': False, 'left': False, 'right': False, 'brake': False},   # 직진
            1: {'forward': True, 'backward': False, 'left': True, 'right': False, 'brake': False},    # 직진+좌
            2: {'forward': True, 'backward': False, 'left': False, 'right': True, 'brake': False},    # 직진+우
            3: {'forward': False, 'backward': False, 'left': True, 'right': False, 'brake': True},     # 드리프트 좌
            4: {'forward': False, 'backward': False, 'left': False, 'right': True, 'brake': True},     # 드리프트 우
            5: {'forward': True, 'backward': False, 'left': True, 'right': False, 'brake': True},     # 드리프트 좌+직
            6: {'forward': True, 'backward': False, 'left': False, 'right': True, 'brake': True},     # 드리프트 우+직
            7: {'forward': False, 'backward': False, 'left': False, 'right': False, 'brake': True},   # 브레이크
        }
        return actions.get(action_index, actions[0])
    
    def get_duration_frames(self, duration_index):
        """Duration index를 실제 frame 수로 변환"""
        return self.duration_map.get(duration_index, 3)
    
    def predict(self, state):
        """Action과 Duration 동시 예측 (greedy)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_q, duration_q = self.model(state_tensor)
            
            action = action_q.argmax().item()
            duration_idx = duration_q.argmax().item()
            duration = self.get_duration_frames(duration_idx)
            
            return action_q, duration_q, action, duration_idx, duration
    
    def get_decomposed_values(self, state):
        """V(s)와 A(s,a) 분리 반환"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.model.get_decomposed_values(state_tensor)


def get_sensors(game):
    sensors = []
    sensor_range = game.car.sensor_range
    direction_num = 12
    for i in range(direction_num):
        angle = game.car.angle + (i * math.pi / (direction_num/2))
        for d in range(0, sensor_range, 5):
            x = int(game.car.x + math.cos(angle) * d)
            y = int(game.car.y + math.sin(angle) * d)
            if (x < 0 or y < 0 or 
                x >= game.track_mask.shape[1] or 
                y >= game.track_mask.shape[0] or
                game.track_mask[y, x] == 0):
                sensors.append(d)
                break
        else:
            sensors.append(sensor_range)
    return sensors

def get_data(game, standard_cp=None, dis_gap=None): 
    angle = game.car.angle
    cos_angle = (math.cos(angle) + 1) / 2
    sin_angle = (math.sin(angle) + 1) / 2
    speed = game.car.speed / game.car.max_speed
    vel_x = game.car.velocity_x / game.car.max_speed
    vel_y = game.car.velocity_y / game.car.max_speed
    sensors = [sensor / game.car.sensor_range for sensor in get_sensors(game)]
    
    if standard_cp is not None and dis_gap is not None and dis_gap > 0:
        # 체크포인트까지 거리
        current_distance = math.dist([game.car.x, game.car.y], standard_cp)
        normalized_dist = min(current_distance / dis_gap, 1.5)
        is_drifting = 1.0 if game.car.is_drifting else 0.0
        # 체크포인트 방향 (상대 각도)
        dx = standard_cp[0] - game.car.x
        dy = standard_cp[1] - game.car.y
        target_angle = math.atan2(dy, dx)
        relative_angle = target_angle - game.car.angle
        
        # -π ~ π 정규화
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # 0~1로 정규화
        normalized_angle = (relative_angle + math.pi) / (2 * math.pi)
    else:
        normalized_dist = 1.0
        normalized_angle = 0.5
    
    car_features = [
        game.car.max_speed / 1000,              # 최대속도 (0~1 정규화)
        game.car.acceleration_force / 1000,     # 가속력
        game.car.brake_force,                  # 브레이크 효율 (이미 0~1)
        game.car.base_friction,                # 마찰계수 (이미 0~1)
        game.car.lateral_friction,       # 측면 마찰 (이미 0~1)
        game.car.turn_speed / 10,               # 회전속도
        game.car.drift_lateral_friction,                # 드리프트 마찰 (이미 0~1)
        game.car.sensor_range / 1000            # 센서 범위 (0~1 정규화)
    ]
    
    return sensors + [cos_angle, sin_angle, speed, vel_x, vel_y, normalized_dist, normalized_angle, is_drifting] + car_features


def draw_sensors(screen, game, camera_offset):
    """센서 시각화"""
    sensor_range = game.car.sensor_range
    for i in range(12):
        angle = game.car.angle + (i * math.pi / 4)
        for d in range(0, sensor_range, 5):
            x = int(game.car.x + math.cos(angle) * d)
            y = int(game.car.y + math.sin(angle) * d)
            if (x < 0 or y < 0 or 
                x >= game.track_mask.shape[1] or 
                y >= game.track_mask.shape[0] or
                game.track_mask[y, x] == 0):
                end_x = game.car.x + math.cos(angle) * d - camera_offset[0]
                end_y = game.car.y + math.sin(angle) * d - camera_offset[1]
                start_x = game.car.x - camera_offset[0]
                start_y = game.car.y - camera_offset[1]
                pygame.draw.line(screen, (0, 255, 0), (start_x, start_y), (end_x, end_y), 1)
                pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)
                break
        else:
            end_x = game.car.x + math.cos(angle) * sensor_range - camera_offset[0]
            end_y = game.car.y + math.sin(angle) * sensor_range - camera_offset[1]
            start_x = game.car.x - camera_offset[0]
            start_y = game.car.y - camera_offset[1]
            pygame.draw.line(screen, (0, 255, 0), (start_x, start_y), (end_x, end_y), 1)


def draw_dueling_dual_head_values(screen, decomposed, action_q, duration_q, 
                                   current_action, current_duration_idx, font):
    """
    Dueling DQN Dual Head의 V(s), A(s,a), Q(s,a) 시각화
    - Action Branch와 Duration Branch의 값들을 화면에 표시
    - 현재 선택된 action과 duration을 하이라이트
    """
    action_names = ['Forward', 'Fwd+Left', 'Fwd+Right', 'Drift L', 'Drift R', 'Brake']
    
    # 값 추출
    action_v = decomposed['action_value'].cpu().numpy().flatten()[0]
    action_a = decomposed['action_advantage'].cpu().numpy().flatten()
    action_q_vals = action_q.cpu().numpy().flatten()
    
    duration_v = decomposed['duration_value'].cpu().numpy().flatten()[0]
    duration_a = decomposed['duration_advantage'].cpu().numpy().flatten()
    duration_q_vals = duration_q.cpu().numpy().flatten()
    
    x_start = 10
    y_start = 100
    bar_width = 60
    bar_height = 14
    
    # ===== Action Branch =====
    # V(s) for Action
    v_text = font.render(f"Action V(s) = {action_v:.2f}", True, (0, 0, 150))
    screen.blit(v_text, (x_start, y_start - 35))
    
    title = font.render("A(s,a) | Q(s,a):", True, (0, 0, 100))
    screen.blit(title, (x_start, y_start - 18))
    
    max_a = max(action_a)
    min_a = min(action_a)
    max_q = max(action_q_vals)
    min_q = min(action_q_vals)
    
    for i, (name, a_val, q_val) in enumerate(zip(action_names, action_a, action_q_vals)):
        y = y_start + i * (bar_height + 4)
        
        # Advantage 바 (파란색)
        pygame.draw.rect(screen, (200, 200, 230), (x_start, y, bar_width, bar_height))
        if max_a != min_a:
            norm_a = (a_val - min_a) / (max_a - min_a)
        else:
            norm_a = 0.5
        
        a_color = (0, 100, 200) if i == current_action else (100, 100, 200)
        pygame.draw.rect(screen, a_color, (x_start, y, int(bar_width * norm_a), bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (x_start, y, bar_width, bar_height), 1)
        
        # Q-value 바 (녹색)
        q_x = x_start + bar_width + 3
        pygame.draw.rect(screen, (200, 230, 200), (q_x, y, bar_width, bar_height))
        if max_q != min_q:
            norm_q = (q_val - min_q) / (max_q - min_q)
        else:
            norm_q = 0.5
        
        q_color = (0, 200, 0) if i == current_action else (100, 200, 100)
        pygame.draw.rect(screen, q_color, (q_x, y, int(bar_width * norm_q), bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (q_x, y, bar_width, bar_height), 1)
        
        # 텍스트
        marker = "►" if i == current_action else " "
        text = font.render(f"{marker}{name[:6]}: {a_val:.1f}|{q_val:.1f}", True, (0, 0, 0))
        screen.blit(text, (q_x + bar_width + 5, y))
    
    # ===== Duration Branch =====
    duration_y_start = y_start + 6 * (bar_height + 4) + 25
    
    # V(s) for Duration
    v_dur_text = font.render(f"Duration V(s) = {duration_v:.2f}", True, (0, 100, 0))
    screen.blit(v_dur_text, (x_start, duration_y_start - 35))
    
    title_dur = font.render("Duration A(s,d) | Q(s,d):", True, (0, 100, 0))
    screen.blit(title_dur, (x_start, duration_y_start - 18))
    
    max_d_a = max(duration_a)
    min_d_a = min(duration_a)
    max_d_q = max(duration_q_vals)
    min_d_q = min(duration_q_vals)
    
    # Duration은 2행으로 표시 (5개씩)
    small_bar_width = 35
    for i, (a_val, q_val) in enumerate(zip(duration_a, duration_q_vals)):
        row = i // 5
        col = i % 5
        
        x = x_start + col * (small_bar_width * 2 + 30)
        y = duration_y_start + row * (bar_height + 20)
        
        # Advantage 바
        pygame.draw.rect(screen, (200, 200, 230), (x, y, small_bar_width, bar_height))
        if max_d_a != min_d_a:
            norm_a = (a_val - min_d_a) / (max_d_a - min_d_a)
        else:
            norm_a = 0.5
        
        a_color = (0, 100, 200) if i == current_duration_idx else (100, 100, 200)
        pygame.draw.rect(screen, a_color, (x, y, int(small_bar_width * norm_a), bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, small_bar_width, bar_height), 1)
        
        # Q-value 바
        q_x = x + small_bar_width + 2
        pygame.draw.rect(screen, (200, 230, 200), (q_x, y, small_bar_width, bar_height))
        if max_d_q != min_d_q:
            norm_q = (q_val - min_d_q) / (max_d_q - min_d_q)
        else:
            norm_q = 0.5
        
        q_color = (0, 200, 0) if i == current_duration_idx else (100, 200, 100)
        pygame.draw.rect(screen, q_color, (q_x, y, int(small_bar_width * norm_q), bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (q_x, y, small_bar_width, bar_height), 1)
        
        # 프레임 수 표시
        marker = "►" if i == current_duration_idx else ""
        text = font.render(f"{marker}{i+1}f", True, (0, 0, 0))
        screen.blit(text, (x, y + bar_height + 1))


def parse_model_name(model_name):
    """
    모델 이름에서 파라미터 추출
    """
    pattern = r"_L(\d+)_S(\d+)_E(\d+)"
    match = re.search(pattern, model_name)
    
    if match:
        return {
            'layer_num': int(match.group(1)),
            'max_size': int(match.group(2)),
            'save_episode': int(match.group(3))
        }
    else:
        raise ValueError(f"파싱 실패: {model_name}")


def evaluate_dddqn_dual_head(model_path, track_file, car_json_path, layer_num=3, max_size=512,
                              action_size=6, duration_size=10,
                              num_episodes=5, show_sensors=True, show_values=True, speed_multiplier=1.0):
    """
    Dueling Double DQN Dual Head 모델 평가 함수
    - 학습된 모델을 로드하여 시각적 렌더링과 함께 평가
    - 센서 및 Q값 시각화 옵션 제공
    - 성공률 및 평균 시간 계산
    """
    pygame.init()
    
    # 모델 파일명에서 파라미터 추출 시도
    try:
        model_infos = parse_model_name(model_path)
        layer_num = model_infos['layer_num']
        max_size = model_infos['max_size']
    except:
        print(f"모델 파일명 파싱 실패, 기본값 사용: layer_num={layer_num}, max_size={max_size}")
    
    # 게임 초기화 (렌더링 모드)
    game = RacingGame(track_file, car_json_path=car_json_path, headless=False)
    ori_checkpoints = copy.deepcopy(game.checkpoints)
    
    # 에이전트 초기화 및 모델 로드
    agent = DuelingDoubleDQN_DualHead(input_size=28, action_size=action_size, duration_size=duration_size,
                                       layer_num=layer_num, max_size=max_size)
    agent.load_model(model_path)
    
    # 폰트
    font = pygame.font.SysFont('arial,sans-serif', 12)
    big_font = pygame.font.SysFont('arial,sans-serif', 36)
    
    print("=" * 60)
    print("🎮 DUELING DOUBLE DQN DUAL HEAD MODEL EVALUATION")
    print(f"   Model: {model_path}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Speed: {speed_multiplier}x")
    print(f"   Action Size: {action_size}, Duration Size: {duration_size}")
    print("=" * 60)
    print("\nControls:")
    print("  ESC: Quit")
    print("  R: Reset current episode")
    print("  S: Toggle sensors")
    print("  V: Toggle V/A/Q values display")
    print("  +/-: Adjust speed")
    print("=" * 60)
    
    episode = 0
    results = []
    running = True
    
    while running and episode < num_episodes:
        # 에피소드 시작
        game.reset()
        game.checkpoints = copy.deepcopy(ori_checkpoints)
        
        done = False
        current_action = 0
        current_duration_idx = 0
        current_duration = 1
        current_action_q = None
        current_duration_q = None
        current_decomposed = None
        frames_remaining = 0
        current_controls = agent.get_real_action(0)
        
        print(f"\n▶ Episode {episode + 1}/{num_episodes} started")
        standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
        dis_gap = math.dist([game.start_pos[0], game.start_pos[1]], 
                            [standard_cp[0], standard_cp[1]])
        
        while not done and running:
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        done = True
                        episode -= 1
                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    elif event.key == pygame.K_v:
                        show_values = not show_values
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        speed_multiplier = min(speed_multiplier + 0.5, 5.0)
                        print(f"Speed: {speed_multiplier}x")
                    elif event.key == pygame.K_MINUS:
                        speed_multiplier = max(speed_multiplier - 0.5, 0.5)
                        print(f"Speed: {speed_multiplier}x")
            
            # 프레임 스킵이 끝나면 새 액션 선택
            if frames_remaining <= 0:
                state = get_data(game, standard_cp, dis_gap)
                current_action_q, current_duration_q, current_action, current_duration_idx, current_duration = agent.predict(state)
                current_decomposed = agent.get_decomposed_values(state)
                frames_remaining = current_duration
                current_controls = agent.get_real_action(current_action)
            
            # 게임 스텝
            _, step_done, info = game.step(current_controls)
            frames_remaining -= 1
            
            # 체크포인트 처리
            if len(game.checkpoints_reached) > 0:
                    cp_idx = game.checkpoints_reached.pop(0)
                    reached_cp = game.checkpoints[cp_idx]
                    game.checkpoints.pop(cp_idx)
                    ori_cp_idx = ori_checkpoints.index(reached_cp)
                    current_segment = ori_cp_idx + 1
                    standard_cp = ori_checkpoints[ori_cp_idx + 1] if ori_cp_idx < len(ori_checkpoints) - 1 else game.end_pos
            
            # 종료 조건
            if step_done:
                done = True
            
            # 타임아웃
            if game.current_time / 1000 > 60:
                done = True
            
            # 렌더링
            game._draw()
            
            # 추가 시각화
            if show_sensors:
                draw_sensors(game.screen, game, (game.camera_x, game.camera_y))
            
            if show_values and current_action_q is not None:
                draw_dueling_dual_head_values(game.screen, current_decomposed, 
                                               current_action_q, current_duration_q,
                                               current_action, current_duration_idx, font)
            
            # 에피소드 정보 표시
            episode_text = big_font.render(f"Episode: {episode + 1}/{num_episodes}", True, (0, 0, 0))
            game.screen.blit(episode_text, (game.width - 250, 10))
            
            # action_names = ['Forward', 'Fwd+Left', 'Fwd+Right', 'Drift L', 'Drift R', 'Brake']
            # action_text = font.render(f"Action: {action_names[current_action]}", True, (0, 0, 0))
            # game.screen.blit(action_text, (game.width - 250, 50))
            
            duration_text = font.render(f"Duration: {current_duration} frames", True, (0, 0, 0))
            game.screen.blit(duration_text, (game.width - 250, 70))
            
            speed_text = font.render(f"Game Speed: {speed_multiplier}x", True, (0, 0, 0))
            game.screen.blit(speed_text, (game.width - 250, 90))
            
            # 알고리즘 표시
            algo_text = font.render("Dueling Double DQN Dual Head", True, (150, 0, 150))
            game.screen.blit(algo_text, (game.width - 250, 110))
            
            pygame.display.flip()
            game.clock.tick(int(60 * speed_multiplier))
        
        # 에피소드 결과 기록
        if running:
            episode += 1
            episode_time = game.current_time / 1000
            
            result = {
                'episode': episode,
                'goal': game.goal_reached,
                'collision': game.collision,
                'time': episode_time
            }
            results.append(result)
            
            if game.goal_reached:
                print(f"  ✓ GOAL! Time: {episode_time:.2f}s")
            elif game.collision:
                print(f"  ✗ CRASHED at {episode_time:.2f}s")
            else:
                print(f"  ✗ TIMEOUT at {episode_time:.2f}s")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS (Dueling Double DQN Dual Head)")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['goal'])
    success_times = [r['time'] for r in results if r['goal']]
    
    print(f"Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    
    if success_times:
        print(f"Average Goal Time: {np.mean(success_times):.3f}s")
        print(f"Best Time: {min(success_times):.3f}s")
        print(f"Worst Time: {max(success_times):.3f}s")
    
    print("=" * 60)
    
    pygame.quit()
    return results


if __name__ == "__main__":
    # 설정
    MODEL_PATH = "dddqn_dualhead_best_lr_0_001_L3_S2048_E121290_T135_026.pth"  # 모델 파일 경로
    TRACK_FILE = "track.json"
    CAR_JSON_PATH = "racing_car.json"
    
    # 평가 실행
    results = evaluate_dddqn_dual_head(
        model_path=MODEL_PATH,
        track_file=TRACK_FILE,
        car_json_path=CAR_JSON_PATH,
        layer_num=3,
        max_size=512,
        action_size=8,
        duration_size=6,
        num_episodes=10,
        show_sensors=True,
        show_values=True,
        speed_multiplier=1.0
    )
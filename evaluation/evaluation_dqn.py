"""
DQN 모델 평가 코드 - 시각적 렌더링 포함
"""

import pygame
import math
import numpy as np
import torch
import torch.nn as nn
import copy
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

# 이제 상대 경로(..) 없이 바로 env에서 임포트 가능합니다.
from env.racing_game_2d import RacingGame


# 평가용 DQN Class
class DQN:
    def __init__(self, input_size=28, output_size=6, layer_num=3, max_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.layer_num = layer_num
        self.max_size = max_size
        
        self.model = self.create_model(input_size, output_size, layer_num, max_size).to(self.device)
        
        # 6개 액션 + 고정된 frame 수 매핑
        self.action_repeat_map = {
            0: 5,   # 직진
            1: 5,   # 직진+좌
            2: 5,   # 직진+우
            3: 5,  # 직진+브레이크+좌 (드리프트)
            4: 5,  # 직진+브레이크+우 (드리프트)
            5: 2,   # 브레이크
        }
    
    # 모델 생성 (DQN 이므로 단순한 layer 나열)    
    def create_model(self, input_size, output_size, layer_num, max_size):
        layers = []
        
        layers.append(nn.Linear(input_size, max_size))
        layers.append(nn.ReLU())
        
        current_size = max_size
        for i in range(layer_num - 1):
            next_size = current_size // 2
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        return nn.Sequential(*layers)
    
    # 모델 불러오기
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from: {path}")
    
    def get_real_action(self, action_index): 
        """6개의 단순화된 액션을 실제 컨트롤로 변환"""
        actions = {
            # 직진
            0: {'forward': True, 'backward': False, 'left': False, 'right': False, 'brake': False},
            # 직진 + 좌
            1: {'forward': True, 'backward': False, 'left': True, 'right': False, 'brake': False},
            # 직진 + 우
            2: {'forward': True, 'backward': False, 'left': False, 'right': True, 'brake': False},
            # 직진 + 브레이크 + 좌 (드리프트 액션)
            3: {'forward': True, 'backward': False, 'left': True, 'right': False, 'brake': True},
            # 직진 + 브레이크 + 우 (드리프트 액션)
            4: {'forward': True, 'backward': False, 'left': False, 'right': True, 'brake': True},
            # 브레이크
            5: {'forward': False, 'backward': False, 'left': False, 'right': False, 'brake': True},
        }
        return actions.get(action_index, actions[0])
    
    def get_action_frames(self, action_index):
        """액션에 해당하는 프레임 수 반환"""
        return self.action_repeat_map.get(action_index, 3)
    
    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values, q_values.argmax().item()

# 센서 데이터 추출
def get_sensors(game):
    sensors = []
    sensor_range = game.car.sensor_range
    direction_num = 12
    # 12 방향에 해당하는 거리 계산. (sensor_range 만큼)
    for i in range(direction_num):
        angle = game.car.angle + (i * math.pi / (direction_num/2))
        # 벽까지의 거리또한 계산하기 위해 5씩 계산 진행.
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

# State 데이터 추출
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
        is_drifting = 0.0
    
    car_features = [
        game.car.max_speed / 1000,              # 최대속도 (0~1 정규화)
        game.car.acceleration_force / 1000,     # 가속력
        game.car.brake_force / 1000, 
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
                # 센서 끝점 그리기
                end_x = game.car.x + math.cos(angle) * d - camera_offset[0]
                end_y = game.car.y + math.sin(angle) * d - camera_offset[1]
                start_x = game.car.x - camera_offset[0]
                start_y = game.car.y - camera_offset[1]
                pygame.draw.line(screen, (0, 255, 0), (start_x, start_y), (end_x, end_y), 1)
                pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)
                break
        else:
            # 최대 거리까지 도달
            end_x = game.car.x + math.cos(angle) * sensor_range - camera_offset[0]
            end_y = game.car.y + math.sin(angle) * sensor_range - camera_offset[1]
            start_x = game.car.x - camera_offset[0]
            start_y = game.car.y - camera_offset[1]
            pygame.draw.line(screen, (0, 255, 0), (start_x, start_y), (end_x, end_y), 1)


def draw_q_values(screen, q_values, current_action, font):
    """
    Q-values 시각화
    - 각 action에 대한 Q값을 바 차트로 표시
    - 현재 선택된 action을 하이라이트
    """
    action_names = ['Forward', 'Fwd+Left', 'Fwd+Right', 'Drift L', 'Drift R', 'Brake']
    
    q_vals = q_values.cpu().numpy().flatten()
    max_q = max(q_vals)
    min_q = min(q_vals)
    
    x_start = 10
    y_start = 150
    bar_width = 100
    bar_height = 20
    
    title = font.render("Q-Values:", True, (0, 0, 0))
    screen.blit(title, (x_start, y_start - 25))
    
    for i, (name, q_val) in enumerate(zip(action_names, q_vals)):
        y = y_start + i * (bar_height + 5)
        
        # 배경 바
        pygame.draw.rect(screen, (200, 200, 200), (x_start, y, bar_width, bar_height))
        
        # Q-value 바
        if max_q != min_q:
            normalized = (q_val - min_q) / (max_q - min_q)
        else:
            normalized = 0.5
        
        if i == current_action:
            color = (0, 200, 0)  # 선택된 액션은 녹색
        else:
            color = (100, 100, 255)
        
        pygame.draw.rect(screen, color, (x_start, y, int(bar_width * normalized), bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (x_start, y, bar_width, bar_height), 1)
        
        # 액션 이름과 Q-value
        text = font.render(f"{name}: {q_val:.1f}", True, (0, 0, 0))
        screen.blit(text, (x_start + bar_width + 10, y))

import re

def parse_model_name(model_name):
    """
    모델 이름에서 파라미터 추출
    
    Args:
        model_name (str): 모델 파일 이름
        
    Returns:
        dict: layer_num, max_size, save_episode를 담은 딕셔너리
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

def evaluate_dqn(model_path, track_file, car_json_path, layer_num=3, max_size=512, 
                 num_episodes=5, show_sensors=True, show_q_values=True, speed_multiplier=1.0):
    """
    DQN 모델 평가 함수
    - 학습된 모델을 로드하여 시각적 렌더링과 함께 평가
    - 센서 및 Q값 시각화 옵션 제공
    - 성공률 및 평균 시간 계산
    
    Args:
        model_path: 학습된 모델 파일 경로
        track_file: 트랙 JSON 파일 경로
        car_json_path: 차량 설정 JSON 파일 경로
        layer_num: 모델 레이어 수
        max_size: 모델 최대 hidden size
        num_episodes: 평가할 에피소드 수
        show_sensors: 센서 시각화 여부
        show_q_values: Q-values 시각화 여부
        speed_multiplier: 게임 속도 배율 (1.0 = 정상, 2.0 = 2배속)
    """
    pygame.init()
    
    model_infos = parse_model_name(model_path)
    layer_num = model_infos['layer_num']
    max_size = model_infos['max_size']
    
    # 게임 초기화 (렌더링 모드)
    game = RacingGame(track_file, car_json_path=car_json_path, headless=False)
    ori_checkpoints = copy.deepcopy(game.checkpoints)
    
    # 에이전트 초기화 및 모델 로드
    agent = DQN(input_size=28, output_size=6, layer_num=layer_num, max_size=max_size)
    agent.load_model(model_path)
    
    # 폰트
    font = pygame.font.SysFont('arial,sans-serif', 18)
    big_font = pygame.font.SysFont('arial,sans-serif', 36)
    
    print("=" * 60)
    print("🎮 DQN MODEL EVALUATION")
    print(f"   Model: {model_path}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Speed: {speed_multiplier}x")
    print("=" * 60)
    print("\nControls:")
    print("  ESC: Quit")
    print("  R: Reset current episode")
    print("  S: Toggle sensors")
    print("  Q: Toggle Q-values")
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
                        done = True  # 에피소드 재시작
                        episode -= 1
                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    elif event.key == pygame.K_q:
                        show_q_values = not show_q_values
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        speed_multiplier = min(speed_multiplier + 0.5, 5.0)
                        print(f"Speed: {speed_multiplier}x")
                    elif event.key == pygame.K_MINUS:
                        speed_multiplier = max(speed_multiplier - 0.5, 0.5)
                        print(f"Speed: {speed_multiplier}x")
            
            # 프레임 스킵이 끝나면 새 액션 선택
            if frames_remaining <= 0:
                state = get_data(game, standard_cp, dis_gap)
                current_q_values, current_action = agent.predict(state)
                frames_remaining = agent.get_action_frames(current_action)
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
                dis_gap = math.dist([reached_cp[0], reached_cp[1]], 
                                    [standard_cp[0], standard_cp[1]])
            
            # 종료 조건
            if step_done:
                done = True
            
            # 타임아웃
            if game.current_time / 1000 > 60:
                done = True
            
            # 렌더링
            game._draw()
            
            # 추가 시각화
            # if show_sensors:
            #     draw_sensors(game.screen, game, (game.camera_x, game.camera_y))
            
            if show_q_values and current_q_values is not None:
                draw_q_values(game.screen, current_q_values, current_action, font)
            
            # 에피소드 정보 표시
            episode_text = big_font.render(f"Episode: {episode + 1}/{num_episodes}", True, (0, 0, 0))
            game.screen.blit(episode_text, (game.width - 250, 10))
            
            action_names = ['Forward', 'Fwd+Left', 'Fwd+Right', 'Drift L', 'Drift R', 'Brake']
            action_text = font.render(f"Action: {action_names[current_action]}", True, (0, 0, 0))
            game.screen.blit(action_text, (game.width - 250, 50))
            
            speed_text = font.render(f"Game Speed: {speed_multiplier}x", True, (0, 0, 0))
            game.screen.blit(speed_text, (game.width - 250, 70))
            
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
    print("📊 EVALUATION RESULTS")
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
    MODEL_PATH = f"{os.path.join(current_dir, '..')}/example_models/dqn_best_lr_0_0003_L3_S512_E165054_T133_448.pth"  # 모델 파일 경로
    TRACK_FILE = f"{os.path.join(current_dir, '..')}/env/track.json"
    CAR_JSON_PATH = f"{os.path.join(current_dir, '..')}/env/racing_car.json"
    
    # 모델 파일명에서 하이퍼파라미터 추출 (필요시 수정)
    LAYER_NUM = 3
    MAX_SIZE = 512
    
    # 평가 실행
    results = evaluate_dqn(
        model_path=MODEL_PATH,
        track_file=TRACK_FILE,
        car_json_path=CAR_JSON_PATH,
        layer_num=LAYER_NUM,
        max_size=MAX_SIZE,
        num_episodes=10,
        show_sensors=False,
        show_q_values=True,
        speed_multiplier=1.0
    )
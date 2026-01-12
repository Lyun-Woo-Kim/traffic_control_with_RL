"""
레이싱 게임 에이전트 - Headless 학습 버전 (렌더링 없이 빠른 학습)
Action 단순화 버전: 6개 액션으로 friction 기반 제어
[수정됨] Frame skipping 방식 개선 - action 단위로 1개의 transition만 저장
[수정됨] 모델 저장 기준 - 10회 테스트 100% 성공 + 평균 시간 비교
"""

import pygame
import random
import math
import numpy as np
from racing_game_2d import RacingGame 
import torch
import torch.nn as nn
import copy
import time
import os

class DQN:
    def __init__(self, input_size=28, output_size=6, replay_memory_length=100000, num_segments=5, lr=0.0001, layer_num=3, max_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.layer_num = layer_num
        self.max_size = max_size
        
        self.model = self.create_model(input_size, 
                                       output_size, 
                                       self.layer_num, 
                                       self.max_size
                                       ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.gamma = 0.99
        self.epsilon_min = 0.01
        
        self.segment_epsilon = {i: 1.0 for i in range(num_segments)}
        self.segment_decay = 0.995
        self.learned_epsilon = 0.1
        self.epsilon = 1.0
        
        self.replay_memory = []
        self.replay_memory_length = replay_memory_length
        
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
    
    # Epsilon 값 업데이트
    def update_epsilon(self, current_segment, learned_until):
        if current_segment <= learned_until:
            self.epsilon = self.learned_epsilon
        else:
            self.epsilon = self.segment_epsilon.get(current_segment, 1.0)
    
    def decay_segment_epsilon(self, segment):
        if segment in self.segment_epsilon:
            self.segment_epsilon[segment] = max(
                self.segment_epsilon[segment] * self.segment_decay,
                self.epsilon_min
            )
    
    def get_real_action(self, action_index): 
        # 6개 액션 + 고정된 frame 수 매핑
        actions = {
            # 직진
            0: {'forward': True, 'backward': False, 'left': False, 'right': False, 'brake': False},
            # 직진 + 좌
            1: {'forward': True, 'backward': False, 'left': True, 'right': False, 'brake': False},
            # 직진 + 우
            2: {'forward': True, 'backward': False, 'left': False, 'right': True, 'brake': False},
            # 직진 + 브레이크 + 좌 (드리프트)
            3: {'forward': True, 'backward': False, 'left': True, 'right': False, 'brake': True},
            # 직진 + 브레이크 + 우 (드리프트)
            4: {'forward': True, 'backward': False, 'left': False, 'right': True, 'brake': True},
            # 브레이크
            5: {'forward': False, 'backward': False, 'left': False, 'right': False, 'brake': True},
        }
        return actions.get(action_index, actions[0])
    
    def get_action_frames(self, action_index):
        """각 액션에 해당하는 프레임 수 반환"""
        return self.action_repeat_map.get(action_index, 3)
    
    def predict(self, state, greedy=False):
        """액션 예측 (epsilon-greedy 또는 greedy)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            if not greedy and random.random() < self.epsilon: 
                return q_values, random.randint(0, self.output_size - 1)
            else: 
                return q_values, q_values.argmax().item()
    
    def get_frame_reward(self, state, coll, goal, curr_distance, curr_time, max_time, 
                         cp_r, dis_gap, action_index, max_speed, prev_distance): 
        """
        단일 프레임에 대한 보상 계산
        - 충돌/골 도달 등 이벤트 보상
        - 진행 거리 보상
        - 속도 보상 및 저속 패널티
        - 드리프트 보너스 (고속일 때만)
        - 체크포인트 보상 (속도에 따라 가중치 적용)
        """
        speed_norm = state[14]
        is_drifting = state[19] > 0.5
        is_turning = action_index in [0, 1, 2]
        is_brake_only = action_index == 5
        
        # 기본 이벤트 보상
        if coll: 
            r = -200
        elif goal: 
            time_ratio = curr_time / (max_time * 1000)
            r = 100 + (1 - time_ratio) ** 2 * 500.0
        else: 
            r = 0
        
        # 진행 보상: 이전 프레임 대비 목표 체크포인트까지 거리 감소량
        r_dis = (prev_distance - curr_distance) * 0.1
        
        # 시간 패널티: 매 프레임마다 소량의 패널티
        r_time = 0.01
        
        # 속도 보상: 정규화된 속도에 비례
        r_speed = speed_norm * 0.3
        
        # 저속 패널티: 너무 느리게 주행 시 추가 패널티
        if speed_norm < 0.15:
            r_speed -= 0.5
        
        # 드리프트 보너스: 고속에서 드리프트 시 보너스, 저속에서는 패널티
        drift_bonus = 0
        if is_drifting:
            if speed_norm > 0.4:
                drift_bonus = speed_norm * 1.0
            else:
                drift_bonus = -0.2
        
        # 브레이크만 밟고 저속일 때 패널티
        if is_brake_only and speed_norm < 0.15:
            r_speed -= 0.5
        
        # 체크포인트 보상: 속도와 드리프트 상태에 따라 가중치 적용
        if cp_r > 0:
            if is_drifting and speed_norm > 0.4:
                cp_r *= 3.0
            elif speed_norm > 0.3:
                cp_r *= 2.0
            elif speed_norm < 0.15:
                cp_r *= 0.3
        
        reward = r + r_dis + r_speed - r_time + cp_r + drift_bonus
        return reward
    
    def add_memory(self, memory): 
        if len(self.replay_memory) < self.replay_memory_length: 
            self.replay_memory.append(memory)
        else:
            self.replay_memory.pop(0)
            self.replay_memory.append(memory)
    
    def train_step(self, batch_size, target_net):
        """
        DQN 학습 스텝
        - Standard DQN 방식으로 학습
        - Target network를 사용하여 안정적인 학습
        """
        # Replay memory에서 랜덤 샘플링
        batch = random.sample(self.replay_memory, batch_size)
        
        # Memory format: [state, action, reward, next_state, done]
        states = torch.tensor([m[0] for m in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([m[1] for m in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([m[2] for m in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([m[3] for m in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([m[4] for m in batch], dtype=torch.float32).to(self.device)
        
        # 현재 상태에서의 Q값 계산
        Q_values = self.model(states)
        Q_currents = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q값 계산
        with torch.no_grad():
            next_Q = target_net.model(next_states)
            max_next_Q = torch.max(next_Q, dim=1)[0]
            # Bellman equation으로 target 계산
            Q_targets = rewards + self.gamma * max_next_Q * (1 - dones)
        
        # Loss 계산
        loss = self.loss_fn(Q_currents, Q_targets)
        
        # 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss.item()


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
        game.car.max_speed / 1000,              # 최대속도
        game.car.acceleration_force / 1000,     # 가속력
        game.car.brake_force,                  # 브레이크 효율 
        game.car.base_friction,                # 마찰계수
        game.car.lateral_friction,       # 측면 마찰
        game.car.turn_speed / 10,               # 회전속도
        game.car.drift_lateral_friction,                # 드리프트 마찰
        game.car.sensor_range / 1000            # 센서 범위
    ]
    
    return sensors + [cos_angle, sin_angle, speed, vel_x, vel_y, normalized_dist, normalized_angle, is_drifting] + car_features


def execute_action_with_frame_skip(game, policy_net, action, ori_checkpoints, 
                                    current_segment, standard_cp, dis_gap, max_time):
    """
    Action을 지정된 frame 수만큼 실행하고 결과를 반환
    
    Args:
        game: RacingGame 인스턴스 - 게임 상태 및 물리 시뮬레이션
        policy_net: DQN 인스턴스 - 정책 네트워크 (액션 프레임 수, 실제 액션 변환, 리워드 계산)
        action: int - 선택된 액션 인덱스 (0~5)
        ori_checkpoints: list - 원본 체크포인트 리스트 (체크포인트 도달 시 다음 목표 결정용)
        current_segment: int - 현재 트랙 세그먼트 인덱스 (체크포인트 기준으로 트랙을 구간으로 나눈 것)
        standard_cp: tuple - 현재 목표 체크포인트 좌표 (x, y)
        dis_gap: float - 현재 위치에서 목표 체크포인트까지의 거리
        max_time: float - 에피소드 최대 시간 (초 단위)
    
    Returns:
        dict - 액션 실행 결과 딕셔너리
    """
    # 액션에 해당하는 프레임 수 (액션별로 다른 프레임 수 적용)
    frames = policy_net.get_action_frames(action)
    # 액션 인덱스를 실제 게임 제어 명령으로 변환 (forward, left, right, brake 등)
    controls = policy_net.get_real_action(action)
    
    # 누적 리워드 (모든 프레임의 리워드 합계)
    total_reward = 0
    # 에피소드 종료 여부 (골 도달, 충돌, 타임아웃 중 하나)
    done = False
    # 골 도달 여부
    is_goal = False
    # 충돌 발생 여부
    is_collision = False
    # 타임아웃 발생 여부
    is_timeout = False
    # 도달한 세그먼트 인덱스 (체크포인트 도달 시 설정, None이면 미도달)
    segment_reached = None
    
    # 현재 세그먼트 인덱스 (체크포인트 도달 시 업데이트됨)
    curr_segment = current_segment
    # 현재 목표 체크포인트 좌표 (체크포인트 도달 시 다음 체크포인트로 업데이트)
    curr_standard_cp = standard_cp
    # 현재 위치에서 목표 체크포인트까지의 거리 (매 프레임 업데이트)
    curr_dis_gap = dis_gap
    
    # 각 프레임의 속도 값을 저장하는 리스트 (평균 속도 계산용)
    speed_list = []
    
    # 액션에 지정된 프레임 수만큼 반복 실행
    for frame_idx in range(frames):
        # 충돌이나 골 도달 시 즉시 종료
        if game.collision or game.goal_reached:
            break
        
        # 이전 거리 계산 (리워드 계산용)
        prev_distance = math.dist([game.car.x, game.car.y], 
                                   [curr_standard_cp[0], curr_standard_cp[1]])
            
        # 게임 한 프레임 진행 (controls에 따라 차량 제어)
        _, step_done, info = game.step(controls)
        # 현재 게임 경과 시간 (밀리초 단위)
        curr_time = game.current_time
        
        # 현재 차량 위치에서 목표 체크포인트까지의 유클리드 거리
        curr_distance = math.dist([game.car.x, game.car.y], 
                                   [curr_standard_cp[0], curr_standard_cp[1]])
        
        # 현재 게임 상태를 벡터로 변환 (센서값, 각도, 속도 등)
        frame_state = get_data(game, curr_standard_cp, curr_dis_gap)
        # 정규화된 속도 값 저장 (frame_state[14]은 정규화된 속도)
        speed_list.append(frame_state[14])
        
        # 타임아웃 여부 확인 (경과 시간이 max_time 초를 초과했는지)
        timeout = curr_time / 1000 > max_time
        
        # 체크포인트 보상 (기본값 0, 체크포인트 도달 시 300)
        cp_r = 0
        # 체크포인트 도달 여부 확인
        if len(game.checkpoints_reached) > 0:
            cp_r = 50  # 체크포인트 도달 보상
            # 도달한 체크포인트 인덱스 (게임 내부 체크포인트 리스트 기준)
            cp_idx = game.checkpoints_reached.pop(0)
            # 도달한 체크포인트 좌표
            reached_cp = game.checkpoints[cp_idx]
            # 도달한 체크포인트를 리스트에서 제거
            game.checkpoints.pop(cp_idx)
            # 원본 체크포인트 리스트에서의 인덱스
            ori_cp_idx = ori_checkpoints.index(reached_cp)
            
            # 도달한 세그먼트 인덱스 저장 (학습 통계용)
            segment_reached = curr_segment
            
            # 다음 세그먼트로 업데이트 (ori_cp_idx + 1)
            curr_segment = ori_cp_idx + 1
            # 다음 목표 체크포인트 설정 (마지막 체크포인트면 골 위치로)
            curr_standard_cp = ori_checkpoints[ori_cp_idx + 1] if ori_cp_idx < len(ori_checkpoints) - 1 else game.end_pos
            # 도달한 체크포인트에서 다음 목표까지의 거리 계산
            curr_dis_gap = math.dist([reached_cp[0], reached_cp[1]], 
                                      [curr_standard_cp[0], curr_standard_cp[1]])
        
        # 게임 스텝 종료 여부 확인
        if step_done:
            if game.goal_reached:
                is_goal = True
                done = True
            elif game.collision:
                is_collision = True
                done = True
        
        # 타임아웃 처리
        if timeout:
            is_timeout = True
            cp_r = -300  # 타임아웃 페널티
            done = True
        
        # 현재 프레임의 리워드 계산
        frame_reward = policy_net.get_frame_reward(
            frame_state, is_collision, is_goal, curr_distance,
            curr_time, max_time, cp_r, curr_dis_gap, action, game.car.max_speed, prev_distance
        )
        # 누적 리워드에 추가
        total_reward += frame_reward
        
        # 종료 조건 만족 시 루프 탈출
        if done:
            break
    
    # 액션 실행 후 최종 상태 벡터 (다음 상태로 사용됨)
    next_state = get_data(game, curr_standard_cp, curr_dis_gap)
    
    return {
        'total_reward': total_reward,  # 액션 실행 중 누적된 총 리워드
        'next_state': next_state,  # 액션 실행 후의 게임 상태 벡터 (다음 상태)
        'done': done,  # 에피소드 종료 여부 (골/충돌/타임아웃)
        'current_segment': curr_segment,  # 업데이트된 현재 세그먼트 인덱스
        'standard_cp': curr_standard_cp,  # 업데이트된 목표 체크포인트 좌표
        'dis_gap': curr_dis_gap,  # 업데이트된 목표까지의 거리
        'segment_reached': segment_reached,  # 도달한 세그먼트 인덱스 (None이면 미도달)
        'is_goal': is_goal,  # 골 도달 여부
        'is_collision': is_collision,  # 충돌 발생 여부
        'is_timeout': is_timeout,  # 타임아웃 발생 여부
        'curr_time': game.current_time,  # 현재 게임 경과 시간 (밀리초)
        'speed_list': speed_list  # 각 프레임의 정규화된 속도 값 리스트
    }


def evaluate_model(policy_net, game, ori_checkpoints, max_time, num_tests=10):
    """
    [새로운 함수] 모델 평가 - greedy policy로 num_tests회 테스트
    
    Returns:
        success: 모든 테스트 성공 여부
        avg_speed: 성공한 경우 평균 스피드, 실패 시 None
        success_count: 성공 횟수
    """
    policy_net.model.eval()  # 평가 모드
    
    goal_times = []
    success_count = 0
    
    all_speed_list = []
    
    for test_idx in range(num_tests):
        # 게임 리셋
        game.reset()
        game.checkpoints = copy.deepcopy(ori_checkpoints)
        
        standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
        dis_gap = math.dist([game.start_pos[0], game.start_pos[1]], 
                            [standard_cp[0], standard_cp[1]])
        current_segment = 0
        
        # 에피소드 실행 (greedy, epsilon=0)
        while True:
            state = get_data(game, standard_cp, dis_gap)
            
            # greedy=True로 순수 policy만 사용
            _, action = policy_net.predict(state, greedy=True)
            
            frames = policy_net.get_action_frames(action)
            controls = policy_net.get_real_action(action)
            
            done = False
            is_goal = False
            is_collision = False
            
            for _ in range(frames):
                if game.collision or game.goal_reached:
                    break
                
                _, step_done, info = game.step(controls)
                curr_time = game.current_time
                
                all_speed_list.append(game.car.speed * 0.36)
                
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
                
                if step_done:
                    if game.goal_reached:
                        is_goal = True
                        done = True
                    elif game.collision:
                        is_collision = True
                        done = True
                
                # 타임아웃
                if curr_time / 1000 > 2:
                    done = True
                    return False, 1000, 0
                
                if done:
                    break
            
            if done:
                break
        
        # 결과 기록
        if is_goal:
            success_count += 1
            goal_times.append(game.current_time / 1000)
    
    policy_net.model.train()  # 다시 학습 모드로
    
    # 결과 반환
    all_success = (success_count == num_tests)
    avg_speed = np.mean(all_speed_list)
    
    return all_success, avg_speed, success_count


def train_headless(max_episode=10000, output_size=6, replay_length=100000, 
                   target_update=2000, log_interval=100, save_interval=1000, batch_size = 512,
                   track_file="./track.json", 
                   car_json_path="./racing_car.json",
                   layer_num=3, 
                   max_size = 512, 
                   lr = 0.0001):
    """Headless 학습 함수 - 개선된 모델 저장 기준 적용"""
    
    pygame.init()
    
    game = RacingGame(track_file, car_json_path=car_json_path, headless=True)
    
    # 로그 파일 설정
    lr_str = str(lr).replace(".", "_")
    log_filename = f"DQN_withEval_log_lr_{lr_str}_L{layer_num}_S{max_size}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    def log_and_print(message):
        """로그 파일에 쓰고 동시에 콘솔에도 출력"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        log_file.write(log_message)
        log_file.flush()  # 실시간 기록을 위해 flush
        print(message)
    
    log_and_print("=" * 60)
    log_and_print("🚀 HEADLESS TRAINING MODE (DQN)")
    log_and_print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log_and_print(f"   Save Criteria: 10/10 success + best avg time")
    log_and_print(f"   Layer: {layer_num}, MaxSize: {max_size}, LR: {lr}")
    log_and_print(f"   Log File: {log_filename}")
    log_and_print("=" * 60)
    
    ori_checkpoints = copy.deepcopy(game.checkpoints)
    num_segments = len(ori_checkpoints) + 1
    
    policy_net = DQN(input_size=28, output_size=output_size, 
                    replay_memory_length=replay_length, num_segments=num_segments,
                    lr=lr, layer_num=layer_num, max_size=max_size)
    target_net = DQN(input_size=28, output_size=output_size, 
                    replay_memory_length=0, num_segments=num_segments,
                    lr=lr, layer_num=layer_num, max_size=max_size)
    target_net.model.load_state_dict(policy_net.model.state_dict())
    target_net.model.eval()
    
    batch_size = batch_size
    
    episode = 0
    action_step = 0
    max_time = 10
    
    segment_counts = {i: 0 for i in range(num_segments)}
    segment_learned = {i: False for i in range(num_segments)}
    segment_threshold = 50
    current_segment = 0
    
    standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
    dis_gap = math.dist([game.start_pos[0], game.start_pos[1]], 
                        [standard_cp[0], standard_cp[1]])
    
    goal_counts = 0
    # best_avg_speed = max_speed  # [변경] 최고 평균 속도
    best_avg_speed = 0
    save_episode = 0
    episode_rewards = []
    current_episode_reward = 0
    
    start_time = time.time()
    last_log_time = start_time
    
    log_and_print(f"\n📊 Training started at {time.strftime('%H:%M:%S')}")
    log_and_print("-" * 60)
    
    saved_counts = 0
    
    all_speed_list = []
    
    while episode < max_episode:
        learned_until = -1
        for i in range(num_segments):
            if segment_learned[i]:
                learned_until = i
            else:
                break
        
        policy_net.update_epsilon(current_segment, learned_until)
        
        state = get_data(game, standard_cp, dis_gap)
        _, action = policy_net.predict(state)
        
        # 액션 실행 후, 각 프레임 별 얻은 결과들을 총합하는 함수.
        # execute_action_with_frame_skip() 함수 내부 변수 설명:
        #   - game: 게임 인스턴스 (물리 시뮬레이션 및 상태 관리)
        #   - policy_net: 정책 네트워크 (액션 프레임 수, 실제 액션 변환, 리워드 계산)
        #   - action: 선택된 액션 인덱스 (0~5)
        #   - ori_checkpoints: 원본 체크포인트 리스트 (체크포인트 도달 시 다음 목표 결정)
        #   - current_segment: 현재 트랙 세그먼트 인덱스
        #   - standard_cp: 현재 목표 체크포인트 좌표
        #   - dis_gap: 현재 위치에서 목표 체크포인트까지의 거리
        #   - max_time: 에피소드 최대 시간 (초)
        result = execute_action_with_frame_skip(
            game, policy_net, action, ori_checkpoints,
            current_segment, standard_cp, dis_gap, max_time
        )
        
        # execute_action_with_frame_skip() 반환값 사용:
        total_reward = result['total_reward']  # 액션 실행 중 누적된 총 리워드
        next_state = result['next_state']  # 액션 실행 후의 게임 상태 벡터
        done = result['done']  # 에피소드 종료 여부 (골/충돌/타임아웃)
        current_segment = result['current_segment']  # 업데이트된 현재 세그먼트 인덱스 (체크포인트 도달 시 증가)
        standard_cp = result['standard_cp']  # 업데이트된 목표 체크포인트 좌표
        dis_gap = result['dis_gap']  # 업데이트된 목표까지의 거리
        all_speed_list += result['speed_list']  # 각 프레임의 정규화된 속도 값 리스트 (평균 속도 계산용)
        
        policy_net.add_memory([state, action, total_reward, next_state, done])
        current_episode_reward += total_reward
        action_step += 1
        
        if result['segment_reached'] is not None:
            reached_seg = result['segment_reached']
            segment_counts[reached_seg] += 1
            if segment_counts[reached_seg] >= segment_threshold and not segment_learned[reached_seg]:
                segment_learned[reached_seg] = True
                log_and_print(f"\n★★★ Segment {reached_seg} 학습 완료! (Episode: {episode}) ★★★\n")
        
        if result['is_goal']:
            goal_segment_idx = len(ori_checkpoints)
            segment_counts[goal_segment_idx] += 1
            if segment_counts[goal_segment_idx] >= segment_threshold and not segment_learned[goal_segment_idx]:
                segment_learned[goal_segment_idx] = True
                log_and_print(f"\n★★★ GOAL segment 학습 완료! (Episode: {episode}) ★★★\n")
        
        if done:
            episode += 1
            episode_rewards.append(current_episode_reward)
            
            if current_segment > learned_until:
                policy_net.decay_segment_epsilon(current_segment)
            
            if episode % log_interval == 0:
                elapsed = time.time() - last_log_time
                eps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                avg_reward = np.mean(episode_rewards[-log_interval:]) if episode_rewards else 0
                log_msg = (f"[Ep {episode:5d}] Goal: {goal_counts:3d} | Seg: {current_segment} | "
                          f"ε: {policy_net.epsilon:.3f} | Reward: {avg_reward:8.1f} | "
                          f"Speed: {eps_per_sec:.1f} ep/s | Mem: {len(policy_net.replay_memory)} | "
                          f"Avg Speed: {np.mean(all_speed_list) * game.car.max_speed * 0.36: .2f} km/h")
                all_speed_list = []
                log_and_print(log_msg)
                last_log_time = time.time()
            
            if result['is_goal']:
                goal_counts += 1
                goal_time = result['curr_time'] / 1000
                log_and_print(f"  ✓ GOAL! Episode {episode}, Time: {goal_time:.2f}s, Total Goals: {goal_counts}")
                
                # [변경] 모델 평가 및 저장 로직
                if goal_counts > 50:
                    saved_counts += 1
                    log_and_print(f"  🔍 Evaluating model (10 tests)...")
                    all_success, avg_speed, success_count = evaluate_model(
                        policy_net, game, ori_checkpoints, max_time, num_tests=10
                    )
                    
                    if all_success:
                        log_and_print(f"  ✅ Evaluation: {success_count}/10 success, Avg Time: {avg_speed:.3f}s")
                        
                        # if avg_speed < best_avg_speed:
                        if avg_speed > best_avg_speed: 
                            # 이전 best 모델 삭제
                            lr_str = str(lr).replace(".", "_")
                            old_path = f'dqn_best_lr_{lr_str}_L{layer_num}_S{max_size}_E{save_episode}_T{str(round(best_avg_speed, 3)).replace(".", "_")}.pth'
                            if os.path.exists(old_path):
                                os.remove(old_path)
                            
                            # 새 best 모델 저장
                            best_avg_speed = avg_speed
                            save_episode = episode
                            save_path = f'dqn_best_lr_{lr_str}_L{layer_num}_S{max_size}_E{save_episode}_T{str(round(best_avg_speed, 3)).replace(".", "_")}.pth'
                            torch.save(policy_net.model.state_dict(), save_path)
                            log_and_print(f"  💾 New best model saved! (Episode: {episode}, Avg Time: {best_avg_speed:.3f}s)")
                            saved_counts = 0
                    else:
                        log_and_print(f"  ❌ Evaluation: {success_count}/10 success - Not saved")
                    
                    # 평가 후 게임 상태 복구 (evaluate_model에서 reset됨)
                    game.reset()
                    game.checkpoints = copy.deepcopy(ori_checkpoints)
            
            # 게임 리셋
            game.reset()
            game.checkpoints = copy.deepcopy(ori_checkpoints)
            current_segment = 0
            standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
            dis_gap = math.dist([game.start_pos[0], game.start_pos[1]], 
                                [standard_cp[0], standard_cp[1]])
            current_episode_reward = 0
        
        if action_step % 5 == 0 and len(policy_net.replay_memory) > batch_size:
            policy_net.train_step(batch_size, target_net)
        
        if action_step % target_update == 0:
            target_net.model.load_state_dict(policy_net.model.state_dict())
            target_net.model.eval()
        
        if saved_counts >= 500: 
            break
    
    total_time = time.time() - start_time
    log_and_print("\n" + "=" * 60)
    log_and_print("🏁 TRAINING COMPLETED!")
    log_and_print(f"   Algorithm: DQN")
    log_and_print(f"   Total Episodes: {episode}")
    log_and_print(f"   Total Goals: {goal_counts}")
    log_and_print(f"   Best Avg Time: {best_avg_speed:.3f}s")
    log_and_print(f"   Training Time: {total_time/60:.1f} minutes")
    log_and_print(f"   Average Speed: {episode/total_time:.1f} episodes/second")
    log_and_print("=" * 60)
    
    log_file.close()
    pygame.quit()
    
    return policy_net

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_num", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    
    train_headless(
        max_episode = 300000,
        output_size = 6,
        replay_length = 100000,
        target_update = 2000,
        log_interval = 100,
        save_interval = 1000,
        layer_num = args.layer_num,
        max_size = args.max_size,
        lr = args.lr,
        batch_size = args.batch_size,
        car_json_path = "./racing_car.json",
        track_file = "./track.json"
    )
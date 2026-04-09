"""
레이싱 게임 에이전트 - Headless 학습 버전 (렌더링 없이 빠른 학습)
Action + Duration 동시 예측 버전: Dual-head DQN
Loss = action_loss + 0.2 * duration_loss
"""

import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

# 이제 상대 경로(..) 없이 바로 env에서 임포트 가능합니다.
from env.racing_game_2d import RacingGame
from reward_function import get_frame_rewards

class DualHeadNetwork(nn.Module):
    """Action과 Duration을 동시에 예측하는 Dual-head 네트워크"""
    
    def __init__(self, input_size, action_size, duration_size, layer_num, max_size):
        super(DualHeadNetwork, self).__init__()
        
        # Shared layers
        layers = []
        layers.append(nn.Linear(input_size, max_size))
        layers.append(nn.ReLU())
        
        current_size = max_size
        for i in range(layer_num - 1):
            next_size = current_size // 2
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        
        self.shared_net = nn.Sequential(*layers)
        self.feature_size = current_size
        
        # Separate heads
        self.action_head = nn.Linear(current_size, action_size)
        self.duration_head = nn.Linear(current_size, duration_size)
    
    def forward(self, state):
        features = self.shared_net(state)
        action_q = self.action_head(features)
        duration_q = self.duration_head(features)
        return action_q, duration_q


class DQN:
    """DQN 에이전트 (Dual-head: Action + Duration)"""
    
    def __init__(self, input_size=13, action_size=6, duration_size=10, 
                 replay_memory_length=100000, num_segments=5, lr=0.0001, 
                 layer_num=3, max_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.duration_size = duration_size
        self.layer_num = layer_num
        self.max_size = max_size
        
        # Dual-head 모델 생성
        self.model = DualHeadNetwork(
            input_size=input_size,
            action_size=action_size,
            duration_size=duration_size,
            layer_num=layer_num,
            max_size=max_size
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        
        self.gamma = 0.99
        self.epsilon_min = 0.1
        
        self.segment_epsilon = {i: 1.0 for i in range(num_segments)}
        self.segment_decay = 0.998
        self.learned_epsilon = 0.1
        self.epsilon = 1.0
        
        self.replay_memory = []
        self.replay_memory_length = replay_memory_length
        
        # Duration index -> 실제 frame 수 매핑 (1~10)
        # self.duration_map = {i: i + 1 for i in range(duration_size)}  # 0->1, 1->2, ..., 9->10
        self.duration_map = {i: (i+1) * 3 for i in range(duration_size)}
        
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
            
    def predict(self, state, greedy=False):
        """Action과 Duration 동시 예측 (epsilon-greedy 또는 greedy)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_q, duration_q = self.model(state_tensor)
            
            if not greedy and random.random() < self.epsilon:
                action = random.randint(0, self.action_size - 1)
            else: 
                action = action_q.argmax().item()
                
            if not greedy and random.random() < self.epsilon * 0.3: 
                duration_idx = random.randint(0, self.duration_size - 1)
            else:
                duration_idx = duration_q.argmax().item()
            
            duration = self.get_duration_frames(duration_idx)
            
            return action_q, duration_q, action, duration_idx, duration
    
    def add_memory(self, memory): 
        """Memory 추가: [state, action, duration_idx, reward, next_state, done]"""
        if len(self.replay_memory) < self.replay_memory_length: 
            self.replay_memory.append(memory)
        else:
            self.replay_memory.pop(0)
            self.replay_memory.append(memory)
    
    def train_step(self, batch_size, target_net):
        """
        Dual-head DQN 학습 스텝
        - Action과 Duration을 동시에 학습
        - Loss = action_loss + 0.3 * duration_loss
        """
        # Replay memory에서 랜덤 샘플링
        batch = random.sample(self.replay_memory, batch_size)
        
        # Memory format: [state, action, duration_idx, reward, next_state, done]
        states = torch.tensor([m[0] for m in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([m[1] for m in batch], dtype=torch.int64).to(self.device)
        duration_idxs = torch.tensor([m[2] for m in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([m[3] for m in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([m[4] for m in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([m[5] for m in batch], dtype=torch.float32).to(self.device)
        
        # 현재 상태에서의 Q값 계산
        action_q, duration_q = self.model(states)
        action_q_current = action_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        duration_q_current = duration_q.gather(1, duration_idxs.unsqueeze(1)).squeeze(1)
        
        # Target Q값 계산
        with torch.no_grad():
            next_action_q, next_duration_q = target_net.model(next_states)
            max_next_action_q = torch.max(next_action_q, dim=1)[0]
            max_next_duration_q = torch.max(next_duration_q, dim=1)[0]
            
            # Bellman equation으로 target 계산
            action_q_target = rewards + self.gamma * max_next_action_q * (1 - dones)
            duration_q_target = rewards + self.gamma * max_next_duration_q * (1 - dones)
        
        # Loss 계산: action loss와 duration loss의 가중합
        action_loss = self.loss_fn(action_q_current, action_q_target)
        duration_loss = self.loss_fn(duration_q_current, duration_q_target)
        total_loss = action_loss + 0.3 * duration_loss
        
        # 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
        return total_loss.item(), action_loss.item(), duration_loss.item()
    
    def get_frame_reward(self, state, coll, goal, curr_distance, curr_time, max_time, 
                          cp_r, dis_gap, action_index, max_speed, prev_distance):
        return get_frame_rewards(state, coll, goal, curr_distance, curr_time, max_time, 
                                cp_r, dis_gap, action_index, max_speed, prev_distance)


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


# def get_data(game): 
#     angle = game.car.angle
#     cos_angle = (math.cos(angle) + 1) / 2
#     sin_angle = (math.sin(angle) + 1) / 2
#     speed = game.car.speed / game.car.max_speed
#     vel_x = game.car.velocity_x / game.car.max_speed
#     vel_y = game.car.velocity_y / game.car.max_speed
#     sensors = [sensor / game.car.sensor_range for sensor in get_sensors(game)]
#     return sensors + [cos_angle, sin_angle, speed, vel_x, vel_y]

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
        game.car.brake_force / 1000,                  # 브레이크 효율 (이미 0~1)
        game.car.base_friction,                # 마찰계수 (이미 0~1)
        game.car.lateral_friction,       # 측면 마찰 (이미 0~1)
        game.car.turn_speed / 10,               # 회전속도
        game.car.drift_lateral_friction,                # 드리프트 마찰 (이미 0~1)
        game.car.sensor_range / 1000            # 센서 범위 (0~1 정규화)
    ]
    
    return sensors + [cos_angle, sin_angle, speed, vel_x, vel_y, normalized_dist, normalized_angle, is_drifting] + car_features


def execute_action_with_duration(game, agent, action, duration_frames, ori_checkpoints, 
                                  current_segment, standard_cp, dis_gap, max_time):
    """
    Action을 지정된 duration_frames 만큼 실행하고 결과를 반환
    """
    controls = agent.get_real_action(action)
    
    total_reward = 0
    done = False
    is_goal = False
    is_collision = False
    is_timeout = False
    segment_reached = None
    
    curr_segment = current_segment
    curr_standard_cp = standard_cp
    curr_dis_gap = dis_gap
    speed_list = []
    
    for frame_idx in range(duration_frames):
        if game.collision or game.goal_reached:
            break
        prev_distance = math.dist([game.car.x, game.car.y], 
                                   [curr_standard_cp[0], curr_standard_cp[1]])
        
        _, step_done, info = game.step(controls)
        curr_time = game.current_time
        
        frame_state = get_data(game, curr_standard_cp, curr_dis_gap)
        speed_list.append(frame_state[14])
        curr_distance = math.dist([game.car.x, game.car.y], 
                                   [curr_standard_cp[0], curr_standard_cp[1]])
        
        timeout = curr_time / 1000 > max_time
        
        cp_r = 0
        if len(game.checkpoints_reached) > 0:
            cp_r = 50
            cp_idx = game.checkpoints_reached.pop(0)
            reached_cp = game.checkpoints[cp_idx]
            game.checkpoints.pop(cp_idx)
            ori_cp_idx = ori_checkpoints.index(reached_cp)
            
            segment_reached = curr_segment
            
            curr_segment = ori_cp_idx + 1
            curr_standard_cp = ori_checkpoints[ori_cp_idx + 1] if ori_cp_idx < len(ori_checkpoints) - 1 else game.end_pos
            curr_dis_gap = math.dist([reached_cp[0], reached_cp[1]], 
                                      [curr_standard_cp[0], curr_standard_cp[1]])
        
        if step_done:
            if game.goal_reached:
                is_goal = True
                done = True
            elif game.collision:
                is_collision = True
                done = True
        
        if timeout:
            is_timeout = True
            cp_r = -300
            done = True
        
        frame_reward = agent.get_frame_reward(
            frame_state, is_collision, is_goal, curr_distance,
            curr_time, max_time, cp_r, curr_dis_gap, action, game.car.max_speed, prev_distance
        )
        total_reward += frame_reward
        
        # curr_dis_gap = curr_distance
        
        if done:
            break
    
    next_state = get_data(game, curr_standard_cp, curr_dis_gap)
    
    return {
        'total_reward': total_reward,
        'next_state': next_state,
        'done': done,
        'current_segment': curr_segment,
        'standard_cp': curr_standard_cp,
        'dis_gap': curr_dis_gap,
        'segment_reached': segment_reached,
        'is_goal': is_goal,
        'is_collision': is_collision,
        'is_timeout': is_timeout,
        'curr_time': game.current_time,
        'speed_list': speed_list
    }


def evaluate_model(policy_net, game, ori_checkpoints, max_time, num_tests=10):
    """
    모델 평가 - greedy policy로 num_tests회 테스트
    """
    policy_net.model.eval()
    
    goal_times = []
    success_count = 0
    
    all_speed_list = []
    
    for test_idx in range(num_tests):
        game.reset()
        game.checkpoints = copy.deepcopy(ori_checkpoints)
        
        standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
        dis_gap = math.dist([game.start_pos[0], game.start_pos[1]], 
                            [standard_cp[0], standard_cp[1]])
        current_segment = 0
        
        while True:
            state = get_data(game, standard_cp, dis_gap)
            
            # greedy=True로 순수 policy만 사용
            _, _, action, duration_idx, duration = policy_net.predict(state, greedy=True)
            
            controls = policy_net.get_real_action(action)
            
            done = False
            is_goal = False
            is_collision = False
            
            for _ in range(duration):
                if game.collision or game.goal_reached:
                    break
                
                _, step_done, info = game.step(controls)
                curr_time = game.current_time
                
                all_speed_list.append(game.car.speed * 0.36)
                
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
                
                if curr_time / 1000 > 2:
                    done = True
                    return False, 1000, 0
                
                if done:
                    break
            
            if done:
                break
        
        if is_goal:
            success_count += 1
            goal_times.append(game.current_time / 1000)
    
    policy_net.model.train()
    
    all_success = (success_count == num_tests)
    # avg_speed = np.mean(goal_times) if all_success else None
    avg_speed = np.mean(all_speed_list)
    
    return all_success, avg_speed, success_count


def train_headless(max_episode=10000, action_size=8, duration_size=20, 
                   replay_length=100000, target_update=5000, log_interval=100, 
                   save_interval=1000, batch_size = 512,
                   track_file="./track.json", 
                   car_json_path="./racing_car.json",
                   layer_num=3, max_size=512, lr=0.0001):
    """Headless 학습 함수 - Dual-head DQN"""
    
    pygame.init()
    
    game = RacingGame(track_file, car_json_path=car_json_path, headless=True)
    
    # 로그 파일 설정
    lr_str = str(lr).replace(".", "_")
    log_filename = f"{os.path.join(current_dir, '..')}/log_files/DQN_DualHead_log_lr_{lr_str}_L{layer_num}_S{max_size}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    def log_and_print(message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        log_file.write(log_message)
        log_file.flush()
        print(message)
    
    log_and_print("=" * 60)
    log_and_print("🚀 HEADLESS TRAINING MODE (Dual-Head DQN)")
    log_and_print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log_and_print(f"   Action Size: {action_size}, Duration Size: {duration_size}")
    log_and_print(f"   Loss: action_loss + 0.2 * duration_loss")
    log_and_print(f"   Layer: {layer_num}, MaxSize: {max_size}, LR: {lr}")
    log_and_print(f"   Log File: {log_filename}")
    log_and_print("=" * 60)
    
    ori_checkpoints = copy.deepcopy(game.checkpoints)
    num_segments = len(ori_checkpoints) + 1
    
    policy_net = DQN(input_size=28, action_size=action_size, duration_size=duration_size,
                    replay_memory_length=replay_length, num_segments=num_segments,
                    lr=lr, layer_num=layer_num, max_size=max_size)
    target_net = DQN(input_size=28, action_size=action_size, duration_size=duration_size,
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
    # best_avg_speed = max_time
    best_avg_speed = 0
    save_episode = 0
    episode_rewards = []
    current_episode_reward = 0
    
    # Loss 추적용
    action_losses = []
    duration_losses = []
    
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
        
        # Action과 Duration 동시 예측
        _, _, action, duration_idx, duration = policy_net.predict(state)
        
        # Duration을 사용하여 액션 실행
        result = execute_action_with_duration(
            game, policy_net, action, duration, ori_checkpoints,
            current_segment, standard_cp, dis_gap, max_time
        )
        
        total_reward = result['total_reward']
        next_state = result['next_state']
        done = result['done']
        current_segment = result['current_segment']
        standard_cp = result['standard_cp']
        dis_gap = result['dis_gap']
        all_speed_list += result['speed_list']
        
        # Memory에 duration_idx도 저장: [state, action, duration_idx, reward, next_state, done]
        policy_net.add_memory([state, action, duration_idx, total_reward, next_state, done])
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
                avg_a_loss = np.mean(action_losses[-100:]) if action_losses else 0
                avg_d_loss = np.mean(duration_losses[-100:]) if duration_losses else 0
                log_msg = (f"[Ep {episode:5d}] Goal: {goal_counts:3d} | Seg: {current_segment} | "
                          f"ε: {policy_net.epsilon:.3f} | Reward: {avg_reward:8.1f} | "
                          f"A_Loss: {avg_a_loss:.3f} | D_Loss: {avg_d_loss:.3f} | "
                          f"Speed: {eps_per_sec:.1f} ep/s | "
                          f"Avg Speed: {np.mean(all_speed_list) * game.car.max_speed * 0.36: .2f} km/h | "
                          f"Time: {time.strftime('%H:%M:%S')}")
                all_speed_list = []
                log_and_print(log_msg)
                last_log_time = time.time()
            
            if result['is_goal']:
                goal_counts += 1
                goal_time = result['curr_time'] / 1000
                log_and_print(f"  ✓ GOAL! Episode {episode}, Avg Speed: {np.mean(all_speed_list) * game.car.max_speed * 0.36: .2f} km/h, Total Goals: {goal_counts}")
                
                if goal_counts > 50:
                    saved_counts += 1
                    log_and_print(f"  🔍 Evaluating model (10 tests)...")
                    all_success, avg_speed, success_count = evaluate_model(
                        policy_net, game, ori_checkpoints, max_time, num_tests=10
                    )
                    
                    if all_success:
                        log_and_print(f"  ✅ Evaluation: {success_count}/10 success, Avg Speed: {np.mean(all_speed_list) * game.car.max_speed * 0.36: .2f} km/h")
                        
                        # if avg_speed < best_avg_speed:
                        if avg_speed > best_avg_speed: 
                            lr_str = str(lr).replace(".", "_")
                            old_path = f"{os.path.join(current_dir, '..')}/example_models/dqn_dualhead_best_lr_{lr_str}_L{layer_num}_S{max_size}_E{save_episode}_T{str(round(best_avg_speed, 3)).replace('.', '_')}.pth"
                            if os.path.exists(old_path):
                                os.remove(old_path)
                            
                            best_avg_speed = avg_speed
                            save_episode = episode
                            save_path = f"{os.path.join(current_dir, '..')}/example_models/dqn_dualhead_best_lr_{lr_str}_L{layer_num}_S{max_size}_E{save_episode}_T{str(round(best_avg_speed, 3)).replace('.', '_')}.pth"
                            torch.save(policy_net.model.state_dict(), save_path)
                            log_and_print(f"  💾 New best model saved! (Episode: {episode}, Avg Speed: {np.mean(all_speed_list) * game.car.max_speed * 0.36: .2f} km/h)")
                            saved_counts = 0
                    else:
                        log_and_print(f"  ❌ Evaluation: {success_count}/10 success - Not saved")
                    
                    game.reset()
                    game.checkpoints = copy.deepcopy(ori_checkpoints)
            
            game.reset()
            game.checkpoints = copy.deepcopy(ori_checkpoints)
            current_segment = 0
            standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
            dis_gap = math.dist([game.start_pos[0], game.start_pos[1]], 
                                [standard_cp[0], standard_cp[1]])
            current_episode_reward = 0
        
        # 학습 스텝
        if action_step % 5 == 0 and len(policy_net.replay_memory) > batch_size:
            total_loss, a_loss, d_loss = policy_net.train_step(batch_size, target_net)
            action_losses.append(a_loss)
            duration_losses.append(d_loss)
        
        if action_step % target_update == 0:
            target_net.model.load_state_dict(policy_net.model.state_dict())
            target_net.model.eval()
        
        if saved_counts >= 500: 
            break
    
    total_time = time.time() - start_time
    log_and_print("\n" + "=" * 60)
    log_and_print("🏁 TRAINING COMPLETED!")
    log_and_print(f"   Algorithm: Dual-Head DQN (Action + Duration)")
    log_and_print(f"   Total Episodes: {episode}")
    log_and_print(f"   Total Goals: {goal_counts}")
    log_and_print(f"   Best Avg Speed: {best_avg_speed:.3f}km/h")
    log_and_print(f"   Training Time: {total_time/60:.1f} minutes")
    log_and_print(f"   Average Traing Speed: {episode/total_time:.1f} episodes/second")
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
    
    TRACK_FILE = f"{os.path.join(current_dir, '..')}/env/track.json"
    CAR_JSON_PATH = f"{os.path.join(current_dir, '..')}/env/racing_car.json"
    
    train_headless(
        max_episode = 300000,
        action_size = 8,
        duration_size = 6,
        replay_length = 100000,
        target_update = 2000,
        log_interval = 100,
        save_interval = 1000,
        layer_num = args.layer_num,
        max_size = args.max_size,
        lr = args.lr,
        batch_size = args.batch_size,
        car_json_path = CAR_JSON_PATH,
        track_file = TRACK_FILE
    )
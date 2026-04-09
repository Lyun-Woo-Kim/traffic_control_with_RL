"""
새 맵용 학습 코드 - Dueling Double DQN + Dual Head
- racing_game_2d_new.py 사용 (track_data.json 포맷)
- 8방향 차선 레이캐스팅 센서
- 도로 방향벡터 + 교차로 여부 + 신호등 정보 포함
- train_headless() 내 CHECKPOINTS 직접 지정
"""

import pygame
import random
import math
import numpy as np
import copy
import time
import os

import torch
import torch.nn as nn

from traffic_env import RacingGame

# ============================================================
# 상수
# ============================================================
INPUT_SIZE = 29   # 8(센서) + 8(차량상태) + 8(차량스펙) + 5(도로/신호)


# ============================================================
# 상태 수집 함수
# ============================================================
def get_sensors(game):
    """
    8방향 레이캐스팅 - 차선(lane lines)까지의 거리 (비정규화)
    get_data() 에서 sensor_range 로 나눠 정규화함
    """
    sensors = []
    for i in range(8):
        angle    = game.car.angle + i * (math.pi / 4)
        dist_n   = game._cast_ray(game.car.x, game.car.y, angle)
        sensors.append(dist_n * game.car.sensor_range)
    return sensors


def get_data(game, standard_cp=None, dis_gap=None):
    """
    전체 상태 벡터 생성 (항상 INPUT_SIZE=29 고정)

    구성:
      sensors(8) + [cos,sin,speed,vx,vy,dist,angle,drift](8)
      + car_features(8) + [dir_x,dir_y,is_inter,tl_exists,tl_state](5)
    """
    angle     = game.car.angle
    cos_angle = (math.cos(angle) + 1) / 2
    sin_angle = (math.sin(angle) + 1) / 2
    speed     = game.car.speed / game.car.max_speed
    vel_x     = game.car.velocity_x / game.car.max_speed
    vel_y     = game.car.velocity_y / game.car.max_speed
    sensors   = [s / game.car.sensor_range for s in get_sensors(game)]

    is_drifting = 1.0 if game.car.is_drifting else 0.0   # 항상 정의

    if standard_cp is not None and dis_gap is not None and dis_gap > 0:
        current_distance = math.dist([game.car.x, game.car.y], standard_cp)
        normalized_dist  = min(current_distance / dis_gap, 1.5)

        dx = standard_cp[0] - game.car.x
        dy = standard_cp[1] - game.car.y
        target_angle    = math.atan2(dy, dx)
        relative_angle  = target_angle - game.car.angle
        while relative_angle >  math.pi: relative_angle -= 2 * math.pi
        while relative_angle < -math.pi: relative_angle += 2 * math.pi
        normalized_angle = (relative_angle + math.pi) / (2 * math.pi)
    else:
        normalized_dist  = 1.0
        normalized_angle = 0.5

    car_features = [
        game.car.max_speed          / 1000,
        game.car.acceleration_force / 1000,
        game.car.brake_force        / 1000,
        game.car.base_friction,
        game.car.lateral_friction,
        game.car.turn_speed         / 10,
        game.car.base_drift_friction,          # 기존 코드의 drift_lateral_friction 오류 수정
        game.car.sensor_range       / 1000,
    ]

    # 도로 방향벡터 + 교차로 여부
    road_info       = game._get_road_info()    # [is_intersection, dir_x, dir_y]
    is_intersection = road_info[0]
    dir_x           = road_info[1]
    dir_y           = road_info[2]

    # 신호등 존재 여부 + 상태 (빨강:0, 노랑:1, 초록:2)
    tl_exists, tl_state = game._get_traffic_light_info()

    return (sensors
            + [cos_angle, sin_angle, speed, vel_x, vel_y,
               normalized_dist, normalized_angle, is_drifting]
            + car_features
            + [dir_x, dir_y, is_intersection, tl_exists, tl_state])


# ============================================================
# 보상 함수
# ============================================================
def get_frame_reward(state, is_collision, is_goal,
                     curr_distance, curr_time, max_time,
                     cp_reward, dis_gap, action_index,
                     max_speed, prev_distance):
    """프레임별 보상 계산"""
    reward = 0.0

    # 충돌 / 골
    if is_collision: return -500.0
    if is_goal:      return  500.0

    # 타임아웃 패널티
    time_ratio = curr_time / 1000 / max_time
    reward -= time_ratio * 0.5

    # 목표 접근 보상
    if dis_gap > 0:
        dist_improvement = prev_distance - curr_distance
        reward += dist_improvement * 0.5

    # 속도 보상 (state[2] = 정규화 속도)
    reward += state[2] * max_speed * 0.002

    # 체크포인트 보상
    reward += cp_reward

    # ── 역주행 패널티 ────────────────────────────────────────────
    # state 인덱스: sensors(0~7) car_state(8~15) car_feat(16~23) road(24~28)
    #   state[8]=cos_angle(정규화), state[9]=sin_angle(정규화)
    #   state[3]=vel_x(정규화),     state[4]=vel_y(정규화)
    #   state[24]=dir_x, state[25]=dir_y  (교차로 = [0.0, 0.0])
    vel_x_n = state[3]
    vel_y_n = state[4]
    speed_n = state[2]

    # heading 벡터 (정규화 해제: 저장 시 (cos+1)/2 로 인코딩)
    cos_a = state[8] * 2.0 - 1.0
    sin_a = state[9] * 2.0 - 1.0

    # ① 후진 역주행: 차량 heading 과 속도 방향이 반대
    #    실제로 뒤로 움직이는 상황 → 소형 패널티
    heading_vel_dot = cos_a * vel_x_n + sin_a * vel_y_n
    if heading_vel_dot < -0.1 and speed_n > 0.05:
        reward -= 1.0   # 후진 패널티 (소)

    # ② 차선 침범 역주행: 앞으로 가고 있는데 도로 방향과 반대
    #    교차로(dir=[0,0])는 방향 정보 없으므로 제외
    dir_x = state[24]
    dir_y = state[25]
    if dir_x != 0.0 or dir_y != 0.0:
        road_vel_dot = vel_x_n * dir_x + vel_y_n * dir_y
        is_going_forward = heading_vel_dot >= 0.0
        if is_going_forward and road_vel_dot < -0.1:
            reward -= 5.0   # 차선 침범 역주행 패널티 (대)

    # ── 신호 위반 패널티 ──────────────────────────────────────────
    # _get_traffic_light_info() 에서 차량 진행 방향과 일치하는 신호등만 반환하므로
    # 이 패널티는 해당 차량이 실제로 따라야 하는 신호등에 대해서만 적용됨
    tl_exists = state[27]
    tl_state  = state[28]
    if tl_exists == 1:
        if tl_state == 0:   # 빨간불
            if speed_n > 0.05:
                # 속도에 비례한 패널티 — 빠를수록 더 큰 위반
                reward -= 5.0 * speed_n
            else:
                # 빨간불에 정지 → 준수 보상
                reward += 0.5
        elif tl_state == 1:  # 노란불 — 감속해야 함
            if speed_n > 0.2:
                reward -= 2.0 * (speed_n - 0.2)
            else:
                # 노란불에 감속 중 → 소량 보상
                reward += 0.2

    return reward


# ============================================================
# 모델 (기존 DuelingDualHeadNetwork 동일)
# ============================================================
class DuelingDualHeadNetwork(nn.Module):
    def __init__(self, input_size, action_size, duration_size, layer_num, max_size):
        super().__init__()

        shared_layers = [nn.Linear(input_size, max_size), nn.ReLU()]
        current_size  = max_size
        for _ in range(layer_num - 1):
            next_size = current_size // 2
            shared_layers += [nn.Linear(current_size, next_size), nn.ReLU()]
            current_size = next_size
        self.shared       = nn.Sequential(*shared_layers)
        self.feature_size = current_size

        self.action_value_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2), nn.ReLU(),
            nn.Linear(self.feature_size // 2, 1))
        self.action_advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2), nn.ReLU(),
            nn.Linear(self.feature_size // 2, action_size))

        self.duration_value_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2), nn.ReLU(),
            nn.Linear(self.feature_size // 2, 1))
        self.duration_advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2), nn.ReLU(),
            nn.Linear(self.feature_size // 2, duration_size))

    def forward(self, x):
        f  = self.shared(x)
        av = self.action_value_stream(f)
        aa = self.action_advantage_stream(f)
        aq = av + (aa - aa.mean(dim=1, keepdim=True))

        dv = self.duration_value_stream(f)
        da = self.duration_advantage_stream(f)
        dq = dv + (da - da.mean(dim=1, keepdim=True))
        return aq, dq


# ============================================================
# 에이전트 (기존 DuelingDoubleDQN_DualHead 동일)
# ============================================================
class DuelingDoubleDQN_DualHead:
    def __init__(self, input_size=INPUT_SIZE, action_size=8, duration_size=20,
                 replay_memory_length=100000, num_segments=5,
                 lr=0.0001, layer_num=3, max_size=512):
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size   = action_size
        self.duration_size = duration_size
        self.layer_num     = layer_num
        self.max_size      = max_size

        self.model = DuelingDualHeadNetwork(
            input_size, action_size, duration_size, layer_num, max_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.gamma         = 0.99
        self.epsilon_min   = 0.1
        self.segment_epsilon = {i: 1.0 for i in range(num_segments)}
        self.segment_decay   = 0.998
        self.learned_epsilon = 0.1
        self.epsilon         = 1.0

        self.replay_memory        = []
        self.replay_memory_length = replay_memory_length
        self.duration_map = {i: (i + 1) * 3 for i in range(duration_size)}

    def update_epsilon(self, current_segment, learned_until):
        self.epsilon = (self.learned_epsilon
                        if current_segment <= learned_until
                        else self.segment_epsilon.get(current_segment, 1.0))

    def decay_segment_epsilon(self, segment):
        if segment in self.segment_epsilon:
            self.segment_epsilon[segment] = max(
                self.segment_epsilon[segment] * self.segment_decay,
                self.epsilon_min)

    def get_real_action(self, action_index):
        actions = {
            0: {'forward': True,  'backward': False, 'left': False, 'right': False, 'brake': False},
            1: {'forward': True,  'backward': False, 'left': True,  'right': False, 'brake': False},
            2: {'forward': True,  'backward': False, 'left': False, 'right': True,  'brake': False},
            3: {'forward': False, 'backward': False, 'left': True,  'right': False, 'brake': True },
            4: {'forward': False, 'backward': False, 'left': False, 'right': True,  'brake': True },
            5: {'forward': True,  'backward': False, 'left': True,  'right': False, 'brake': True },
            6: {'forward': True,  'backward': False, 'left': False, 'right': True,  'brake': True },
            7: {'forward': False, 'backward': False, 'left': False, 'right': False, 'brake': True },
        }
        return actions.get(action_index, actions[0])

    def get_duration_frames(self, duration_index):
        return self.duration_map.get(duration_index, 3)

    def predict(self, state, greedy=False):
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            aq, dq = self.model(t)
            action = (aq.argmax().item()
                      if greedy or random.random() >= self.epsilon
                      else random.randint(0, self.action_size - 1))
            dur_idx = (dq.argmax().item()
                       if greedy or random.random() >= self.epsilon * 0.3
                       else random.randint(0, self.duration_size - 1))
            return aq, dq, action, dur_idx, self.get_duration_frames(dur_idx)

    def add_memory(self, memory):
        if len(self.replay_memory) >= self.replay_memory_length:
            self.replay_memory.pop(0)
        self.replay_memory.append(memory)

    def train_step(self, batch_size, target_net):
        batch        = random.sample(self.replay_memory, batch_size)
        states       = torch.tensor([m[0] for m in batch], dtype=torch.float32).to(self.device)
        actions      = torch.tensor([m[1] for m in batch], dtype=torch.int64).to(self.device)
        dur_idxs     = torch.tensor([m[2] for m in batch], dtype=torch.int64).to(self.device)
        rewards      = torch.tensor([m[3] for m in batch], dtype=torch.float32).to(self.device)
        next_states  = torch.tensor([m[4] for m in batch], dtype=torch.float32).to(self.device)
        dones        = torch.tensor([m[5] for m in batch], dtype=torch.float32).to(self.device)

        aq, dq = self.model(states)
        aq_cur = aq.gather(1, actions.unsqueeze(1)).squeeze(1)
        dq_cur = dq.gather(1, dur_idxs.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            naq_p, ndq_p = self.model(next_states)
            best_a = naq_p.argmax(dim=1)
            best_d = ndq_p.argmax(dim=1)
            naq_t, ndq_t = target_net.model(next_states)
            aq_target = rewards + self.gamma * naq_t.gather(1, best_a.unsqueeze(1)).squeeze(1) * (1 - dones)
            dq_target = rewards + self.gamma * ndq_t.gather(1, best_d.unsqueeze(1)).squeeze(1) * (1 - dones)

        a_loss     = self.loss_fn(aq_cur, aq_target)
        d_loss     = self.loss_fn(dq_cur, dq_target)
        total_loss = a_loss + 0.3 * d_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), a_loss.item(), d_loss.item()


# ============================================================
# 액션 실행
# ============================================================
def execute_action_with_duration(game, agent, action, duration_frames, ori_checkpoints,
                                  current_segment, standard_cp, dis_gap, max_time):
    controls     = agent.get_real_action(action)
    total_reward = 0
    done = is_goal = is_collision = is_timeout = False
    segment_reached = None
    curr_segment    = current_segment
    curr_standard_cp = list(standard_cp)
    curr_dis_gap    = dis_gap
    speed_list      = []

    # checkpoints_reached 는 pop 하지 않고 커서로만 추적
    # → game.checkpoints 인덱스가 유지되어 _check_checkpoints() 오작동 방지
    processed_cp_count = len(game.checkpoints_reached)

    for _ in range(duration_frames):
        if game.collision or game.goal_reached:
            break

        prev_distance = math.dist([game.car.x, game.car.y], curr_standard_cp)

        _, step_done, info = game.step(controls)
        curr_time = game.current_time

        frame_state   = get_data(game, curr_standard_cp, curr_dis_gap)
        speed_list.append(frame_state[2])   # 정규화 속도
        curr_distance = math.dist([game.car.x, game.car.y], curr_standard_cp)

        timeout = curr_time / 1000 > max_time

        cp_r = 0
        if len(game.checkpoints_reached) > processed_cp_count:
            cp_r       = 50
            cp_idx     = game.checkpoints_reached[processed_cp_count]  # peek, pop 금지
            processed_cp_count += 1
            reached_cp = game.checkpoints[cp_idx]   # game.checkpoints 에서 pop 금지

            try:
                ori_cp_idx = ori_checkpoints.index(reached_cp)
            except ValueError:
                ori_cp_idx = curr_segment

            segment_reached  = curr_segment
            curr_segment     = ori_cp_idx + 1
            curr_standard_cp = (ori_checkpoints[ori_cp_idx + 1]
                                 if ori_cp_idx < len(ori_checkpoints) - 1
                                 else game.end_pos)
            curr_dis_gap = math.dist(reached_cp, curr_standard_cp)

        if step_done:
            if game.goal_reached:  is_goal      = done = True
            elif game.collision:   is_collision = done = True

        if timeout:
            is_timeout = True
            cp_r       = -5000
            done       = True

        frame_reward = get_frame_reward(
            frame_state, is_collision, is_goal,
            curr_distance, curr_time, max_time,
            cp_r, curr_dis_gap, action,
            game.car.max_speed, prev_distance
        )
        total_reward += frame_reward

        if done:
            break

    next_state = get_data(game, curr_standard_cp, curr_dis_gap)
    return {
        'total_reward':    total_reward,
        'next_state':      next_state,
        'done':            done,
        'current_segment': curr_segment,
        'standard_cp':     curr_standard_cp,
        'dis_gap':         curr_dis_gap,
        'segment_reached': segment_reached,
        'is_goal':         is_goal,
        'is_collision':    is_collision,
        'is_timeout':      is_timeout,
        'curr_time':       game.current_time,
        'speed_list':      speed_list,
    }


# ============================================================
# 평가
# ============================================================
def evaluate_model(agent, game, ori_checkpoints, max_time, num_tests=10):
    agent.model.eval()
    goal_times, all_speeds = [], []
    success_count = 0

    for _ in range(num_tests):
        game.reset()
        game.checkpoints = copy.deepcopy(ori_checkpoints)

        standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
        dis_gap     = math.dist(game.start_pos, standard_cp)
        done = is_goal = False

        while True:
            state = get_data(game, standard_cp, dis_gap)
            _, _, action, _, duration = agent.predict(state, greedy=True)
            controls = agent.get_real_action(action)

            processed_cp_count = len(game.checkpoints_reached)
            for _ in range(duration):
                if game.collision or game.goal_reached:
                    break
                _, step_done, _ = game.step(controls)
                all_speeds.append(game.car.speed * 0.36)

                if len(game.checkpoints_reached) > processed_cp_count:
                    cp_idx     = game.checkpoints_reached[processed_cp_count]
                    processed_cp_count += 1
                    reached_cp = game.checkpoints[cp_idx]
                    try:
                        ori_idx = ori_checkpoints.index(reached_cp)
                    except ValueError:
                        ori_idx = 0
                    standard_cp = (ori_checkpoints[ori_idx + 1]
                                   if ori_idx < len(ori_checkpoints) - 1
                                   else game.end_pos)

                if step_done:
                    if game.goal_reached: is_goal = True
                    done = True

                if game.current_time / 1000 > max_time:
                    done = True
                if done:
                    break
            if done:
                break

        if is_goal:
            success_count += 1
            goal_times.append(game.current_time / 1000)

    agent.model.train()
    avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
    return (success_count == num_tests), avg_speed, success_count


# ============================================================
# 학습 함수
# ============================================================
def train_headless(max_episode=300000, action_size=8, duration_size=6,
                   replay_length=100000, target_update=2000, log_interval=100,
                   save_interval=1000, batch_size=512,
                   track_file="./track_data.json",
                   car_json_path="./racing_car.json",
                   layer_num=3, max_size=512, lr=0.0001, 
                   CHECKPOINT_ORDER=None):
    """
    Dueling Double DQN + Dual Head 학습

    ★ CHECKPOINT_ORDER: track_data.json 에 저장된 체크포인트의 인덱스 번호를
      원하는 순서대로 나열하세요.
      예) [0, 1, 2, 3] → 트랙에 찍힌 0번→1번→2번→3번 순으로 진행
          [2, 0, 3, 1] → 순서를 바꿔서 진행도 가능
      실제 좌표는 track_data.json 에서 자동으로 불러옵니다.
    """

    pygame.init()
    game = RacingGame(track_file, car_json_path=car_json_path, headless=True)

    # 인덱스 순서대로 실제 좌표 조합
    all_cps = game.checkpoints   # track_data.json 에서 로드된 전체 체크포인트
    ori_checkpoints = [all_cps[i] for i in CHECKPOINT_ORDER if i < len(all_cps)]
    game.checkpoints = copy.deepcopy(ori_checkpoints)
    num_segments      = len(ori_checkpoints) + 1   # 체크포인트 수 + 골 세그먼트

    # 로그 설정
    lr_str       = str(lr).replace(".", "_")
    log_filename = f"train_new_map_lr_{lr_str}_L{layer_num}_S{max_size}.txt"
    log_file     = open(log_filename, 'w', encoding='utf-8')

    def log(msg):
        ts  = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}\n"
        log_file.write(line); log_file.flush(); print(msg)

    log("=" * 60)
    log("DUELING DOUBLE DQN + DUAL HEAD  |  NEW MAP")
    log(f"  Device      : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log(f"  Input Size  : {INPUT_SIZE}")
    log(f"  Actions     : {action_size}  Durations: {duration_size}")
    log(f"  Layer       : {layer_num}  MaxSize: {max_size}  LR: {lr}")
    log(f"  Checkpoints : {num_segments - 1}개")
    log(f"  Track file  : {track_file}")
    log("=" * 60)

    policy_net = DuelingDoubleDQN_DualHead(
        input_size=INPUT_SIZE, action_size=action_size, duration_size=duration_size,
        replay_memory_length=replay_length, num_segments=num_segments,
        lr=lr, layer_num=layer_num, max_size=max_size
    )
    target_net = DuelingDoubleDQN_DualHead(
        input_size=INPUT_SIZE, action_size=action_size, duration_size=duration_size,
        replay_memory_length=0, num_segments=num_segments,
        lr=lr, layer_num=layer_num, max_size=max_size
    )
    target_net.model.load_state_dict(policy_net.model.state_dict())
    target_net.model.eval()

    episode          = 0
    action_step      = 0
    max_time         = 10
    goal_counts      = 0
    best_avg_speed   = 0.0
    save_episode     = 0
    saved_counts     = 0
    episode_rewards  = []
    current_ep_reward = 0
    action_losses    = []
    duration_losses  = []
    all_speed_list   = []
    start_time       = time.time()
    last_log_time    = start_time

    segment_counts   = {i: 0 for i in range(num_segments)}
    segment_learned  = {i: False for i in range(num_segments)}
    segment_threshold = 50
    current_segment  = 0

    standard_cp = ori_checkpoints[0] if ori_checkpoints else game.end_pos
    dis_gap     = math.dist(game.start_pos, standard_cp)

    log(f"\nTraining started  {time.strftime('%H:%M:%S')}")
    log("-" * 60)

    while episode < max_episode:
        # 어느 세그먼트까지 학습됐는지 파악
        learned_until = -1
        for i in range(num_segments):
            if segment_learned[i]: learned_until = i
            else: break

        policy_net.update_epsilon(current_segment, learned_until)

        state = get_data(game, standard_cp, dis_gap)
        _, _, action, duration_idx, duration = policy_net.predict(state)

        result = execute_action_with_duration(
            game, policy_net, action, duration,
            ori_checkpoints, current_segment, standard_cp, dis_gap, max_time
        )

        total_reward     = result['total_reward']
        next_state       = result['next_state']
        done             = result['done']
        current_segment  = result['current_segment']
        standard_cp      = result['standard_cp']
        dis_gap          = result['dis_gap']
        all_speed_list  += result['speed_list']

        # [state, action, duration_idx, reward, next_state, done]
        policy_net.add_memory([state, action, duration_idx, total_reward, next_state, done])
        current_ep_reward += total_reward
        action_step       += 1

        # 세그먼트 도달 기록
        if result['segment_reached'] is not None:
            seg = result['segment_reached']
            segment_counts[seg] += 1
            if segment_counts[seg] >= segment_threshold and not segment_learned[seg]:
                segment_learned[seg] = True
                log(f"\n★ Segment {seg} 학습 완료! (Episode {episode})\n")

        if result['is_goal']:
            goal_seg = len(ori_checkpoints)
            segment_counts[goal_seg] += 1
            if segment_counts[goal_seg] >= segment_threshold and not segment_learned[goal_seg]:
                segment_learned[goal_seg] = True
                log(f"\n★ GOAL segment 학습 완료! (Episode {episode})\n")

        if done:
            episode += 1
            episode_rewards.append(current_ep_reward)

            if current_segment > learned_until:
                policy_net.decay_segment_epsilon(current_segment)

            # 정기 로그
            if episode % log_interval == 0:
                elapsed     = time.time() - last_log_time
                eps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                avg_reward  = np.mean(episode_rewards[-log_interval:]) if episode_rewards else 0
                avg_a_loss  = np.mean(action_losses[-100:]) if action_losses else 0
                avg_d_loss  = np.mean(duration_losses[-100:]) if duration_losses else 0
                avg_spd_kmh = (np.mean(all_speed_list) * game.car.max_speed * 0.36
                               if all_speed_list else 0)
                log(f"[Ep {episode:5d}] Goal:{goal_counts:3d} | Seg:{current_segment} | "
                    f"ε:{policy_net.epsilon:.3f} | Reward:{avg_reward:8.1f} | "
                    f"A_Loss:{avg_a_loss:.3f} | D_Loss:{avg_d_loss:.3f} | "
                    f"Speed:{eps_per_sec:.1f}ep/s | AvgSpd:{avg_spd_kmh:.1f}km/h | "
                    f"{time.strftime('%H:%M:%S')}")
                all_speed_list = []
                last_log_time  = time.time()

            if result['is_goal']:
                goal_counts += 1
                log(f"  GOAL! Ep {episode}  Total:{goal_counts}")

                if goal_counts > 50:
                    saved_counts += 1
                    all_ok, avg_spd, sc = evaluate_model(
                        policy_net, game, ori_checkpoints, max_time, num_tests=10)

                    if all_ok:
                        log(f"  Eval: {sc}/10 success  AvgSpd:{avg_spd:.1f}km/h")
                        if avg_spd > best_avg_speed:
                            old_path = (f"new_map_best_lr_{lr_str}_L{layer_num}_S{max_size}"
                                        f"_E{save_episode}_T{str(round(best_avg_speed,3)).replace('.','_')}.pth")
                            if os.path.exists(old_path):
                                os.remove(old_path)
                            best_avg_speed = avg_spd
                            save_episode   = episode
                            save_path = (f"new_map_best_lr_{lr_str}_L{layer_num}_S{max_size}"
                                         f"_E{save_episode}_T{str(round(best_avg_speed,3)).replace('.','_')}.pth")
                            torch.save(policy_net.model.state_dict(), save_path)
                            log(f"  Best model saved: {save_path}")
                            saved_counts = 0
                    else:
                        log(f"  Eval: {sc}/10 — not saved")

                    game.reset()
                    game.checkpoints = copy.deepcopy(ori_checkpoints)

            # 에피소드 리셋
            game.reset()
            game.checkpoints    = copy.deepcopy(ori_checkpoints)
            current_segment     = 0
            standard_cp         = ori_checkpoints[0] if ori_checkpoints else game.end_pos
            dis_gap             = math.dist(game.start_pos, standard_cp)
            current_ep_reward   = 0

        # 학습 스텝
        if action_step % 5 == 0 and len(policy_net.replay_memory) > batch_size:
            t_loss, a_loss, d_loss = policy_net.train_step(batch_size, target_net)
            action_losses.append(a_loss)
            duration_losses.append(d_loss)

        if action_step % target_update == 0:
            target_net.model.load_state_dict(policy_net.model.state_dict())
            target_net.model.eval()

        if saved_counts >= 500:
            break

    total_time = time.time() - start_time
    log("\n" + "=" * 60)
    log("TRAINING COMPLETED")
    log(f"  Total Episodes : {episode}")
    log(f"  Total Goals    : {goal_counts}")
    log(f"  Best Avg Speed : {best_avg_speed:.3f} km/h")
    log(f"  Training Time  : {total_time/60:.1f} min")
    log("=" * 60)

    log_file.close()
    pygame.quit()
    return policy_net


# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_num",  type=int,   default=3)
    parser.add_argument("--max_size",   type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int,   default=512)
    args = parser.parse_args()

    train_headless(
        max_episode   = 300000,
        action_size   = 8,
        duration_size = 6,
        replay_length = 100000,
        target_update = 2000,
        log_interval  = 100,
        layer_num     = args.layer_num,
        max_size      = args.max_size,
        lr            = args.lr,
        batch_size    = args.batch_size,
        car_json_path = "./racing_car.json",
        track_file    = "./track_data.json",
    )

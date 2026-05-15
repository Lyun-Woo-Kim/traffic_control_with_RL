"""
다중 에이전트 학습 코드 — Dueling Double DQN + Dual Head
- 에이전트마다 독립 네트워크 / 리플레이 버퍼 / epsilon
- 각 차량 JSON의 checkpoints 마지막 항목 = GOAL
- 모든 에이전트가 한 환경에서 동시에 주행하며 서로를 센서로 인식
- train_headless(..., curriculum=True): 차량 1대로 시작해 조건 충족 시 한 대씩 추가 (옵션)
"""

import pygame
import random
import math
import numpy as np
import copy
import time
import os
import multiprocessing as mp
import queue as pyqueue

import torch
import torch.nn as nn

from traffic_env import RacingGame
from pathlib import Path

# 이 스크립트가 있는 traffic_simulation 폴더 (CWD와 무관하게 에셋 경로 해석)
_SIM_DIR = Path(__file__).resolve().parent


def _sim_asset(*parts: str) -> str:
    """traffic_simulation 기준 상대 경로를 절대 경로 문자열로 반환."""
    return str(_SIM_DIR.joinpath(*parts))

# ============================================================
# 상수
# ============================================================
INPUT_SIZE = 30   # 8(센서) + 8(차량상태) + 8(차량스펙) + 6(도로/신호/우회전)


# ============================================================
# 상태 수집 함수
# ============================================================
def get_sensors(game, car_idx=0):
    """car_idx 차량의 8방향 레이캐스팅 (비정규화 px)"""
    car     = game.cars[car_idx]
    angles  = [car.angle + i * (math.pi / 4) for i in range(8)]
    if hasattr(game, "_cast_rays_for_car"):
        return [dist_n * car.sensor_range
                for dist_n in game._cast_rays_for_car(car_idx, car.x, car.y, angles)]

    return [
        game._cast_ray_for_car(car_idx, car.x, car.y, angle) * car.sensor_range
        for angle in angles
    ]


def get_data(game, car_idx=0, standard_cp=None, dis_gap=None, skip_sensors=False):
    """
    car_idx 에이전트의 상태 벡터 생성 (INPUT_SIZE=30 고정)

    구성:
      sensors(8) + [cos,sin,speed,vx,vy,dist,angle,drift](8)
      + car_features(8) + [dir_x,dir_y,is_inter,tl_exists,tl_state,right_turnable](6)

    skip_sensors=True 이면 8개 센서를 0으로 채우고 광선 캐스팅을 건너뛴다.
    (보상 계산만 필요한 경로용 — 보상 함수는 sensors 슬롯을 사용하지 않음.)
    """
    car   = game.cars[car_idx]
    angle = car.angle

    cos_angle = (math.cos(angle) + 1) / 2
    sin_angle = (math.sin(angle) + 1) / 2
    speed     = car.speed / car.max_speed
    vel_x     = car.velocity_x / car.max_speed
    vel_y     = car.velocity_y / car.max_speed
    if skip_sensors:
        sensors = [0.0] * 8
    else:
        sensors = [s / car.sensor_range for s in get_sensors(game, car_idx)]

    is_drifting = 1.0 if car.is_drifting else 0.0

    if standard_cp is not None and dis_gap is not None and dis_gap > 0:
        current_distance = math.dist([car.x, car.y], standard_cp)
        normalized_dist  = min(current_distance / dis_gap, 1.5)
        dx = standard_cp[0] - car.x
        dy = standard_cp[1] - car.y
        target_angle   = math.atan2(dy, dx)
        relative_angle = target_angle - car.angle
        while relative_angle >  math.pi: relative_angle -= 2 * math.pi
        while relative_angle < -math.pi: relative_angle += 2 * math.pi
        normalized_angle = (relative_angle + math.pi) / (2 * math.pi)
    else:
        normalized_dist  = 1.0
        normalized_angle = 0.5

    car_features = [
        car.max_speed          / 1000,
        car.acceleration_force / 1000,
        car.brake_force        / 1000,
        car.base_friction,
        car.lateral_friction,
        car.turn_speed         / 10,
        car.base_drift_friction,
        car.sensor_range       / 1000,
    ]

    road_info       = game._get_road_info_for_car(car_idx)
    is_intersection = road_info[0]
    dir_x           = road_info[1]
    dir_y           = road_info[2]

    tl_exists, tl_state, right_turnable = game._get_traffic_light_info_for_car(car_idx)

    return (sensors
            + [cos_angle, sin_angle, speed, vel_x, vel_y,
               normalized_dist, normalized_angle, is_drifting]
            + car_features
            + [dir_x, dir_y, is_intersection, tl_exists, tl_state, right_turnable])


# ============================================================
# 보상 함수 (프레임 단위, 에이전트 공통)
# ----------------------------------------------------------------
# 설계 원칙
#   1) 종단 신호(충돌/골)가 가장 강해야 한다 → ±100. 다른 보상은 이보다 작게.
#   2) 매 프레임 shaping은 ‘작고 균일하게’ — 60 FPS × ~수천 프레임에 누적되어도
#      종단 신호를 가리지 않을 정도.
#   3) 빨간불 정지 등 ‘가만히 있기만 +보상’은 제거(hacking 방지). 초록에서
#      전진하는 경우에 한해 아주 작은 +보상만 허용(종단 ±100보다 한참 작게).
#   4) 진행도(목표에 가까워짐)와 체크포인트 도달은 ‘sparse + dense’ 균형용으로
#      유지하되, 종단 신호를 넘지 않게 스케일 조정.
# ============================================================
def get_frame_reward(state, is_collision, is_goal,
                     curr_distance, curr_time, max_time,
                     cp_reward, dis_gap, action_index,
                     max_speed, prev_distance):
    # 종단(에피소드 종료) — 가장 강한 단일 신호
    if is_collision: return -100.0
    if is_goal:      return +100.0

    reward = 0.0

    # 매 프레임 작은 시간 압력 (상수). 60 FPS × 60s ≈ -36 누적.
    reward -= 0.01

    # 목표 접근(진행도) — (prev - curr)는 픽셀 단위. 빠른 차도 프레임당 ~수 px.
    # 0.05 계수로 frame당 최대 ~0.4 → 한 CP까지 누적도 종단 신호보다 작음.
    if dis_gap > 0:
        reward += (prev_distance - curr_distance) * 0.05

    # 체크포인트 도달 — 이벤트성 (외부에서 cp_reward=100가 한 번 들어옴).
    # 종단 신호와의 비율을 위해 5분의 1로 축소: +100 → +20.
    reward += cp_reward * 0.2

    # ── 역주행 / 자세 페널티 ─────────────────────────────────────
    speed_n = state[10]
    vel_x_n = state[11]
    vel_y_n = state[12]
    cos_a   = state[8] * 2.0 - 1.0
    sin_a   = state[9] * 2.0 - 1.0

    # ① 후진 (heading 과 속도 방향 반대)
    heading_vel_dot = cos_a * vel_x_n + sin_a * vel_y_n
    if heading_vel_dot < -0.1 and speed_n > 0.05:
        reward -= 0.2

    # ② 차선 진행방향 역주행 (앞으로 가는데 도로 방향과 반대)
    dir_x = state[24]
    dir_y = state[25]
    if dir_x != 0.0 or dir_y != 0.0:
        road_vel_dot = vel_x_n * dir_x + vel_y_n * dir_y
        if heading_vel_dot >= 0.0 and road_vel_dot < -0.3:
            reward -= 0.5

    # ── 신호등 ───────────────────────────────────────────────────
    # 빨간/노란: 위험·위반 위주 페널티. 초록: 전진 시만 소액 + (정지·공회전은 제외).
    # 정지선 위반 이벤트 패널티는 train_loop에서 별도 부여.
    tl_exists      = state[27]
    tl_state       = state[28]
    right_turnable = state[29]
    if tl_exists == 1:
        if tl_state == 0 and right_turnable == 0:    # 빨간불 + 직진/좌회전
            if speed_n > 0.05:
                reward -= speed_n * 0.2              # 움직이는 정도에 비례
        elif tl_state == 1:                          # 노란불 — 과속 시만 페널티
            if speed_n > 0.2:
                reward -= (speed_n - 0.2) * 0.5
        elif tl_state == 2:                          # 초록 — 전진할 때만 + (<< ±100)
            if speed_n > 0.08 and heading_vel_dot > 0.05:
                reward += 0.02

    return reward


def get_reward_features(game, car_idx=0):
    """프레임 보상에 실제로 필요한 값만 계산한다."""
    car = game.cars[car_idx]
    max_speed = car.max_speed
    speed_n = car.speed / max_speed
    vel_x_n = car.velocity_x / max_speed
    vel_y_n = car.velocity_y / max_speed
    cos_a = math.cos(car.angle)
    sin_a = math.sin(car.angle)
    road_info = game._get_road_info_for_car(car_idx)
    tl_exists, tl_state, right_turnable = game._get_traffic_light_info_for_car(car_idx)
    return (
        speed_n, vel_x_n, vel_y_n, cos_a, sin_a,
        road_info[1], road_info[2],
        tl_exists, tl_state, right_turnable,
    )


def get_frame_reward_from_features(features, is_collision, is_goal,
                                   curr_distance, curr_time, max_time,
                                   cp_reward, dis_gap, action_index,
                                   max_speed, prev_distance):
    # 종단(에피소드 종료) — 가장 강한 단일 신호
    if is_collision: return -100.0
    if is_goal:      return +100.0

    reward = 0.0
    reward -= 0.01

    if dis_gap > 0:
        reward += (prev_distance - curr_distance) * 0.05

    reward += cp_reward * 0.2

    (speed_n, vel_x_n, vel_y_n, cos_a, sin_a,
     dir_x, dir_y, tl_exists, tl_state, right_turnable) = features

    heading_vel_dot = cos_a * vel_x_n + sin_a * vel_y_n
    if heading_vel_dot < -0.1 and speed_n > 0.05:
        reward -= 0.2

    if dir_x != 0.0 or dir_y != 0.0:
        road_vel_dot = vel_x_n * dir_x + vel_y_n * dir_y
        if heading_vel_dot >= 0.0 and road_vel_dot < -0.3:
            reward -= 0.5

    if tl_exists == 1:
        if tl_state == 0 and right_turnable == 0:
            if speed_n > 0.05:
                reward -= speed_n * 0.2
        elif tl_state == 1:
            if speed_n > 0.2:
                reward -= (speed_n - 0.2) * 0.5
        elif tl_state == 2:
            if speed_n > 0.08 and heading_vel_dot > 0.05:
                reward += 0.02

    return reward


# ============================================================
# 모델
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
# Replay Buffer (NumPy 순환 버퍼) — list+pop(0) O(N) 비용 제거,
# train_step 의 텐서 변환을 한 번의 numpy 슬라이싱으로 처리.
# 샘플링은 기존과 동일한 uniform random.
# ============================================================
class NumpyReplayBuffer:
    def __init__(self, capacity: int, state_size: int):
        self.capacity = max(int(capacity), 1)
        self.size = 0
        self.idx = 0
        self.states      = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.actions     = np.zeros(self.capacity, dtype=np.int64)
        self.dur_idxs    = np.zeros(self.capacity, dtype=np.int64)
        self.rewards     = np.zeros(self.capacity, dtype=np.float32)
        self.dones       = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def add(self, state, action, dur_idx, reward, next_state, done):
        i = self.idx
        self.states[i]      = state
        self.actions[i]     = action
        self.dur_idxs[i]    = dur_idx
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = done
        self.idx = (self.idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.dur_idxs[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


# ============================================================
# 에이전트
# ============================================================
class DuelingDoubleDQN_DualHead:
    def __init__(self, input_size=INPUT_SIZE, action_size=8, duration_size=20,
                 replay_memory_length=100000,
                 lr=0.0001, layer_num=3, max_size=512, device=None):
        self.device        = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size   = action_size
        self.duration_size = duration_size
        self.input_size    = input_size

        self.model = DuelingDualHeadNetwork(
            input_size, action_size, duration_size, layer_num, max_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_min   = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_table = [self.epsilon]

        self.replay_memory_length = replay_memory_length
        self.replay_memory        = NumpyReplayBuffer(replay_memory_length, input_size)
        self.duration_map = {i: (i + 1) * 1 for i in range(duration_size)}

    def init_epsilon_table(self, segment_count):
        segment_count = max(int(segment_count), 1)
        self.epsilon_table = [self.epsilon for _ in range(segment_count)]

    def get_segment_epsilon(self, segment_idx):
        if not self.epsilon_table:
            return self.epsilon
        idx = min(max(int(segment_idx), 0), len(self.epsilon_table) - 1)
        return self.epsilon_table[idx]

    def decay_epsilon(self, segment_idx=None):
        if segment_idx is None:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.epsilon_table = [max(e * self.epsilon_decay, self.epsilon_min) for e in self.epsilon_table]
            return

        if not self.epsilon_table:
            self.init_epsilon_table(1)
        idx = min(max(int(segment_idx), 0), len(self.epsilon_table) - 1)
        self.epsilon_table[idx] = max(self.epsilon_table[idx] * self.epsilon_decay, self.epsilon_min)

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

    def predict(self, state, segment_idx=0, greedy=False):
        with torch.inference_mode():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            aq, dq = self.model(t)
            epsilon = self.get_segment_epsilon(segment_idx)
            action  = (aq.argmax().item()
                       if greedy or random.random() >= epsilon
                       else random.randint(0, self.action_size - 1))
            dur_idx = (dq.argmax().item()
                       if greedy or random.random() >= epsilon * 0.7
                       else random.randint(0, self.duration_size - 1))
            return aq, dq, action, dur_idx, self.get_duration_frames(dur_idx)

    def add_memory(self, memory):
        # 호환성: 기존 호출부 [state, action, dur_idx, reward, next_state, done] 형태 유지
        s, a, d, r, ns, done = memory
        self.replay_memory.add(s, a, d, r, ns, done)

    def train_step(self, batch_size, target_net):
        s_np, a_np, d_np, r_np, ns_np, done_np = self.replay_memory.sample(batch_size)
        states      = torch.from_numpy(s_np).to(self.device, non_blocking=True)
        actions     = torch.from_numpy(a_np).to(self.device, non_blocking=True)
        dur_idxs    = torch.from_numpy(d_np).to(self.device, non_blocking=True)
        rewards     = torch.from_numpy(r_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(ns_np).to(self.device, non_blocking=True)
        dones       = torch.from_numpy(done_np).to(self.device, non_blocking=True)

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
# 병렬 rollout worker 유틸리티
# ============================================================
def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _agent_snapshots(agents):
    return [_cpu_state_dict(agent.model) for agent in agents]


def _epsilon_tables(agents):
    return [list(agent.epsilon_table) for agent in agents]


def _build_worker_agents(n, snapshots, epsilon_tables,
                         action_size, duration_size, lr, layer_num, max_size):
    agents = [
        DuelingDoubleDQN_DualHead(
            INPUT_SIZE, action_size, duration_size,
            replay_memory_length=1,
            lr=lr, layer_num=layer_num, max_size=max_size,
            device="cpu",
        )
        for _ in range(n)
    ]
    for i, agent in enumerate(agents):
        agent.model.load_state_dict(snapshots[i])
        agent.model.eval()
        agent.epsilon_table = list(epsilon_tables[i])
        agent.epsilon = agent.epsilon_table[0] if agent.epsilon_table else agent.epsilon
    return agents


def _worker_reset_episode_state(game, n):
    std_cps = []
    dis_gps = []
    for i in range(n):
        nav_cps = game.car_checkpoints[i]
        cj = game.car_jsons[i]
        sp_idx = cj.get('start_point', i % max(len(game.start_positions), 1))
        sp = (game.start_positions[sp_idx % len(game.start_positions)]
              if game.start_positions else [game.width // 2, game.height // 2])
        first = nav_cps[0] if nav_cps else game.car_goals[i]
        std_cps.append(list(first) if first else [sp[0], sp[1]])
        dis_gps.append(math.dist(sp, first) if first else 1.0)
    return std_cps, dis_gps


def _worker_new_env_state(game, n, agents):
    standard_cps, dis_gaps = _worker_reset_episode_state(game, n)
    return {
        'standard_cps': standard_cps,
        'dis_gaps': dis_gaps,
        'remaining_frames': [0] * n,
        'pending_states': [None] * n,
        'pending_actions': [0] * n,
        'pending_dur_idxs': [0] * n,
        'accumulated_rews': [0.0] * n,
        'processed_cp_cnts': [0] * n,
        'current_controls': [agents[i].get_real_action(0) for i in range(n)],
        'active': [True] * n,
        'ep_rew_buf': [0.0] * n,
        'ep_done_reason': [''] * n,
        'ep_cp_reached': [0] * n,
    }


def _worker_segment_idx(agent, processed_count):
    segment_count = max(len(agent.epsilon_table), 1)
    return min(processed_count, segment_count - 1)


def _rollout_worker_main(worker_id, cmd_q, result_q, config, snapshots, epsilon_tables):
    try:
        random.seed(config['seed'] + worker_id)
        np.random.seed(config['seed'] + worker_id)
        pygame.init()

        stage_id = config['stage_id']
        active_paths = list(config['active_paths'])
        n = len(active_paths)
        game = RacingGame(config['track_file'], vehicle_configs=active_paths, headless=True)
        agents = _build_worker_agents(
            n, snapshots, epsilon_tables,
            config['action_size'], config['duration_size'],
            config['lr'], config['layer_num'], config['max_size'],
        )
        env_state = _worker_new_env_state(game, n, agents)
        result_q.put({
            'type': 'worker_ready',
            'stage_id': stage_id,
            'worker_id': worker_id,
        })

        transitions = []
        decay_events = []
        episodes = []
        steps_since_send = 0
        total_steps = 0
        chunk_steps = max(int(config.get('rollout_chunk_steps', 32)), 1)
        max_time = config['max_time']

        while True:
            # Apply pending learner commands without blocking rollout.
            try:
                while True:
                    cmd = cmd_q.get_nowait()
                    kind = cmd.get('type')
                    if kind == 'stop':
                        return
                    if kind == 'sync':
                        snapshots = cmd['snapshots']
                        epsilon_tables = cmd['epsilon_tables']
                        for i, agent in enumerate(agents):
                            agent.model.load_state_dict(snapshots[i])
                            agent.model.eval()
                            agent.epsilon_table = list(epsilon_tables[i])
                            agent.epsilon = agent.epsilon_table[0] if agent.epsilon_table else agent.epsilon
                    elif kind == 'stage':
                        stage_id = cmd['stage_id']
                        active_paths = list(cmd['active_paths'])
                        n = len(active_paths)
                        snapshots = cmd['snapshots']
                        epsilon_tables = cmd['epsilon_tables']
                        game = RacingGame(config['track_file'], vehicle_configs=active_paths, headless=True)
                        agents = _build_worker_agents(
                            n, snapshots, epsilon_tables,
                            config['action_size'], config['duration_size'],
                            config['lr'], config['layer_num'], config['max_size'],
                        )
                        env_state = _worker_new_env_state(game, n, agents)
                        transitions = []
                        decay_events = []
                        episodes = []
                        steps_since_send = 0
                        total_steps = 0
                        result_q.put({
                            'type': 'worker_ready',
                            'stage_id': stage_id,
                            'worker_id': worker_id,
                        })
            except pyqueue.Empty:
                pass

            standard_cps = env_state['standard_cps']
            dis_gaps = env_state['dis_gaps']
            remaining_frames = env_state['remaining_frames']
            pending_states = env_state['pending_states']
            pending_actions = env_state['pending_actions']
            pending_dur_idxs = env_state['pending_dur_idxs']
            accumulated_rews = env_state['accumulated_rews']
            processed_cp_cnts = env_state['processed_cp_cnts']
            current_controls = env_state['current_controls']
            active = env_state['active']
            ep_rew_buf = env_state['ep_rew_buf']
            ep_done_reason = env_state['ep_done_reason']
            ep_cp_reached = env_state['ep_cp_reached']

            for i in range(n):
                if remaining_frames[i] <= 0 and active[i]:
                    state = get_data(game, i, standard_cps[i], dis_gaps[i])
                    if pending_states[i] is not None:
                        transitions.append((
                            i, pending_states[i], pending_actions[i], pending_dur_idxs[i],
                            accumulated_rews[i], state, 0.0,
                        ))
                        ep_rew_buf[i] += accumulated_rews[i]
                        accumulated_rews[i] = 0.0

                    seg_idx = _worker_segment_idx(agents[i], processed_cp_cnts[i])
                    _, _, action, dur_idx, duration = agents[i].predict(state, segment_idx=seg_idx)
                    pending_states[i] = state
                    pending_actions[i] = action
                    pending_dur_idxs[i] = dur_idx
                    current_controls[i] = agents[i].get_real_action(action)
                    remaining_frames[i] = duration

            prev_dists = [
                math.dist([game.cars[i].x, game.cars[i].y], standard_cps[i])
                if active[i] else 0.0
                for i in range(n)
            ]
            results = game.step(current_controls)
            steps_since_send += 1
            total_steps += 1

            for i, result in enumerate(results):
                if not active[i]:
                    continue

                cp_r = 0
                curr_dist = math.dist([game.cars[i].x, game.cars[i].y], standard_cps[i])
                if result['cp_reached']:
                    cp_r = 100
                    ep_cp_reached[i] += 1
                    completed_seg_idx = _worker_segment_idx(agents[i], processed_cp_cnts[i])
                    agents[i].decay_epsilon(completed_seg_idx)
                    decay_events.append((i, completed_seg_idx))
                    processed_cp_cnts[i] += 1
                    nav_cps = game.car_checkpoints[i]
                    new_target = (nav_cps[processed_cp_cnts[i]]
                                  if processed_cp_cnts[i] < len(nav_cps)
                                  else game.car_goals[i])
                    if new_target:
                        dis_gaps[i] = math.dist(standard_cps[i], new_target)
                        standard_cps[i] = list(new_target)

                reward_features = get_reward_features(game, i)
                is_timeout = (game.current_time or 0) / 1000 > max_time
                frame_reward = get_frame_reward_from_features(
                    reward_features,
                    result['collision'], result['goal_reached'],
                    curr_dist, game.current_time or 0, max_time,
                    cp_r, dis_gaps[i], pending_actions[i],
                    game.cars[i].max_speed, prev_dists[i],
                )
                if result['red_light_crossed'] and not result['red_light_right_turn']:
                    frame_reward -= 50.0
                if is_timeout:
                    frame_reward -= 50.0

                accumulated_rews[i] += frame_reward
                remaining_frames[i] -= 1

                done_i = result['collision'] or result['goal_reached'] or is_timeout
                if done_i:
                    terminal_seg_idx = _worker_segment_idx(agents[i], processed_cp_cnts[i])
                    agents[i].decay_epsilon(terminal_seg_idx)
                    decay_events.append((i, terminal_seg_idx))
                    next_state = get_data(game, i, standard_cps[i], dis_gaps[i])
                    transitions.append((
                        i, pending_states[i], pending_actions[i], pending_dur_idxs[i],
                        accumulated_rews[i], next_state, 1.0,
                    ))
                    ep_rew_buf[i] += accumulated_rews[i]
                    accumulated_rews[i] = 0.0
                    pending_states[i] = None
                    active[i] = False

                    if result['goal_reached']:
                        ep_done_reason[i] = 'goal'
                    elif result['collision']:
                        ep_done_reason[i] = 'collision'
                    else:
                        ep_done_reason[i] = 'timeout'

            if not any(active):
                episodes.append({
                    'worker_id': worker_id,
                    'rewards': list(ep_rew_buf),
                    'done_reasons': list(ep_done_reason),
                    'cp_reached': list(ep_cp_reached),
                })
                game.reset()
                env_state = _worker_new_env_state(game, n, agents)

            if total_steps in (1, 5, 10):
                result_q.put({
                    'type': 'worker_heartbeat',
                    'stage_id': stage_id,
                    'worker_id': worker_id,
                    'steps': total_steps,
                })

            if transitions or decay_events or episodes or steps_since_send >= chunk_steps:
                result_q.put({
                    'type': 'rollout',
                    'stage_id': stage_id,
                    'worker_id': worker_id,
                    'steps': steps_since_send,
                    'transitions': transitions,
                    'decay_events': decay_events,
                    'episodes': episodes,
                })
                transitions = []
                decay_events = []
                episodes = []
                steps_since_send = 0
    except Exception as exc:
        try:
            result_q.put({
                'type': 'worker_error',
                'stage_id': config.get('stage_id', -1),
                'worker_id': worker_id,
                'error': repr(exc),
            })
        finally:
            raise


def _run_greedy_evaluation(eval_game, agents, n, eval_trials, eval_max_time):
    for i in range(n):
        agents[i].model.eval()

    eval_success = [0] * n
    eval_times = [[] for _ in range(n)]

    for _trial in range(eval_trials):
        eval_game.reset()
        _std_cps, _dis_gps = _worker_reset_episode_state(eval_game, n)
        _rem = [0] * n
        _ctrl = [agents[i].get_real_action(0) for i in range(n)]
        _alive = [True] * n
        _cp_c = [0] * n

        while any(_alive):
            for i in range(n):
                if _rem[i] <= 0 and _alive[i]:
                    st = get_data(eval_game, i, _std_cps[i], _dis_gps[i])
                    _, _, a, d, dur = agents[i].predict(st, segment_idx=0, greedy=True)
                    _ctrl[i] = agents[i].get_real_action(a)
                    _rem[i] = dur

            step_res = eval_game.step(_ctrl)

            for i, res in enumerate(step_res):
                if not _alive[i]:
                    continue
                _rem[i] -= 1
                is_to = (eval_game.current_time or 0) / 1000 > eval_max_time

                if res['cp_reached']:
                    _cp_c[i] += 1
                    nav_cps = eval_game.car_checkpoints[i]
                    new_tgt = (nav_cps[_cp_c[i]]
                               if _cp_c[i] < len(nav_cps)
                               else eval_game.car_goals[i])
                    if new_tgt:
                        _dis_gps[i] = math.dist(_std_cps[i], new_tgt)
                        _std_cps[i] = list(new_tgt)

                if res['goal_reached'] or res['collision'] or is_to:
                    if res['goal_reached']:
                        eval_success[i] += 1
                        eval_times[i].append((eval_game.current_time or 0) / 1000.0)
                    _alive[i] = False

    for i in range(n):
        agents[i].model.train()

    return eval_success, eval_times


def train_headless_parallel(vehicle_config_paths,
                            max_episode=300000,
                            action_size=8, duration_size=6,
                            replay_length=50000,
                            target_update=2000,
                            log_interval=100,
                            batch_size=2048,
                            track_file=None,
                            layer_num=3, max_size=512, lr=0.0001,
                            max_time=60,
                            eval_interval=500,
                            eval_trials=10,
                            eval_max_time=60,
                            curriculum=True,
                            curriculum_perfect_evals_per_agent=10,
                            curriculum_min_episodes_per_stage=0,
                            curriculum_min_action_steps_per_stage=5000,
                            parallel_workers=4,
                            rollout_chunk_steps=32,
                            train_every_steps=20,
                            gradient_steps=1,
                            sync_interval_steps=1000,
                            seed=1234):
    if track_file is None:
        track_file = _sim_asset("track_data.json")

    full_vehicle_paths = list(vehicle_config_paths)
    use_curriculum = bool(curriculum) and len(full_vehicle_paths) > 1
    active_paths = full_vehicle_paths[:1] if use_curriculum else full_vehicle_paths
    parallel_workers = max(int(parallel_workers), 1)
    train_every_steps = max(int(train_every_steps), 1)
    gradient_steps = max(int(gradient_steps), 1)
    sync_interval_steps = max(int(sync_interval_steps), 1)

    pygame.init()
    eval_game = RacingGame(track_file, vehicle_configs=active_paths, headless=True)
    n = eval_game.n_agents

    agents = [
        DuelingDoubleDQN_DualHead(
            INPUT_SIZE, action_size, duration_size,
            replay_length, lr, layer_num, max_size,
        )
        for _ in range(n)
    ]
    target_nets = [
        DuelingDoubleDQN_DualHead(
            INPUT_SIZE, action_size, duration_size,
            0, lr, layer_num, max_size,
        )
        for _ in range(n)
    ]
    for i in range(n):
        seg_cnt = len(eval_game.car_checkpoints[i]) + (1 if eval_game.car_goals[i] is not None else 0)
        agents[i].init_epsilon_table(seg_cnt)
        target_nets[i].init_epsilon_table(seg_cnt)
        target_nets[i].model.load_state_dict(agents[i].model.state_dict())
        target_nets[i].model.eval()

    save_dir = _SIM_DIR / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    lr_str = str(lr).replace(".", "_")
    run_tag = (f"par{parallel_workers}_curr_max{len(full_vehicle_paths)}_st{n}ag_lr_{lr_str}_"
               f"L{layer_num}_S{max_size}") if use_curriculum else (
                   f"par{parallel_workers}_ma_{n}agents_lr_{lr_str}_L{layer_num}_S{max_size}")
    log_filename = str(save_dir / f"train_{run_tag}.txt")
    log_file = open(log_filename, 'w', encoding='utf-8')
    csv_filename = log_filename.replace('.txt', '_detail.csv')
    csv_file = open(csv_filename, 'w', encoding='utf-8')

    def log(msg):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}\n"
        log_file.write(line)
        log_file.flush()
        print(msg)

    def write_csv_header():
        csv_file.write(
            'episode,worker_id,'
            + ','.join(f'A{i+1}_reward,A{i+1}_done,A{i+1}_cp' for i in range(n))
            + '\n'
        )
        csv_file.flush()

    write_csv_header()

    log("=" * 60)
    log(f"PARALLEL DDDQN  |  workers={parallel_workers}  |  agents={n}"
        + (f"  |  CURRICULUM → max {len(full_vehicle_paths)} vehicles" if use_curriculum else ""))
    log(f"  Device : {agents[0].device}")
    log(f"  Batch  : {batch_size}  | train_every_steps={train_every_steps} "
        f"| gradient_steps={gradient_steps} | sync_interval_steps={sync_interval_steps}")
    log(f"  Track  : {track_file}")
    log("=" * 60)

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue(maxsize=max(parallel_workers * 8, 16))
    cmd_queues = [ctx.Queue() for _ in range(parallel_workers)]
    stage_id = 0
    worker_config = {
        'track_file': track_file,
        'active_paths': active_paths,
        'stage_id': stage_id,
        'action_size': action_size,
        'duration_size': duration_size,
        'lr': lr,
        'layer_num': layer_num,
        'max_size': max_size,
        'max_time': max_time,
        'rollout_chunk_steps': rollout_chunk_steps,
        'seed': seed,
    }
    snapshots = _agent_snapshots(agents)
    eps_tables = _epsilon_tables(agents)
    workers = [
        ctx.Process(
            target=_rollout_worker_main,
            args=(wid, cmd_queues[wid], result_q, worker_config, snapshots, eps_tables),
            daemon=True,
        )
        for wid in range(parallel_workers)
    ]
    for p in workers:
        p.start()
    log("  Workers started: " + " ".join(str(p.pid) for p in workers))

    episode = 0
    action_step = 0
    next_train_step = train_every_steps
    next_target_step = target_update
    next_sync_step = sync_interval_steps
    next_eval_episode = eval_interval
    episodes_in_stage = 0
    action_step_at_stage_start = 0
    curriculum_eval_wins = [0] * n
    goal_counts = [0] * n
    ep_rewards = [[] for _ in range(n)]
    action_losses = [[] for _ in range(n)]
    duration_losses = [[] for _ in range(n)]
    ep_collision_cnts = [0] * n
    ep_timeout_cnts = [0] * n
    ep_goal_cnts_ep = [0] * n
    best_eval_times = [float('inf')] * n
    best_eval_paths = [None] * n
    start_time = time.time()
    last_log_time = start_time
    last_wait_log_time = start_time

    def broadcast(kind='sync'):
        payload = {
            'type': kind,
            'stage_id': stage_id,
            'active_paths': active_paths,
            'snapshots': _agent_snapshots(agents),
            'epsilon_tables': _epsilon_tables(agents),
        }
        for q in cmd_queues:
            q.put(payload)

    try:
        while episode < max_episode:
            try:
                msg = result_q.get(timeout=5.0)
            except pyqueue.Empty:
                dead = [(idx, p.exitcode) for idx, p in enumerate(workers) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"Parallel rollout worker(s) exited before sending data: {dead}")
                now = time.time()
                if now - last_wait_log_time >= 30:
                    alive = sum(1 for p in workers if p.is_alive())
                    log(f"  [Parallel] waiting for rollout data... "
                        f"alive_workers={alive}/{parallel_workers}, "
                        f"steps={action_step}, episodes={episode}")
                    last_wait_log_time = now
                continue
            if msg.get('type') == 'worker_error':
                raise RuntimeError(
                    f"Parallel rollout worker {msg.get('worker_id')} failed: {msg.get('error')}"
                )
            if msg.get('type') == 'worker_ready':
                log(f"  [Parallel] worker {msg.get('worker_id')} ready")
                continue
            if msg.get('type') == 'worker_heartbeat':
                log(f"  [Parallel] worker {msg.get('worker_id')} heartbeat "
                    f"steps={msg.get('steps')}")
                continue
            if msg.get('stage_id') != stage_id:
                continue

            action_step += int(msg.get('steps', 0))

            for agent_idx, state, action, dur_idx, reward, next_state, done in msg.get('transitions', []):
                agents[agent_idx].add_memory([state, action, dur_idx, reward, next_state, done])

            for agent_idx, seg_idx in msg.get('decay_events', []):
                agents[agent_idx].decay_epsilon(seg_idx)

            while action_step >= next_train_step:
                for _ in range(gradient_steps):
                    for i in range(n):
                        if len(agents[i].replay_memory) > batch_size:
                            tl, al, dl = agents[i].train_step(batch_size, target_nets[i])
                            action_losses[i].append(al)
                            duration_losses[i].append(dl)
                next_train_step += train_every_steps

            while action_step >= next_target_step:
                for i in range(n):
                    target_nets[i].model.load_state_dict(agents[i].model.state_dict())
                    target_nets[i].model.eval()
                next_target_step += target_update

            if action_step >= next_sync_step:
                broadcast('sync')
                next_sync_step += sync_interval_steps

            for ep in msg.get('episodes', []):
                episode += 1
                episodes_in_stage += 1
                rewards = ep['rewards']
                reasons = ep['done_reasons']
                cps = ep['cp_reached']

                for i in range(n):
                    ep_rewards[i].append(rewards[i])
                    if reasons[i] == 'goal':
                        goal_counts[i] += 1
                        ep_goal_cnts_ep[i] += 1
                    elif reasons[i] == 'collision':
                        ep_collision_cnts[i] += 1
                    elif reasons[i] == 'timeout':
                        ep_timeout_cnts[i] += 1

                row = f"{episode},{ep['worker_id']}"
                for i in range(n):
                    row += f",{rewards[i]:.2f},{reasons[i]},{cps[i]}"
                csv_file.write(row + '\n')
                if episode % 10 == 0:
                    csv_file.flush()

                if episode % log_interval == 0:
                    elapsed = time.time() - last_log_time
                    eps_per_sec = log_interval / elapsed if elapsed > 0 else 0.0
                    avg_rews = [
                        np.mean(ep_rewards[i][-log_interval:]) if ep_rewards[i] else 0.0
                        for i in range(n)
                    ]
                    avg_a_losses = [
                        np.mean(action_losses[i][-100:]) if action_losses[i] else 0.0
                        for i in range(n)
                    ]
                    col_rate = [ep_collision_cnts[i] / max(episode, 1) * 100 for i in range(n)]
                    to_rate = [ep_timeout_cnts[i] / max(episode, 1) * 100 for i in range(n)]
                    rew_str = " | ".join(f"A{i+1}:{avg_rews[i]:7.1f}" for i in range(n))
                    loss_str = " | ".join(f"A{i+1}:{avg_a_losses[i]:.3f}" for i in range(n))
                    goal_str = " ".join(f"A{i+1}:{goal_counts[i]}" for i in range(n))
                    eps_str = " ".join(f"A{i+1}:S0={agents[i].get_segment_epsilon(0):.3f}" for i in range(n))
                    col_str = " | ".join(f"A{i+1}:{col_rate[i]:.0f}%" for i in range(n))
                    to_str = " | ".join(f"A{i+1}:{to_rate[i]:.0f}%" for i in range(n))
                    log(f"[Ep {episode:5d}] Goals:[{goal_str}] | ε:[{eps_str}] | "
                        f"Workers:{parallel_workers} | Reward:[{rew_str}] | "
                        f"ALoss:[{loss_str}] | {eps_per_sec:.1f}ep/s | {time.strftime('%H:%M:%S')}")
                    log(f"          Collision:[{col_str}] | Timeout:[{to_str}]")
                    last_log_time = time.time()

                if episode % 1000 == 0:
                    for i in range(n):
                        path = str(save_dir / f"agent{i+1}_ep{episode}_{run_tag}.pth")
                        torch.save(agents[i].model.state_dict(), path)
                    log(f"  [Ckpt] Ep{episode} 모델 저장 완료")

                if episode >= next_eval_episode:
                    log(f"  [Eval] Ep{episode} — {eval_trials}회 평가 시작 "
                        f"(greedy, MaxT={eval_max_time}s) ...")
                    eval_success, eval_times = _run_greedy_evaluation(
                        eval_game, agents, n, eval_trials, eval_max_time)

                    for i in range(n):
                        sc = eval_success[i]
                        if sc == eval_trials:
                            avg_t = np.mean(eval_times[i])
                            if avg_t < best_eval_times[i]:
                                if best_eval_paths[i] and os.path.exists(best_eval_paths[i]):
                                    os.remove(best_eval_paths[i])
                                best_eval_times[i] = avg_t
                                new_path = str(save_dir /
                                               f"agent{i+1}_eval_best_{avg_t:.1f}s_{run_tag}.pth")
                                torch.save(agents[i].model.state_dict(), new_path)
                                best_eval_paths[i] = new_path
                                log(f"  [Eval] Agent{i+1} ✓ {eval_trials}/{eval_trials} "
                                    f"avg={avg_t:.1f}s  → saved  ({new_path})")
                            else:
                                log(f"  [Eval] Agent{i+1} ✓ {eval_trials}/{eval_trials} "
                                    f"avg={avg_t:.1f}s  (best={best_eval_times[i]:.1f}s, skip)")
                        else:
                            log(f"  [Eval] Agent{i+1} ✗ {sc}/{eval_trials} success — skip")

                    next_eval_episode += eval_interval

                    steps_in_stage = action_step - action_step_at_stage_start
                    mins_ok = (
                        episodes_in_stage >= curriculum_min_episodes_per_stage
                        and steps_in_stage >= curriculum_min_action_steps_per_stage
                    )
                    if use_curriculum and n < len(full_vehicle_paths):
                        for ci in range(n):
                            if eval_success[ci] == eval_trials:
                                curriculum_eval_wins[ci] += 1
                        wins_str = " ".join(
                            f"A{ci+1}:{curriculum_eval_wins[ci]}/{curriculum_perfect_evals_per_agent}"
                            for ci in range(n)
                        )
                        log(f"  [Curriculum] 완벽 평가 누적 ({wins_str})  |  "
                            f"단계 ep={episodes_in_stage}, env_step={steps_in_stage}")

                        all_agents_ready = all(
                            curriculum_eval_wins[ci] >= curriculum_perfect_evals_per_agent
                            for ci in range(n)
                        )
                        if all_agents_ready and mins_ok:
                            old_n = n
                            new_n = old_n + 1
                            donor_sd = agents[old_n - 1].model.state_dict()
                            active_paths = full_vehicle_paths[:new_n]
                            eval_game = RacingGame(
                                track_file, vehicle_configs=active_paths, headless=True)

                            na = DuelingDoubleDQN_DualHead(
                                INPUT_SIZE, action_size, duration_size,
                                replay_length, lr, layer_num, max_size,
                            )
                            na.model.load_state_dict(donor_sd)
                            seg_new = (len(eval_game.car_checkpoints[new_n - 1])
                                       + (1 if eval_game.car_goals[new_n - 1] is not None else 0))
                            na.init_epsilon_table(seg_new)
                            agents.append(na)

                            nt = DuelingDoubleDQN_DualHead(
                                INPUT_SIZE, action_size, duration_size,
                                0, lr, layer_num, max_size,
                            )
                            nt.init_epsilon_table(seg_new)
                            nt.model.load_state_dict(na.model.state_dict())
                            nt.model.eval()
                            target_nets.append(nt)
                            n = new_n

                            for ai in range(n):
                                seg_cnt = (len(eval_game.car_checkpoints[ai])
                                           + (1 if eval_game.car_goals[ai] is not None else 0))
                                agents[ai].epsilon = 1.0
                                agents[ai].init_epsilon_table(seg_cnt)
                                target_nets[ai].epsilon = 1.0
                                target_nets[ai].init_epsilon_table(seg_cnt)

                            run_tag = (f"par{parallel_workers}_curr_max{len(full_vehicle_paths)}_"
                                       f"st{n}ag_lr_{lr_str}_L{layer_num}_S{max_size}")
                            csv_file.close()
                            csv_filename = str(save_dir / f"train_{run_tag}_detail.csv")
                            csv_file = open(csv_filename, 'w', encoding='utf-8')
                            write_csv_header()

                            goal_counts.append(0)
                            ep_rewards.append([])
                            action_losses.append([])
                            duration_losses.append([])
                            ep_collision_cnts.append(0)
                            ep_timeout_cnts.append(0)
                            ep_goal_cnts_ep.append(0)
                            best_eval_times.append(float('inf'))
                            best_eval_paths.append(None)

                            curriculum_eval_wins = [0] * n
                            episodes_in_stage = 0
                            action_step_at_stage_start = action_step
                            stage_id += 1
                            broadcast('stage')

                            log("=" * 60)
                            log(f"[Curriculum] 단계 상승: {old_n}대 → {new_n}대  |  "
                                f"Agent{new_n} 초기 가중치 ← Agent{old_n} (직전 인덱스)")
                            log(f"  Parallel workers reset: {parallel_workers}")
                            ck_cur = str(save_dir / f"curriculum_{old_n}to{new_n}_ep{episode}_{run_tag}.pth")
                            torch.save(donor_sd, ck_cur)
                            log(f"  전이 소스 가중치 저장: {ck_cur}")
                            log("=" * 60)

    finally:
        for q in cmd_queues:
            q.put({'type': 'stop'})
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        total_time = time.time() - start_time
        log("\n" + "=" * 60)
        log("PARALLEL TRAINING COMPLETED")
        log(f"  Total Episodes : {episode}")
        log(f"  Total Goals    : {dict(enumerate(goal_counts, 1))}")
        log(f"  Training Time  : {total_time/60:.1f} min")
        log("=" * 60)

        log_file.close()
        csv_file.close()
        pygame.quit()

    for i in range(n):
        path = str(save_dir / f"agent{i+1}_final_ep{episode}_{run_tag}.pth")
        torch.save(agents[i].model.state_dict(), path)
        print(f"  Saved: {path}")

    return agents


# ============================================================
# 학습 함수 (다중 에이전트)
# ============================================================
def train_headless(vehicle_config_paths,
                   max_episode=300000,
                   action_size=8, duration_size=6,
                   replay_length=50000,
                   target_update=2000,
                   log_interval=100,
                   batch_size=512,
                   track_file=None,
                   layer_num=3, max_size=512, lr=0.0001,
                   max_time=60,
                   eval_interval=500,
                   eval_trials=10,
                   eval_max_time=60,
                   curriculum=True,
                   curriculum_perfect_evals_per_agent=10,
                   curriculum_min_episodes_per_stage=0,
                   curriculum_min_action_steps_per_stage=5000,
                   num_envs=1):
    """
    다중 에이전트 독립 DDDQN 학습.

    vehicle_config_paths: list[str] — 차량 설정 파일 경로 목록
      예) ["./vehicles/vehicle_1.json", "./vehicles/vehicle_2.json"]

    각 에이전트:
      - 독립 네트워크 + 리플레이 버퍼 + epsilon
      - 차량 JSON의 checkpoints 마지막 항목 = GOAL
      - 주행 중 다른 차량을 센서로 감지 (레이캐스팅에 bounding box 포함)

    curriculum=True 인 경우:
      - 처음에는 첫 번째 차량 JSON만으로 환경을 만들고(나머지 차량 없음), 단계적으로 경로를 늘림.
      - eval_interval마다 평가할 때, 에이전트 i가 eval_trials회 모두 골인하면 i에 대한
        ‘완벽 평가 1회’로 누적(연속일 필요 없음).
      - 현재 단계의 모든 활성 에이전트가 각각 curriculum_perfect_evals_per_agent회 이상 누적했고,
        단계 내 에피소드·환경 스텝이 curriculum_min_episodes_per_stage /
        curriculum_min_action_steps_per_stage 이상이면 다음 차량을 추가하고 환경을 재구성.
      - 단계가 바뀌면 누적 카운터는 0으로 리셋. 새 에이전트 가중치는 직전 인덱스에서 전이.
    """

    if track_file is None:
        track_file = _sim_asset("track_data.json")

    full_vehicle_paths = list(vehicle_config_paths)
    use_curriculum       = bool(curriculum) and len(full_vehicle_paths) > 1
    if use_curriculum:
        active_paths = full_vehicle_paths[:1]
    else:
        active_paths = full_vehicle_paths
    num_envs = max(int(num_envs), 1)

    pygame.init()
    games = [
        RacingGame(track_file, vehicle_configs=active_paths, headless=True)
        for _ in range(num_envs)
    ]
    eval_game = RacingGame(track_file, vehicle_configs=active_paths, headless=True)
    n    = games[0].n_agents

    # 에이전트 생성
    agents      = [DuelingDoubleDQN_DualHead(INPUT_SIZE, action_size, duration_size,
                                              replay_length, lr, layer_num, max_size)
                   for _ in range(n)]
    target_nets = [DuelingDoubleDQN_DualHead(INPUT_SIZE, action_size, duration_size,
                                              0, lr, layer_num, max_size)
                   for _ in range(n)]
    for i in range(n):
        segment_count = len(games[0].car_checkpoints[i]) + (1 if games[0].car_goals[i] is not None else 0)
        agents[i].init_epsilon_table(segment_count)
        target_nets[i].init_epsilon_table(segment_count)
        target_nets[i].model.load_state_dict(agents[i].model.state_dict())
        target_nets[i].model.eval()

    # 커리큘럼: 단계마다 에이전트별 ‘완벽 평가’ 누적 횟수 (연속 아님)
    curriculum_eval_wins        = [0] * n
    episodes_in_stage           = 0
    action_step_at_stage_start  = 0

    # 로그 / 모델 저장 경로 (스크립트 기준 폴더)
    save_dir     = _SIM_DIR / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    lr_str       = str(lr).replace(".", "_")
    if use_curriculum:
        run_tag = (f"curr_max{len(full_vehicle_paths)}_st{n}ag_lr_{lr_str}_"
                   f"L{layer_num}_S{max_size}")
    else:
        run_tag = f"ma_{n}agents_lr_{lr_str}_L{layer_num}_S{max_size}"
    log_filename = str(save_dir / f"train_{run_tag}.txt")
    log_file     = open(log_filename, 'w', encoding='utf-8')

    def log(msg):
        ts   = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}\n"
        log_file.write(line); log_file.flush(); print(msg)

    log("=" * 60)
    log(f"MULTI-AGENT DDDQN  |  {n} agents"
        + (f"  |  CURRICULUM → max {len(full_vehicle_paths)} vehicles" if use_curriculum else ""))
    if use_curriculum:
        log(f"  Curriculum: per-agent perfect evals>={curriculum_perfect_evals_per_agent} "
            f"(누적, 비연속), "
            f"min_episodes/stage>={curriculum_min_episodes_per_stage}, "
            f"min_action_steps/stage>={curriculum_min_action_steps_per_stage}")
    log(f"  Device : {agents[0].device}")
    log(f"  Input  : {INPUT_SIZE}  Actions: {action_size}  Durations: {duration_size}")
    log(f"  Layer  : {layer_num}  MaxSize: {max_size}  LR: {lr}")
    log(f"  Track  : {track_file}")
    log(f"  Rollout envs: {num_envs} (single process)")
    log(f"  MaxTime(train): {max_time}s  |  Eval every {eval_interval} ep  |  "
        f"Eval trials: {eval_trials}  |  Eval MaxTime: {eval_max_time}s")
    for i in range(n):
        cps = games[0].car_checkpoints[i]
        log(f"  Agent{i+1}: {len(cps)} nav cps → goal {games[0].car_goals[i]}")
    log("=" * 60)

    # ── 에피소드 루프용 초기 상태 ────────────────────────────────
    def _reset_episode_state(env_game):
        """에피소드 시작 시 에이전트별 상태 초기화"""
        std_cps = []
        dis_gps = []
        for i in range(n):
            nav_cps = env_game.car_checkpoints[i]
            cj      = env_game.car_jsons[i]
            sp_idx  = cj.get('start_point', i % max(len(env_game.start_positions), 1))
            sp      = (env_game.start_positions[sp_idx % len(env_game.start_positions)]
                       if env_game.start_positions else [env_game.width//2, env_game.height//2])
            first   = nav_cps[0] if nav_cps else env_game.car_goals[i]
            std_cps.append(list(first) if first else [sp[0], sp[1]])
            dis_gps.append(math.dist(sp, first) if first else 1.0)
        return std_cps, dis_gps

    def _get_segment_idx(agent_idx, processed_cp_cnts):
        segment_count = max(len(agents[agent_idx].epsilon_table), 1)
        return min(processed_cp_cnts[agent_idx], segment_count - 1)

    def _new_env_state(env_game):
        standard_cps, dis_gaps = _reset_episode_state(env_game)
        return {
            'game': env_game,
            'standard_cps': standard_cps,
            'dis_gaps': dis_gaps,
            'remaining_frames': [0] * n,
            'pending_states': [None] * n,
            'pending_actions': [0] * n,
            'pending_dur_idxs': [0] * n,
            'accumulated_rews': [0.0] * n,
            'processed_cp_cnts': [0] * n,
            'current_controls': [agents[i].get_real_action(0) for i in range(n)],
            'active': [True] * n,
            'ep_rew_buf': [0.0] * n,
            'ep_done_reason': [''] * n,
            'ep_cp_reached': [0] * n,
        }

    def _reset_env_state(env_state):
        env_state['game'].reset()
        env_state.update(_new_env_state(env_state['game']))

    env_states = [_new_env_state(g) for g in games]

    episode      = 0
    action_step  = 0
    goal_counts  = [0] * n
    ep_rewards   = [[] for _ in range(n)]
    ep_rew_buf   = [0.0] * n
    action_losses   = [[] for _ in range(n)]
    duration_losses = [[] for _ in range(n)]
    start_time   = time.time()
    last_log_time = start_time
    next_eval_episode = eval_interval

    # ── 에피소드별 상세 통계 ─────────────────────────────────────
    # 에피소드 종료 원인 카운터 (누적)
    ep_collision_cnts = [0] * n
    ep_timeout_cnts   = [0] * n
    ep_goal_cnts_ep   = [0] * n   # goal_counts와 별도로 interval 집계용

    # 에피소드 내 상태 추적
    ep_done_reason  = [''] * n   # 'collision' / 'timeout' / 'goal'
    ep_cp_reached   = [0]  * n   # 에피소드 내 CP 도달 수

    # CSV 상세 로그
    csv_filename = log_filename.replace('.txt', '_detail.csv')
    csv_file     = open(csv_filename, 'w', encoding='utf-8')

    # ── Evaluation 저장 추적 ─────────────────────────────────────
    best_eval_times = [float('inf')] * n   # 에이전트별 최고 기록 (낮을수록 좋음)
    best_eval_paths = [None] * n           # 현재 저장된 best-eval 모델 경로

    csv_file.write('episode,' +
                   'env_id,' +
                   ','.join(f'A{i+1}_reward,A{i+1}_done,A{i+1}_cp' for i in range(n)) +
                   '\n')
    csv_file.flush()

    log(f"\nTraining started  {time.strftime('%H:%M:%S')}")
    log("-" * 60)

    while episode < max_episode:
        for env_id, env_state in enumerate(env_states):
            if episode >= max_episode:
                break

            game = env_state['game']
            standard_cps = env_state['standard_cps']
            dis_gaps = env_state['dis_gaps']
            remaining_frames = env_state['remaining_frames']
            pending_states = env_state['pending_states']
            pending_actions = env_state['pending_actions']
            pending_dur_idxs = env_state['pending_dur_idxs']
            accumulated_rews = env_state['accumulated_rews']
            processed_cp_cnts = env_state['processed_cp_cnts']
            current_controls = env_state['current_controls']
            active = env_state['active']
            ep_rew_buf = env_state['ep_rew_buf']
            ep_done_reason = env_state['ep_done_reason']
            ep_cp_reached = env_state['ep_cp_reached']

            # ── Phase 1: 새 행동 결정 (duration 만료된 에이전트) ────────
            for i in range(n):
                if remaining_frames[i] <= 0 and active[i]:
                    state = get_data(game, i, standard_cps[i], dis_gaps[i])

                    # 이전 경험 저장 (있으면)
                    if pending_states[i] is not None:
                        agents[i].add_memory([
                            pending_states[i], pending_actions[i], pending_dur_idxs[i],
                            accumulated_rews[i], state, 0.0
                        ])
                        ep_rew_buf[i]     += accumulated_rews[i]
                        accumulated_rews[i] = 0.0

                    seg_idx = _get_segment_idx(i, processed_cp_cnts)
                    _, _, action, dur_idx, duration = agents[i].predict(state, segment_idx=seg_idx)
                    pending_states[i]    = state
                    pending_actions[i]   = action
                    pending_dur_idxs[i]  = dur_idx
                    current_controls[i]  = agents[i].get_real_action(action)
                    remaining_frames[i]  = duration

            # ── Phase 2: 환경 스텝 ──────────────────────────────────────
            prev_dists = [
                math.dist([game.cars[i].x, game.cars[i].y], standard_cps[i])
                if active[i] else 0.0
                for i in range(n)
            ]

            results = game.step(current_controls)
            action_step += 1

            # ── Phase 3: 결과 처리 ─────────────────────────────────────
            for i, result in enumerate(results):
                if not active[i]:
                    continue

                cp_r = 0
                curr_dist = math.dist([game.cars[i].x, game.cars[i].y], standard_cps[i])
                if result['cp_reached']:
                    cp_r = 100
                    ep_cp_reached[i] += 1
                    completed_seg_idx = _get_segment_idx(i, processed_cp_cnts)
                    agents[i].decay_epsilon(completed_seg_idx)
                    processed_cp_cnts[i] += 1
                    nav_cps = game.car_checkpoints[i]
                    if processed_cp_cnts[i] < len(nav_cps):
                        new_target = nav_cps[processed_cp_cnts[i]]
                    else:
                        new_target = game.car_goals[i]
                    if new_target:
                        dis_gaps[i] = math.dist(standard_cps[i], new_target)
                        standard_cps[i] = list(new_target)

                reward_features = get_reward_features(game, i)
                is_timeout = (game.current_time or 0) / 1000 > max_time

                frame_reward = get_frame_reward_from_features(
                    reward_features,
                    result['collision'], result['goal_reached'],
                    curr_dist, game.current_time or 0, max_time,
                    cp_r, dis_gaps[i], pending_actions[i],
                    game.cars[i].max_speed, prev_dists[i]
                )

                if result['red_light_crossed'] and not result['red_light_right_turn']:
                    frame_reward -= 50.0
                if is_timeout:
                    frame_reward -= 50.0

                accumulated_rews[i] += frame_reward
                remaining_frames[i] -= 1

                done_i = result['collision'] or result['goal_reached'] or is_timeout
                if done_i:
                    terminal_seg_idx = _get_segment_idx(i, processed_cp_cnts)
                    agents[i].decay_epsilon(terminal_seg_idx)
                    next_state = get_data(game, i, standard_cps[i], dis_gaps[i])
                    agents[i].add_memory([
                        pending_states[i], pending_actions[i], pending_dur_idxs[i],
                        accumulated_rews[i], next_state, 1.0
                    ])
                    ep_rew_buf[i] += accumulated_rews[i]
                    accumulated_rews[i] = 0.0
                    pending_states[i] = None
                    active[i] = False

                    if result['goal_reached']:
                        goal_counts[i] += 1
                        ep_goal_cnts_ep[i] += 1
                        ep_done_reason[i] = 'goal'
                    elif result['collision']:
                        ep_collision_cnts[i] += 1
                        ep_done_reason[i] = 'collision'
                    else:
                        ep_timeout_cnts[i] += 1
                        ep_done_reason[i] = 'timeout'

            # ── Phase 4: 학습 ───────────────────────────────────────────
            if action_step % 20 == 0:
                for i in range(n):
                    if len(agents[i].replay_memory) > batch_size:
                        tl, al, dl = agents[i].train_step(batch_size, target_nets[i])
                        action_losses[i].append(al)
                        duration_losses[i].append(dl)

            if action_step % target_update == 0:
                for i in range(n):
                    target_nets[i].model.load_state_dict(agents[i].model.state_dict())
                    target_nets[i].model.eval()

            # ── Phase 5: 에피소드 종료 ──────────────────────────────────
            if not any(active):
                episode += 1
                episodes_in_stage += 1

                for i in range(n):
                    ep_rewards[i].append(ep_rew_buf[i])

                # CSV 상세 기록 (매 에피소드)
                row = f"{episode},{env_id}"
                for i in range(n):
                    row += f",{ep_rew_buf[i]:.2f},{ep_done_reason[i]},{ep_cp_reached[i]}"
                csv_file.write(row + '\n')
                if episode % 10 == 0:
                    csv_file.flush()

                # 콘솔 + txt 로그 (log_interval 마다)
                if episode % log_interval == 0:
                    elapsed     = time.time() - last_log_time
                    eps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                    avg_rews    = [
                        np.mean(ep_rewards[i][-log_interval:]) if ep_rewards[i] else 0.0
                        for i in range(n)
                    ]
                    avg_a_losses = [
                        np.mean(action_losses[i][-100:]) if action_losses[i] else 0.0
                        for i in range(n)
                    ]

                    col_rate  = [ep_collision_cnts[i] / max(episode, 1) * 100 for i in range(n)]
                    to_rate   = [ep_timeout_cnts[i]   / max(episode, 1) * 100 for i in range(n)]
                    rew_str  = " | ".join(f"A{i+1}:{avg_rews[i]:7.1f}" for i in range(n))
                    loss_str = " | ".join(f"A{i+1}:{avg_a_losses[i]:.3f}" for i in range(n))
                    goal_str = " ".join(f"A{i+1}:{goal_counts[i]}" for i in range(n))
                    eps_str  = " ".join(
                        f"A{i+1}:S0={agents[i].get_segment_epsilon(0):.3f}"
                        for i in range(n)
                    )
                    col_str  = " | ".join(f"A{i+1}:{col_rate[i]:.0f}%" for i in range(n))
                    to_str   = " | ".join(f"A{i+1}:{to_rate[i]:.0f}%"  for i in range(n))

                    log(f"[Ep {episode:5d}] Goals:[{goal_str}] | ε:[{eps_str}] | "
                        f"Env:{num_envs} | MaxT:{max_time:2d}s | "
                        f"Reward:[{rew_str}] | ALoss:[{loss_str}] | "
                        f"{eps_per_sec:.1f}ep/s | {time.strftime('%H:%M:%S')}")
                    log(f"          Collision:[{col_str}] | Timeout:[{to_str}]")
                    last_log_time = time.time()

                # 주기적 체크포인트 저장 (1000 에피소드마다)
                if episode % 1000 == 0:
                    for i in range(n):
                        path = str(save_dir / f"agent{i+1}_ep{episode}_{run_tag}.pth")
                        torch.save(agents[i].model.state_dict(), path)
                    log(f"  [Ckpt] Ep{episode} 모델 저장 완료")

                stage_advanced = False
                # ── Evaluation (eval_interval 마다) ──────────────────────
                if episode >= next_eval_episode:
                    log(f"  [Eval] Ep{episode} — {eval_trials}회 평가 시작 "
                        f"(greedy, MaxT={eval_max_time}s) ...")

                    for i in range(n):
                        agents[i].model.eval()

                    eval_success = [0] * n
                    eval_times   = [[] for _ in range(n)]

                    for _trial in range(eval_trials):
                        eval_game.reset()
                        _std_cps, _dis_gps = _reset_episode_state(eval_game)
                        _rem   = [0]    * n
                        _ctrl  = [agents[i].get_real_action(0) for i in range(n)]
                        _alive = [True] * n
                        _cp_c  = [0]    * n

                        while any(_alive):
                            for i in range(n):
                                if _rem[i] <= 0 and _alive[i]:
                                    st = get_data(eval_game, i, _std_cps[i], _dis_gps[i])
                                    _, _, a, d, dur = agents[i].predict(
                                        st, segment_idx=0, greedy=True)
                                    _ctrl[i] = agents[i].get_real_action(a)
                                    _rem[i]  = dur

                            step_res = eval_game.step(_ctrl)

                            for i, res in enumerate(step_res):
                                if not _alive[i]:
                                    continue
                                _rem[i] -= 1
                                is_to = (eval_game.current_time or 0) / 1000 > eval_max_time

                                if res['cp_reached']:
                                    _cp_c[i] += 1
                                    nav_cps = eval_game.car_checkpoints[i]
                                    new_tgt = (nav_cps[_cp_c[i]]
                                               if _cp_c[i] < len(nav_cps)
                                               else eval_game.car_goals[i])
                                    if new_tgt:
                                        _dis_gps[i] = math.dist(_std_cps[i], new_tgt)
                                        _std_cps[i] = list(new_tgt)

                                if res['goal_reached'] or res['collision'] or is_to:
                                    if res['goal_reached']:
                                        eval_success[i] += 1
                                        eval_times[i].append(
                                            (eval_game.current_time or 0) / 1000.0)
                                    _alive[i] = False

                    for i in range(n):
                        agents[i].model.train()

                    for i in range(n):
                        sc = eval_success[i]
                        if sc == eval_trials:
                            avg_t = np.mean(eval_times[i])
                            if avg_t < best_eval_times[i]:
                                if best_eval_paths[i] and os.path.exists(best_eval_paths[i]):
                                    os.remove(best_eval_paths[i])
                                best_eval_times[i] = avg_t
                                new_path = str(save_dir /
                                    f"agent{i+1}_eval_best_{avg_t:.1f}s_{run_tag}.pth")
                                torch.save(agents[i].model.state_dict(), new_path)
                                best_eval_paths[i] = new_path
                                log(f"  [Eval] Agent{i+1} ✓ {eval_trials}/{eval_trials} "
                                    f"avg={avg_t:.1f}s  → saved  ({new_path})")
                            else:
                                log(f"  [Eval] Agent{i+1} ✓ {eval_trials}/{eval_trials} "
                                    f"avg={avg_t:.1f}s  (best={best_eval_times[i]:.1f}s, skip)")
                        else:
                            log(f"  [Eval] Agent{i+1} ✗ {sc}/{eval_trials} success — skip")

                    next_eval_episode += eval_interval

                    steps_in_stage = action_step - action_step_at_stage_start
                    mins_ok = (
                        episodes_in_stage >= curriculum_min_episodes_per_stage
                        and steps_in_stage >= curriculum_min_action_steps_per_stage
                    )
                    if use_curriculum and n < len(full_vehicle_paths):
                        for ci in range(n):
                            if eval_success[ci] == eval_trials:
                                curriculum_eval_wins[ci] += 1
                        wins_str = " ".join(
                            f"A{ci+1}:{curriculum_eval_wins[ci]}/{curriculum_perfect_evals_per_agent}"
                            for ci in range(n)
                        )
                        log(f"  [Curriculum] 완벽 평가 누적 ({wins_str})  |  "
                            f"단계 ep={episodes_in_stage}, env_step={steps_in_stage}")

                        all_agents_ready = all(
                            curriculum_eval_wins[ci] >= curriculum_perfect_evals_per_agent
                            for ci in range(n)
                        )
                        if all_agents_ready and mins_ok and n < len(full_vehicle_paths):
                            old_n = n
                            new_n = old_n + 1
                            donor_sd = agents[old_n - 1].model.state_dict()
                            stage_paths = full_vehicle_paths[:new_n]
                            eval_game = RacingGame(
                                track_file,
                                vehicle_configs=stage_paths,
                                headless=True,
                            )
                            games = [
                                RacingGame(track_file, vehicle_configs=stage_paths, headless=True)
                                for _ in range(num_envs)
                            ]
                            na = DuelingDoubleDQN_DualHead(
                                INPUT_SIZE, action_size, duration_size,
                                replay_length, lr, layer_num, max_size,
                            )
                            na.model.load_state_dict(donor_sd)
                            seg_new = (len(eval_game.car_checkpoints[new_n - 1])
                                       + (1 if eval_game.car_goals[new_n - 1] is not None else 0))
                            na.init_epsilon_table(seg_new)
                            agents.append(na)
                            nt = DuelingDoubleDQN_DualHead(
                                INPUT_SIZE, action_size, duration_size,
                                0, lr, layer_num, max_size,
                            )
                            nt.init_epsilon_table(seg_new)
                            nt.model.load_state_dict(na.model.state_dict())
                            nt.model.eval()
                            target_nets.append(nt)
                            n = new_n

                            for ai in range(n):
                                seg_cnt = (len(eval_game.car_checkpoints[ai])
                                           + (1 if eval_game.car_goals[ai] is not None else 0))
                                agents[ai].epsilon = 1.0
                                agents[ai].init_epsilon_table(seg_cnt)
                                target_nets[ai].epsilon = 1.0
                                target_nets[ai].init_epsilon_table(seg_cnt)

                            run_tag = (f"curr_max{len(full_vehicle_paths)}_st{n}ag_lr_{lr_str}_"
                                       f"L{layer_num}_S{max_size}")
                            csv_file.close()
                            csv_filename = str(save_dir / f"train_{run_tag}_detail.csv")
                            csv_file = open(csv_filename, 'w', encoding='utf-8')
                            csv_file.write(
                                'episode,env_id,'
                                + ','.join(
                                    f'A{i+1}_reward,A{i+1}_done,A{i+1}_cp' for i in range(n))
                                + '\n'
                            )
                            csv_file.flush()

                            goal_counts.append(0)
                            ep_rewards.append([])
                            action_losses.append([])
                            duration_losses.append([])
                            ep_collision_cnts.append(0)
                            ep_timeout_cnts.append(0)
                            ep_goal_cnts_ep.append(0)
                            best_eval_times.append(float('inf'))
                            best_eval_paths.append(None)

                            curriculum_eval_wins = [0] * n
                            episodes_in_stage = 0
                            action_step_at_stage_start = action_step
                            env_states = [_new_env_state(g) for g in games]
                            stage_advanced = True

                            log("=" * 60)
                            log(f"[Curriculum] 단계 상승: {old_n}대 → {new_n}대  |  "
                                f"Agent{new_n} 초기 가중치 ← Agent{old_n} (직전 인덱스)")
                            log(f"  Rollout envs reset: {num_envs}")
                            log(f"  새 CSV: {csv_filename}")
                            ck_cur = str(save_dir / f"curriculum_{old_n}to{new_n}_ep{episode}_{run_tag}.pth")
                            torch.save(donor_sd, ck_cur)
                            log(f"  전이 소스 가중치 저장: {ck_cur}")
                            log("=" * 60)

                if stage_advanced:
                    break

                _reset_env_state(env_state)

    # ── 학습 완료 ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    log("\n" + "=" * 60)
    log("TRAINING COMPLETED")
    log(f"  Total Episodes : {episode}")
    log(f"  Total Goals    : {dict(enumerate(goal_counts, 1))}")
    log(f"  Training Time  : {total_time/60:.1f} min")
    log("=" * 60)
    log_file.close()
    csv_file.close()
    pygame.quit()

    # 최종 모델 저장
    for i in range(n):
        path = str(save_dir / f"agent{i+1}_final_ep{episode}_{run_tag}.pth")
        torch.save(agents[i].model.state_dict(), path)
        print(f"  Saved: {path}")

    return agents


# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_num",  type=int,   default=4)
    parser.add_argument("--max_size",   type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--max_time",      type=int,   default=60)
    parser.add_argument("--eval_interval", type=int,   default=500)
    parser.add_argument("--eval_trials",   type=int,   default=10)
    parser.add_argument("--eval_max_time", type=int,   default=60)
    parser.add_argument("--num_envs",       type=int,   default=1,
                        help="한 프로세스 안에서 순차 rollout할 RacingGame 환경 수")
    parser.add_argument("--parallel_workers", type=int, default=4,
                        help="multiprocessing rollout worker 수. 0이면 기존 단일 프로세스 경로 사용")
    parser.add_argument("--rollout_chunk_steps", type=int, default=32,
                        help="parallel worker가 learner로 전송하기 전 모을 최대 환경 step 수")
    parser.add_argument("--train_every_steps", type=int, default=20,
                        help="parallel learner의 gradient update 주기(action_step 기준)")
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help="parallel learner가 학습 타이밍마다 반복할 gradient step 수")
    parser.add_argument("--sync_interval_steps", type=int, default=1000,
                        help="parallel worker에 최신 모델/epsilon을 동기화하는 action_step 주기")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="1대로 시작해 조건 충족 시 차량을 한 대씩 추가 (최대 vehicle 리스트 길이)",
        default = True
    )
    parser.add_argument(
        "--curriculum_perfect_evals",
        type=int,
        default=10,
        help="커리큘럼: 현재 단계에서 각 활성 에이전트가 ‘eval_trials 전부 골인’ 평가를 몇 번 누적하면 다음 차 추가(비연속)",
    )
    parser.add_argument(
        "--curriculum_min_episodes",
        type=int,
        default=0,
        help="커리큘럼: 다음 단계 진입 시 단계 내 최소 완료 에피소드 수",
    )
    parser.add_argument(
        "--curriculum_min_action_steps",
        type=int,
        default=5000,
        help="커리큘럼: 다음 단계 진입 시 단계 내 최소 환경 step(action_step)",
    )
    args = parser.parse_args()

    common_kwargs = dict(
        vehicle_config_paths=[
            _sim_asset("vehicles", "vehicle_1.json"),
            _sim_asset("vehicles", "vehicle_2.json"),
            _sim_asset("vehicles", "vehicle_3.json"),
            _sim_asset("vehicles", "vehicle_4.json"),
        ],
        max_episode=90000,
        action_size=8,
        duration_size=6,
        replay_length=50000,
        target_update=2000,
        log_interval=100,
        layer_num=args.layer_num,
        max_size=args.max_size,
        lr=args.lr,
        batch_size=args.batch_size,
        track_file=_sim_asset("track_data.json"),
        max_time=args.max_time,
        eval_interval=args.eval_interval,
        eval_trials=args.eval_trials,
        eval_max_time=args.eval_max_time,
        curriculum=args.curriculum,
        curriculum_perfect_evals_per_agent=args.curriculum_perfect_evals,
        curriculum_min_episodes_per_stage=args.curriculum_min_episodes,
        curriculum_min_action_steps_per_stage=args.curriculum_min_action_steps,
    )

    if args.parallel_workers > 0:
        train_headless_parallel(
            **common_kwargs,
            parallel_workers=args.parallel_workers,
            rollout_chunk_steps=args.rollout_chunk_steps,
            train_every_steps=args.train_every_steps,
            gradient_steps=args.gradient_steps,
            sync_interval_steps=args.sync_interval_steps,
        )
    else:
        train_headless(
            **common_kwargs,
            num_envs=args.num_envs,
        )

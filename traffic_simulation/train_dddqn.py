"""
다중 에이전트 학습 코드 — Dueling Double DQN + Dual Head
- 에이전트마다 독립 네트워크 / 리플레이 버퍼 / epsilon
- 각 차량 JSON의 checkpoints 마지막 항목 = GOAL
- 모든 에이전트가 한 환경에서 동시에 주행하며 서로를 센서로 인식
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
    sensors = []
    for i in range(8):
        angle  = car.angle + i * (math.pi / 4)
        dist_n = game._cast_ray_for_car(car_idx, car.x, car.y, angle)
        sensors.append(dist_n * car.sensor_range)
    return sensors


def get_data(game, car_idx=0, standard_cp=None, dis_gap=None):
    """
    car_idx 에이전트의 상태 벡터 생성 (INPUT_SIZE=30 고정)

    구성:
      sensors(8) + [cos,sin,speed,vx,vy,dist,angle,drift](8)
      + car_features(8) + [dir_x,dir_y,is_inter,tl_exists,tl_state,right_turnable](6)
    """
    car   = game.cars[car_idx]
    angle = car.angle

    cos_angle = (math.cos(angle) + 1) / 2
    sin_angle = (math.sin(angle) + 1) / 2
    speed     = car.speed / car.max_speed
    vel_x     = car.velocity_x / car.max_speed
    vel_y     = car.velocity_y / car.max_speed
    sensors   = [s / car.sensor_range for s in get_sensors(game, car_idx)]

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
# ============================================================
def get_frame_reward(state, is_collision, is_goal,
                     curr_distance, curr_time, max_time,
                     cp_reward, dis_gap, action_index,
                     max_speed, prev_distance):
    reward = 0.0

    if is_collision: return -500.0
    if is_goal:      return  500.0

    # 타임아웃 패널티
    time_ratio = curr_time / 1000 / max_time
    reward -= time_ratio * 0.5

    # 목표 접근 보상
    if dis_gap > 0:
        reward += (prev_distance - curr_distance) * 0.5

    # 속도 보상
    reward += state[2] * max_speed * 0.002

    # 체크포인트 보상
    reward += cp_reward

    # ── 역주행 패널티 ─────────────────────────────────────────────
    vel_x_n = state[3]
    vel_y_n = state[4]
    speed_n = state[2]
    cos_a   = state[8] * 2.0 - 1.0
    sin_a   = state[9] * 2.0 - 1.0

    # ① 후진 (차량 heading 과 속도 반대)
    heading_vel_dot = cos_a * vel_x_n + sin_a * vel_y_n
    if heading_vel_dot < -0.1 and speed_n > 0.05:
        reward -= 1.0

    # ② 차선 침범 역주행 (앞으로 가는데 도로 방향과 반대)
    dir_x = state[24]
    dir_y = state[25]
    if dir_x != 0.0 or dir_y != 0.0:
        road_vel_dot = vel_x_n * dir_x + vel_y_n * dir_y
        if heading_vel_dot >= 0.0 and road_vel_dot < -0.1:
            reward -= 5.0

    # ── 신호 패널티 ───────────────────────────────────────────────
    # 정지선 위반 패널티는 execute 루프에서 이벤트 기반으로 추가
    tl_exists      = state[27]
    tl_state       = state[28]
    right_turnable = state[29]
    if tl_exists == 1:
        if tl_state == 0 and right_turnable == 0:   # 빨간불 + 직진/좌회전
            if speed_n <= 0.05:
                reward += 0.5   # 정지 준수 보상
        elif tl_state == 1:                          # 노란불
            if speed_n > 0.2:
                reward -= 2.0 * (speed_n - 0.2)
            else:
                reward += 0.2

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
# 에이전트
# ============================================================
class DuelingDoubleDQN_DualHead:
    def __init__(self, input_size=INPUT_SIZE, action_size=8, duration_size=20,
                 replay_memory_length=100000,
                 lr=0.0001, layer_num=3, max_size=512):
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size   = action_size
        self.duration_size = duration_size

        self.model = DuelingDualHeadNetwork(
            input_size, action_size, duration_size, layer_num, max_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.gamma        = 0.99
        self.epsilon      = 1.0
        self.epsilon_min  = 0.1
        self.epsilon_decay = 0.9995

        self.replay_memory        = []
        self.replay_memory_length = replay_memory_length
        self.duration_map = {i: (i + 1) * 3 for i in range(duration_size)}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

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
            action  = (aq.argmax().item()
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
        batch       = random.sample(self.replay_memory, batch_size)
        states      = torch.tensor([m[0] for m in batch], dtype=torch.float32).to(self.device)
        actions     = torch.tensor([m[1] for m in batch], dtype=torch.int64).to(self.device)
        dur_idxs    = torch.tensor([m[2] for m in batch], dtype=torch.int64).to(self.device)
        rewards     = torch.tensor([m[3] for m in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([m[4] for m in batch], dtype=torch.float32).to(self.device)
        dones       = torch.tensor([m[5] for m in batch], dtype=torch.float32).to(self.device)

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
# 학습 함수 (다중 에이전트)
# ============================================================
def train_headless(vehicle_config_paths,
                   max_episode=300000,
                   action_size=8, duration_size=6,
                   replay_length=50000,
                   target_update=2000,
                   log_interval=100,
                   batch_size=512,
                   track_file="./track_data.json",
                   layer_num=3, max_size=512, lr=0.0001,
                   max_time=30):
    """
    다중 에이전트 독립 DDDQN 학습.

    vehicle_config_paths: list[str] — 차량 설정 파일 경로 목록
      예) ["./vehicles/vehicle_1.json", "./vehicles/vehicle_2.json"]

    각 에이전트:
      - 독립 네트워크 + 리플레이 버퍼 + epsilon
      - 차량 JSON의 checkpoints 마지막 항목 = GOAL
      - 주행 중 다른 차량을 센서로 감지 (레이캐스팅에 bounding box 포함)
    """

    pygame.init()
    game = RacingGame(track_file, vehicle_configs=vehicle_config_paths, headless=True)
    n    = game.n_agents

    # 에이전트 생성
    agents      = [DuelingDoubleDQN_DualHead(INPUT_SIZE, action_size, duration_size,
                                              replay_length, lr, layer_num, max_size)
                   for _ in range(n)]
    target_nets = [DuelingDoubleDQN_DualHead(INPUT_SIZE, action_size, duration_size,
                                              0, lr, layer_num, max_size)
                   for _ in range(n)]
    for i in range(n):
        target_nets[i].model.load_state_dict(agents[i].model.state_dict())
        target_nets[i].model.eval()

    # 로그
    lr_str       = str(lr).replace(".", "_")
    log_filename = f"train_ma_{n}agents_lr_{lr_str}_L{layer_num}_S{max_size}.txt"
    log_file     = open(log_filename, 'w', encoding='utf-8')

    def log(msg):
        ts   = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}\n"
        log_file.write(line); log_file.flush(); print(msg)

    log("=" * 60)
    log(f"MULTI-AGENT DDDQN  |  {n} agents")
    log(f"  Device : {agents[0].device}")
    log(f"  Input  : {INPUT_SIZE}  Actions: {action_size}  Durations: {duration_size}")
    log(f"  Layer  : {layer_num}  MaxSize: {max_size}  LR: {lr}")
    log(f"  Track  : {track_file}")
    for i in range(n):
        cps = game.car_checkpoints[i]
        log(f"  Agent{i+1}: {len(cps)} nav cps → goal {game.car_goals[i]}")
    log("=" * 60)

    # ── 에피소드 루프용 초기 상태 ────────────────────────────────
    def _reset_episode_state():
        """에피소드 시작 시 에이전트별 상태 초기화"""
        std_cps = []
        dis_gps = []
        for i in range(n):
            nav_cps = game.car_checkpoints[i]
            cj      = game.car_jsons[i]
            sp_idx  = cj.get('start_point', i % max(len(game.start_positions), 1))
            sp      = (game.start_positions[sp_idx % len(game.start_positions)]
                       if game.start_positions else [game.width//2, game.height//2])
            first   = nav_cps[0] if nav_cps else game.car_goals[i]
            std_cps.append(list(first) if first else [sp[0], sp[1]])
            dis_gps.append(math.dist(sp, first) if first else 1.0)
        return std_cps, dis_gps

    standard_cps, dis_gaps = _reset_episode_state()

    # 에이전트별 duration 추적
    remaining_frames  = [0]    * n
    pending_states    = [None] * n
    pending_actions   = [0]    * n
    pending_dur_idxs  = [0]    * n
    accumulated_rews  = [0.0]  * n
    processed_cp_cnts = [0]    * n
    current_controls  = [agents[i].get_real_action(0) for i in range(n)]
    active            = [True] * n

    episode      = 0
    action_step  = 0
    goal_counts  = [0] * n
    ep_rewards   = [[] for _ in range(n)]
    ep_rew_buf   = [0.0] * n
    action_losses   = [[] for _ in range(n)]
    duration_losses = [[] for _ in range(n)]
    start_time   = time.time()
    last_log_time = start_time

    log(f"\nTraining started  {time.strftime('%H:%M:%S')}")
    log("-" * 60)

    while episode < max_episode:

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

                _, _, action, dur_idx, duration = agents[i].predict(state)
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

        results    = game.step(current_controls)
        action_step += 1

        # ── Phase 3: 결과 처리 ─────────────────────────────────────
        for i, result in enumerate(results):
            if not active[i]:
                continue

            # 체크포인트 진행
            cp_r = 0
            if result['cp_reached']:
                cp_r = 50
                processed_cp_cnts[i] += 1
                nav_cps = game.car_checkpoints[i]
                if processed_cp_cnts[i] < len(nav_cps):
                    new_target = nav_cps[processed_cp_cnts[i]]
                else:
                    new_target = game.car_goals[i]
                if new_target:
                    dis_gaps[i]    = math.dist(standard_cps[i], new_target)
                    standard_cps[i] = list(new_target)

            curr_dist  = math.dist([game.cars[i].x, game.cars[i].y], standard_cps[i])
            frame_state = get_data(game, i, standard_cps[i], dis_gaps[i])
            is_timeout  = (game.current_time or 0) / 1000 > max_time

            frame_reward = get_frame_reward(
                frame_state,
                result['collision'], result['goal_reached'],
                curr_dist, game.current_time or 0, max_time,
                cp_r, dis_gaps[i], pending_actions[i],
                game.cars[i].max_speed, prev_dists[i]
            )

            # 정지선 위반 패널티 (이벤트 기반)
            if result['red_light_crossed'] and not result['red_light_right_turn']:
                frame_reward -= 300.0

            # 타임아웃 추가 패널티
            if is_timeout:
                frame_reward -= 5000.0

            accumulated_rews[i] += frame_reward
            remaining_frames[i] -= 1

            done_i = result['collision'] or result['goal_reached'] or is_timeout

            if done_i:
                next_state = get_data(game, i, standard_cps[i], dis_gaps[i])
                agents[i].add_memory([
                    pending_states[i], pending_actions[i], pending_dur_idxs[i],
                    accumulated_rews[i], next_state, 1.0
                ])
                ep_rew_buf[i]     += accumulated_rews[i]
                accumulated_rews[i] = 0.0
                pending_states[i]   = None
                active[i]           = False

                if result['goal_reached']:
                    goal_counts[i] += 1

        # ── Phase 4: 학습 ───────────────────────────────────────────
        if action_step % 5 == 0:
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

            for i in range(n):
                ep_rewards[i].append(ep_rew_buf[i])
                agents[i].decay_epsilon()

            # 로그
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
                rew_str  = " | ".join(f"A{i+1}:{avg_rews[i]:7.1f}" for i in range(n))
                loss_str = " | ".join(f"A{i+1}:{avg_a_losses[i]:.3f}" for i in range(n))
                goal_str = " ".join(f"A{i+1}:{goal_counts[i]}" for i in range(n))
                eps_str  = " ".join(f"A{i+1}:{agents[i].epsilon:.3f}" for i in range(n))
                log(f"[Ep {episode:5d}] Goals:[{goal_str}] | ε:[{eps_str}] | "
                    f"Reward:[{rew_str}] | ALoss:[{loss_str}] | "
                    f"{eps_per_sec:.1f}ep/s | {time.strftime('%H:%M:%S')}")
                last_log_time = time.time()

            # 리셋
            game.reset()
            standard_cps, dis_gaps = _reset_episode_state()
            remaining_frames  = [0]    * n
            pending_states    = [None] * n
            pending_actions   = [0]    * n
            pending_dur_idxs  = [0]    * n
            accumulated_rews  = [0.0]  * n
            processed_cp_cnts = [0]    * n
            current_controls  = [agents[i].get_real_action(0) for i in range(n)]
            active            = [True] * n
            ep_rew_buf        = [0.0]  * n

    # ── 학습 완료 ─────────────────────────────────────────────────
    total_time = time.time() - start_time
    log("\n" + "=" * 60)
    log("TRAINING COMPLETED")
    log(f"  Total Episodes : {episode}")
    log(f"  Total Goals    : {dict(enumerate(goal_counts, 1))}")
    log(f"  Training Time  : {total_time/60:.1f} min")
    log("=" * 60)
    log_file.close()
    pygame.quit()

    # 최종 모델 저장
    for i in range(n):
        path = f"agent{i+1}_final_ep{episode}.pth"
        torch.save(agents[i].model.state_dict(), path)
        print(f"  Saved: {path}")

    return agents


# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_num",  type=int,   default=3)
    parser.add_argument("--max_size",   type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--max_time",   type=int,   default=30)
    args = parser.parse_args()

    train_headless(
        vehicle_config_paths=[
            _sim_asset("vehicles", "vehicle_1.json"),
            _sim_asset("vehicles", "vehicle_2.json"),
            _sim_asset("vehicles", "vehicle_3.json"),
            _sim_asset("vehicles", "vehicle_4.json"),
        ],
        max_episode   = 300000,
        action_size   = 8,
        duration_size = 6,
        replay_length = 50000,
        target_update = 2000,
        log_interval  = 100,
        layer_num     = args.layer_num,
        max_size      = args.max_size,
        lr            = args.lr,
        batch_size    = args.batch_size,
        track_file    = "./track_data.json",
        max_time      = args.max_time,
    )

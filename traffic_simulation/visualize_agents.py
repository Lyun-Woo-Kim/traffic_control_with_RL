"""
visualize_agents.py — 학습된 모델 시각화 도구

사용법:
    # 4개 에이전트 모두 지정
    python visualize_agents.py \\
        --agent1 checkpoints/agent1_eval_best_12.5s_...pth \\
        --agent2 checkpoints/agent2_eval_best_10.3s_...pth \\
        --agent3 checkpoints/agent3_eval_best_8.1s_...pth \\
        --agent4 checkpoints/agent4_eval_best_7.5s_...pth

    # 일부만 지정 (미지정 에이전트는 랜덤 행동)
    python visualize_agents.py --agent3 path/to/model.pth

    # 네트워크 구조가 기본값(L4 S512)과 다를 경우
    python visualize_agents.py --agent1 model.pth --layer_num 3 --max_size 1024

조작:
    ESC / Q   : 종료
    R         : 에피소드 즉시 리셋
    SPACE     : 일시정지 / 재개
"""

import pygame
import math
import sys
import os
import argparse

import torch
import numpy as np
from pathlib import Path

from traffic_env import RacingGame
from train_dddqn import (
    DuelingDoubleDQN_DualHead,
    get_data,
    INPUT_SIZE,
    _sim_asset,
    _SIM_DIR,
)

# ── 색상 팔레트 ───────────────────────────────────────────────────
AGENT_COLORS = [
    (0,   120, 255),   # A1 파랑
    (255,  80,   0),   # A2 주황
    (0,   200,  50),   # A3 초록
    (200,   0, 200),   # A4 보라
]
STATUS_COLORS = {
    'running':   (220, 220, 220),
    'goal':      (0,   230,  80),
    'collision': (255,  60,  60),
    'timeout':   (255, 200,   0),
}

WIN_W, WIN_H = 1280, 800


# ── 모델 로드 ────────────────────────────────────────────────────
def load_agent(model_path, action_size, duration_size, layer_num, max_size):
    agent = DuelingDoubleDQN_DualHead(
        INPUT_SIZE, action_size, duration_size, 0, 0.0003, layer_num, max_size
    )
    state_dict = torch.load(model_path, map_location=agent.device,
                            weights_only=True)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent


# ── 렌더링 헬퍼 ──────────────────────────────────────────────────
def _draw_scene(surface, game, font_s, zoom=1.0):
    """트랙·신호·차량·체크포인트·목표를 surface에 직접 그린다."""
    cx, cy = game.camera_x, game.camera_y

    # 배경 + 트랙
    surface.fill((245, 245, 240))
    surface.blit(game.track_surface, (-cx, -cy))

    # 시작 위치
    for idx, sp in enumerate(game.start_positions):
        sx, sy = int(sp[0] - cx), int(sp[1] - cy)
        pygame.draw.circle(surface, (0, 180, 0), (sx, sy), 14)
        lbl = font_s.render(f"S{idx}", True, (255, 255, 255))
        surface.blit(lbl, lbl.get_rect(center=(sx, sy)))

    # 전체 체크포인트 (트랙 순번)
    for i, cp in enumerate(game.checkpoints):
        sx, sy = int(cp[0] - cx), int(cp[1] - cy)
        pygame.draw.circle(surface, (255, 230, 0), (sx, sy), 9)
        lbl = font_s.render(str(i), True, (0, 0, 0))
        surface.blit(lbl, lbl.get_rect(center=(sx, sy)))

    # 에이전트별 목표 (색상 테두리)
    for i, goal in enumerate(game.car_goals):
        if goal:
            gx, gy = int(goal[0] - cx), int(goal[1] - cy)
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            pygame.draw.circle(surface, color, (gx, gy), 14, 3)
            lbl = font_s.render(f"G{i+1}", True, color)
            surface.blit(lbl, (gx + 12, gy - 8))

    # 신호등
    TL_R = (255, 30, 30);  TL_Y = (255, 220, 0);  TL_G = (0, 220, 0)
    OFF_R = (80, 0, 0);    OFF_Y = (80, 70, 0);   OFF_G = (0, 70, 0)
    for tl in game.traffic_lights:
        x = int(tl['pos'][0] - cx)
        y = int(tl['pos'][1] - cy)
        pygame.draw.rect(surface, (10, 10, 10), (x - 14, y - 38, 28, 76), border_radius=5)
        st = tl.get('state', 'red')
        pygame.draw.circle(surface, TL_R if st == 'red'    else OFF_R, (x, y - 25), 9)
        pygame.draw.circle(surface, TL_Y if st == 'yellow' else OFF_Y, (x, y),      9)
        pygame.draw.circle(surface, TL_G if st == 'green'  else OFF_G, (x, y + 25), 9)

    # 차량
    for i, car in enumerate(game.cars):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        car.draw(surface, (cx, cy), color=color)
        lx = int(car.x - cx)
        ly = int(car.y - cy - car.length - 4)
        lbl = font_s.render(f"A{i+1}", True, color)
        surface.blit(lbl, lbl.get_rect(center=(lx, ly)))


def _draw_hud(surface, font_s, font_m, font_l,
              game, agents, model_names, status,
              goal_cnts, episode, max_time, paused):
    """우측 HUD 패널 + 상단 정보 바를 그린다."""
    n = game.n_agents

    # ── 상단 상태 바 ────────────────────────────────────────────
    elapsed_s = (game.current_time or 0) / 1000.0
    remaining = max(0.0, max_time - elapsed_s)
    bar_color = (255, 80, 80) if remaining < 5 else (80, 200, 80)
    top_txt = (f"Episode {episode}    "
               f"Time: {elapsed_s:.1f}s / {max_time}s    "
               f"Remaining: {remaining:.1f}s")
    if paused:
        top_txt += "    [ PAUSED ]"
    lbl = font_m.render(top_txt, True, (20, 20, 20))
    pygame.draw.rect(surface, (240, 240, 240), (0, 0, WIN_W, 30))
    surface.blit(lbl, (10, 5))

    # 타임 바
    ratio = min(elapsed_s / max_time, 1.0)
    pygame.draw.rect(surface, (180, 180, 180), (0, 28, WIN_W, 6))
    pygame.draw.rect(surface, bar_color,       (0, 28, int(WIN_W * ratio), 6))

    # ── 우측 에이전트 패널 ───────────────────────────────────────
    panel_x = WIN_W - 260
    panel_w = 255
    panel_h = 34 + n * 100
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((20, 20, 20, 180))
    surface.blit(panel_surf, (panel_x, 40))

    title = font_m.render("Agents", True, (255, 255, 255))
    surface.blit(title, (panel_x + 10, 46))

    for i in range(n):
        base_y = 40 + 34 + i * 100
        color  = AGENT_COLORS[i % len(AGENT_COLORS)]
        st     = status[i]
        st_col = STATUS_COLORS.get(st, (200, 200, 200))
        car    = game.cars[i] if i < len(game.cars) else None

        # 에이전트 헤더
        pygame.draw.rect(surface, color, (panel_x, base_y, panel_w, 24))
        hdr = font_m.render(f"Agent {i+1}  Goals: {goal_cnts[i]}", True, (0, 0, 0))
        surface.blit(hdr, (panel_x + 6, base_y + 3))

        # 상태
        pygame.draw.rect(surface, st_col, (panel_x, base_y + 24, panel_w, 20))
        st_lbl = font_s.render(st.upper(), True, (0, 0, 0))
        surface.blit(st_lbl, (panel_x + 6, base_y + 27))

        # 속도
        speed_val = f"{car.speed * 0.36:.1f} km/h" if car else "—"
        sp_lbl = font_s.render(f"Speed : {speed_val}", True, (220, 220, 220))
        surface.blit(sp_lbl, (panel_x + 6, base_y + 48))

        # 모델 이름 (짧게)
        mname = model_names[i] if i < len(model_names) else None
        if mname:
            short = os.path.basename(mname)
            if len(short) > 30:
                short = "…" + short[-27:]
        else:
            short = "(no model — random)"
        mn_lbl = font_s.render(short, True, (160, 160, 160))
        surface.blit(mn_lbl, (panel_x + 6, base_y + 66))

    # ── 하단 조작 안내 ───────────────────────────────────────────
    guide = "ESC/Q: 종료    R: 리셋    SPACE: 일시정지"
    g_lbl = font_s.render(guide, True, (80, 80, 80))
    surface.blit(g_lbl, (10, WIN_H - 20))


# ── 에피소드 초기화 헬퍼 ─────────────────────────────────────────
def _make_episode_state(game, agents):
    n = game.n_agents
    game.reset()
    game.camera_x = game.camera_y = 0

    std_cps, dis_gps = [], []
    for i in range(n):
        nav_cps = game.car_checkpoints[i]
        cj      = game.car_jsons[i]
        sp_idx  = cj.get('start_point', i % max(len(game.start_positions), 1))
        sp      = (game.start_positions[sp_idx % len(game.start_positions)]
                   if game.start_positions else [game.width // 2, game.height // 2])
        first   = nav_cps[0] if nav_cps else game.car_goals[i]
        std_cps.append(list(first) if first else [sp[0], sp[1]])
        dis_gps.append(math.dist(sp, first) if first else 1.0)

    return {
        'std_cps': std_cps,
        'dis_gps': dis_gps,
        'rem':     [0]          * n,
        'ctrl':    [agents[i].get_real_action(0) for i in range(n)],
        'active':  [True]       * n,
        'cp_cnt':  [0]          * n,
        'status':  ['running']  * n,
    }


# ── 메인 ─────────────────────────────────────────────────────────
def run_visualization(vehicle_config_paths, model_paths,
                      action_size=8, duration_size=6,
                      layer_num=4, max_size=512,
                      max_time=40,
                      track_file=None):

    if track_file is None:
        track_file = _sim_asset("track_data.json")

    # pygame 초기화 + 디스플레이 생성
    pygame.init()
    win = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Agent Visualizer")
    clock = pygame.time.Clock()

    font_s = pygame.font.SysFont('arial,sans-serif', 13)
    font_m = pygame.font.SysFont('arial,sans-serif', 16, bold=True)
    font_l = pygame.font.SysFont('arial,sans-serif', 22, bold=True)

    # 게임 환경 (headless=True → step()이 내부 draw/flip 안 함)
    game = RacingGame(track_file, vehicle_configs=vehicle_config_paths, headless=True)
    game.screen = win        # 헤드리스지만 win을 실제 렌더 대상으로 사용
    game.fps    = 60
    n = game.n_agents

    # 에이전트 + 모델 로드
    agents     = []
    has_model  = []
    for i in range(n):
        mp = model_paths[i] if i < len(model_paths) else None
        if mp and os.path.exists(mp):
            agent = load_agent(mp, action_size, duration_size, layer_num, max_size)
            agents.append(agent)
            has_model.append(True)
            print(f"  Agent{i+1}: loaded  {mp}")
        else:
            agent = DuelingDoubleDQN_DualHead(
                INPUT_SIZE, action_size, duration_size, 0, 0.0003, layer_num, max_size
            )
            agent.model.eval()
            agents.append(agent)
            has_model.append(False)
            if mp:
                print(f"  Agent{i+1}: NOT FOUND ({mp}) → 랜덤 행동")
            else:
                print(f"  Agent{i+1}: 모델 미지정 → 랜덤 행동")

    episode   = 1
    goal_cnts = [0] * n
    paused    = False

    ep = _make_episode_state(game, agents)

    print(f"\n시작 — ESC/Q: 종료  R: 리셋  SPACE: 일시정지\n")

    while True:
        # ── 이벤트 ──────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_r:
                    ep = _make_episode_state(game, agents)
                    episode += 1
                    print(f"  [Reset] Episode {episode}")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"  [{'Pause' if paused else 'Resume'}]")

        if paused:
            _draw_scene(win, game, font_s)
            _draw_hud(win, font_s, font_m, font_l,
                      game, agents, model_paths, ep['status'],
                      goal_cnts, episode, max_time, paused=True)
            pygame.display.flip()
            clock.tick(30)
            continue

        # ── 행동 결정 (duration 만료 에이전트만) ───────────────
        for i in range(n):
            if ep['rem'][i] <= 0 and ep['active'][i]:
                state = get_data(game, i, ep['std_cps'][i], ep['dis_gps'][i])
                greedy = has_model[i]
                _, _, a, d, dur = agents[i].predict(state, segment_idx=0, greedy=greedy)
                ep['ctrl'][i] = agents[i].get_real_action(a)
                ep['rem'][i]  = dur

        # ── 환경 스텝 (headless → 내부 draw/clock 없음) ─────────
        results = game.step(ep['ctrl'])
        game.camera_x = game.camera_y = 0   # 전체 맵 고정 뷰

        # ── 결과 처리 ────────────────────────────────────────────
        for i, res in enumerate(results):
            if not ep['active'][i]:
                continue

            ep['rem'][i] -= 1
            is_to = (game.current_time or 0) / 1000.0 > max_time

            if res['cp_reached']:
                ep['cp_cnt'][i] += 1
                nav_cps = game.car_checkpoints[i]
                new_tgt = (nav_cps[ep['cp_cnt'][i]]
                           if ep['cp_cnt'][i] < len(nav_cps)
                           else game.car_goals[i])
                if new_tgt:
                    ep['dis_gps'][i] = math.dist(ep['std_cps'][i], new_tgt)
                    ep['std_cps'][i] = list(new_tgt)

            if res['goal_reached']:
                ep['active'][i] = False
                ep['status'][i] = 'goal'
                goal_cnts[i]   += 1
                print(f"  [Ep {episode}] Agent{i+1} GOAL  "
                      f"({(game.current_time or 0)/1000:.1f}s)  "
                      f"총 {goal_cnts[i]}회")
            elif res['collision']:
                ep['active'][i] = False
                ep['status'][i] = 'collision'
            elif is_to:
                ep['active'][i] = False
                ep['status'][i] = 'timeout'

        # ── 렌더링 ──────────────────────────────────────────────
        _draw_scene(win, game, font_s)
        _draw_hud(win, font_s, font_m, font_l,
                  game, agents, model_paths, ep['status'],
                  goal_cnts, episode, max_time, paused=False)
        pygame.display.flip()
        clock.tick(60)

        # ── 에피소드 종료 판정 ──────────────────────────────────
        if not any(ep['active']):
            # 결과를 1.5초간 보여준 후 자동 리셋
            end_surf = pygame.Surface((WIN_W, 60), pygame.SRCALPHA)
            end_surf.fill((0, 0, 0, 160))
            parts = [f"A{i+1}:{ep['status'][i].upper()}" for i in range(n)]
            end_lbl = font_l.render("  |  ".join(parts) + "   → 리셋 중...",
                                    True, (255, 255, 255))
            end_surf.blit(end_lbl, end_lbl.get_rect(
                center=(WIN_W // 2, 30)))
            win.blit(end_surf, (0, WIN_H // 2 - 30))
            pygame.display.flip()
            pygame.time.wait(1500)

            ep = _make_episode_state(game, agents)
            episode += 1
            print(f"  [Ep {episode}] 시작")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="학습된 에이전트 모델 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent1",      type=str, default=None, help="Agent 1 모델 경로")
    parser.add_argument("--agent2",      type=str, default=None, help="Agent 2 모델 경로")
    parser.add_argument("--agent3",      type=str, default=None, help="Agent 3 모델 경로")
    parser.add_argument("--agent4",      type=str, default=None, help="Agent 4 모델 경로")
    parser.add_argument("--layer_num",   type=int, default=4,    help="네트워크 레이어 수 (기본 4)")
    parser.add_argument("--max_size",    type=int, default=512,  help="레이어 최대 크기 (기본 512)")
    parser.add_argument("--action_size", type=int, default=8,    help="액션 수 (기본 8)")
    parser.add_argument("--dur_size",    type=int, default=6,    help="Duration 수 (기본 6)")
    parser.add_argument("--max_time",    type=int, default=40,   help="에피소드 최대 시간 초 (기본 40)")
    parser.add_argument("--track",       type=str, default=None, help="트랙 JSON 경로")
    args = parser.parse_args()

    model_paths = [args.agent1, args.agent2, args.agent3, args.agent4]

    vehicle_configs = [
        _sim_asset("vehicles", "vehicle_1.json"),
        _sim_asset("vehicles", "vehicle_2.json"),
        _sim_asset("vehicles", "vehicle_3.json"),
        _sim_asset("vehicles", "vehicle_4.json"),
    ]

    print("=" * 55)
    print("  Agent Visualizer")
    print("=" * 55)
    for i, (mp, vc) in enumerate(zip(model_paths, vehicle_configs)):
        print(f"  Agent{i+1} | vehicle: {os.path.basename(vc)}")
        print(f"          | model  : {mp or '(없음 — 랜덤)'}")
    print(f"  Layer: {args.layer_num}  MaxSize: {args.max_size}  MaxTime: {args.max_time}s")
    print("=" * 55)

    run_visualization(
        vehicle_config_paths=vehicle_configs,
        model_paths=model_paths,
        action_size=args.action_size,
        duration_size=args.dur_size,
        layer_num=args.layer_num,
        max_size=args.max_size,
        max_time=args.max_time,
        track_file=args.track,
    )

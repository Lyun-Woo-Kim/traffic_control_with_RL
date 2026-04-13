"""
track_data.json 시각화 도구
사용법:
    python visualize_track.py                        # track_data.json (기본)
    python visualize_track.py my_track.json          # 지정 파일
    python visualize_track.py track.json --save      # PNG로 저장

조작:
    마우스 드래그  : 화면 이동
    마우스 휠      : 줌 인/아웃
    D              : 방향 벡터 표시 토글
    R              : 초기 뷰로 복귀
    ESC / Q        : 종료
"""

import pygame
import json
import math
import sys
import argparse
import os


# ── 색상 ────────────────────────────────────────────────────────
BG          = (180, 200, 180)
ROAD        = (60,  60,  60)
ROAD_EDGE   = (30,  30,  30)
CENTER_DASH = (255, 200,   0)
CP_COLOR    = (255, 210,   0)
CP_TEXT     = (0,   0,    0)
START_COLOR = (0,   200,  0)
END_COLOR   = (220,  50, 50)
TL_BOX      = (10,  10,  10)
TL_RED_ON   = (255,  30, 30)
TL_YEL_ON   = (255, 220,  0)
TL_GRN_ON   = (0,   220,  0)
TL_OFF_R    = (80,    0,  0)
TL_OFF_Y    = (80,   70,  0)
TL_OFF_G    = (0,    70,  0)
DIR_COLOR   = (100, 180, 255)
ARROW_COLOR = (255, 100,   0)

AGENT_COLORS = [
    (0,100,255),(255,80,0),(0,180,0),(180,0,180),(200,170,0),
    (0,180,180),(180,80,0),(80,0,180),(220,0,100),(0,200,80),
]


def _bezier(p0, p1, p2, p3, steps=50):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def _road_polygon(pts, width):
    half = width / 2
    left, right = [], []
    for i, p in enumerate(pts):
        if i == 0:
            dx, dy = pts[1][0]-p[0], pts[1][1]-p[1]
        elif i == len(pts)-1:
            dx, dy = p[0]-pts[i-1][0], p[1]-pts[i-1][1]
        else:
            dx, dy = pts[i+1][0]-pts[i-1][0], pts[i+1][1]-pts[i-1][1]
        ln = math.hypot(dx, dy)
        if ln == 0:
            continue
        ux, uy = dx/ln, dy/ln
        px, py = -uy, ux
        left.append( (p[0]+px*half, p[1]+py*half))
        right.append((p[0]-px*half, p[1]-py*half))
    return left, right


class Viewer:
    def __init__(self, data, win_w=1200, win_h=800):
        pygame.init()
        self.screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Track Visualizer")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont('arial', 14)
        self.font_m = pygame.font.SysFont('arial', 18, bold=True)
        self.font_l = pygame.font.SysFont('arial', 22, bold=True)

        self.W, self.H = win_w, win_h
        self.data = data

        # 뷰 상태
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom     = 1.0
        self.dragging = False
        self.drag_ox  = 0
        self.drag_oy  = 0

        self.show_dirs = True   # 방향 벡터 표시 여부

        self._fit_to_screen()

    # ── 좌표 변환 ────────────────────────────────────────────────
    def _w2s(self, wx, wy):
        """월드 → 스크린"""
        sx = (wx + self.offset_x) * self.zoom
        sy = (wy + self.offset_y) * self.zoom
        return int(sx), int(sy)

    def _fit_to_screen(self):
        """트랙 전체가 화면에 맞도록 초기 뷰 설정"""
        all_pts = []
        for seg in self.data.get('segments', []):
            all_pts.extend(seg['points'])
        for cp in self.data.get('checkpoints', []):
            all_pts.append(cp)
        for sp in self.data.get('start_positions', []):
            all_pts.append(sp)

        if not all_pts:
            return

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        pad = 80
        w_range = max_x - min_x or 1
        h_range = max_y - min_y or 1
        zoom_x  = (self.W - pad*2) / w_range
        zoom_y  = (self.H - pad*2) / h_range
        self.zoom     = min(zoom_x, zoom_y)
        self.offset_x = -min_x + pad / self.zoom
        self.offset_y = -min_y + pad / self.zoom

    # ── 렌더링 ───────────────────────────────────────────────────
    def _draw_road(self, surf):
        for seg in self.data.get('segments', []):
            pts = seg['points']
            w   = seg['width']
            if seg['type'] == 'curve' and len(pts) == 4:
                draw_pts = _bezier(*pts)
            else:
                draw_pts = pts

            left, right = _road_polygon(draw_pts, w)
            if not left:
                continue

            # 도로 채우기
            for i in range(len(left)-1):
                poly = [
                    self._w2s(*left[i]),
                    self._w2s(*left[i+1]),
                    self._w2s(*right[i+1]),
                    self._w2s(*right[i]),
                ]
                pygame.draw.polygon(surf, ROAD, poly)

            # 외곽선
            pygame.draw.lines(surf, ROAD_EDGE,
                              False, [self._w2s(*p) for p in left],  max(1, int(2*self.zoom)))
            pygame.draw.lines(surf, ROAD_EDGE,
                              False, [self._w2s(*p) for p in right], max(1, int(2*self.zoom)))

            # 중앙 점선
            dash = max(8, int(12 * self.zoom))
            s_pts = [self._w2s(*p) for p in draw_pts]
            for i in range(len(s_pts)-1):
                if i % 2 == 0:
                    pygame.draw.line(surf, CENTER_DASH, s_pts[i], s_pts[i+1],
                                     max(1, int(1.5*self.zoom)))

            # 방향 화살표 (중앙선 위)
            for i in range(0, len(s_pts)-1, max(1, len(s_pts)//5)):
                p1, p2 = s_pts[i], s_pts[i+1]
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                if math.hypot(dx, dy) < 5:
                    continue
                angle = math.atan2(dy, dx)
                mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
                sz = max(6, int(10*self.zoom))
                tip   = (mx+int(math.cos(angle)*sz),       my+int(math.sin(angle)*sz))
                aleft = (mx+int(math.cos(angle+2.5)*sz),   my+int(math.sin(angle+2.5)*sz))
                aright= (mx+int(math.cos(angle-2.5)*sz),   my+int(math.sin(angle-2.5)*sz))
                pygame.draw.polygon(surf, ARROW_COLOR, [tip, aleft, aright])

    def _draw_direction_grid(self, surf):
        """direction_grid 샘플 표시 (격자 셀마다 작은 화살표)"""
        grid = self.data.get('direction_grid', {})
        cell = 10
        skip = max(1, int(3 / self.zoom))   # 줌이 작으면 듬성듬성

        count = 0
        for key, val in grid.items():
            count += 1
            if count % (skip * skip) != 0:
                continue
            dx, dy = val['dir']
            if dx == 0 and dy == 0:
                continue
            parts = key.split('_')
            wx = int(parts[0]) * cell + cell//2
            wy = int(parts[1]) * cell + cell//2
            sx, sy = self._w2s(wx, wy)
            if not (0 <= sx <= self.W and 0 <= sy <= self.H):
                continue
            ln = max(4, int(6 * self.zoom))
            ex, ey = int(sx + dx*ln), int(sy + dy*ln)
            pygame.draw.line(surf, DIR_COLOR, (sx, sy), (ex, ey), 1)
            pygame.draw.circle(surf, DIR_COLOR, (ex, ey), max(1, ln//4))

    def _draw_checkpoints(self, surf):
        r = max(6, int(12 * self.zoom))
        for idx, cp in enumerate(self.data.get('checkpoints', [])):
            sx, sy = self._w2s(*cp)
            pygame.draw.circle(surf, CP_COLOR, (sx, sy), r)
            pygame.draw.circle(surf, (0, 0, 0), (sx, sy), r, max(1, int(1.5*self.zoom)))
            lbl = self.font_s.render(str(idx), True, CP_TEXT)
            surf.blit(lbl, lbl.get_rect(center=(sx, sy)))

    def _draw_start_positions(self, surf):
        r = max(10, int(18 * self.zoom))
        for idx, sp in enumerate(self.data.get('start_positions', [])):
            sx, sy = self._w2s(*sp)
            pygame.draw.circle(surf, START_COLOR, (sx, sy), r)
            pygame.draw.circle(surf, (0, 0, 0), (sx, sy), r, max(1, int(2*self.zoom)))
            lbl = self.font_m.render(f"S{idx}", True, (0, 0, 0))
            surf.blit(lbl, lbl.get_rect(center=(sx, sy)))

            # 시작 각도 화살표
            angles = self.data.get('start_angles_deg', [])
            if idx < len(angles):
                angle = math.radians(angles[idx])
                ln = max(20, int(35 * self.zoom))
                ex = int(sx + math.cos(angle) * ln)
                ey = int(sy + math.sin(angle) * ln)
                pygame.draw.line(surf, ARROW_COLOR, (sx, sy), (ex, ey), max(2, int(3*self.zoom)))
                a1 = (int(ex + math.cos(angle+2.5)*8), int(ey + math.sin(angle+2.5)*8))
                a2 = (int(ex + math.cos(angle-2.5)*8), int(ey + math.sin(angle-2.5)*8))
                pygame.draw.polygon(surf, ARROW_COLOR, [(ex, ey), a1, a2])

    def _draw_end(self, surf):
        ep = self.data.get('end_pos')
        if ep is None:
            return
        sx, sy = self._w2s(*ep)
        r = max(10, int(18 * self.zoom))
        pygame.draw.circle(surf, END_COLOR, (sx, sy), r)
        pygame.draw.circle(surf, (0, 0, 0), (sx, sy), r, max(1, int(2*self.zoom)))
        lbl = self.font_m.render("E", True, (255, 255, 255))
        surf.blit(lbl, lbl.get_rect(center=(sx, sy)))

    def _draw_traffic_lights(self, surf):
        for tl in self.data.get('traffic_lights', []):
            tx, ty = tl['pos']
            sx, sy = self._w2s(tx, ty)
            bw, bh = max(14, int(20*self.zoom)), max(38, int(55*self.zoom))
            pygame.draw.rect(surf, TL_BOX, (sx-bw//2, sy-bh//2, bw, bh), border_radius=4)
            r = max(4, int(7*self.zoom))
            pygame.draw.circle(surf, TL_RED_ON, (sx, sy - bh//3), r)
            pygame.draw.circle(surf, TL_OFF_Y,  (sx, sy),          r)
            pygame.draw.circle(surf, TL_OFF_G,  (sx, sy + bh//3), r)

            # 방향 화살표
            if 'dir' in tl:
                dx, dy = tl['dir']
                ln = max(20, int(30*self.zoom))
                ex, ey = int(sx + dx*ln), int(sy + dy*ln)
                pygame.draw.line(surf, ARROW_COLOR, (sx, sy), (ex, ey), max(2, int(2*self.zoom)))
                angle = math.atan2(dy, dx)
                a1 = (int(ex + math.cos(angle+2.5)*7), int(ey + math.sin(angle+2.5)*7))
                a2 = (int(ex + math.cos(angle-2.5)*7), int(ey + math.sin(angle-2.5)*7))
                pygame.draw.polygon(surf, ARROW_COLOR, [(ex, ey), a1, a2])

    def _draw_hud(self, surf):
        lines = [
            f"Zoom: {self.zoom:.2f}x",
            f"Segments: {len(self.data.get('segments', []))}",
            f"Checkpoints: {len(self.data.get('checkpoints', []))}",
            f"Start positions: {len(self.data.get('start_positions', []))}",
            f"Traffic lights: {len(self.data.get('traffic_lights', []))}",
            f"Dir grid cells: {len(self.data.get('direction_grid', {}))}",
            "",
            "[D] Direction grid: " + ("ON" if self.show_dirs else "OFF"),
            "[R] Fit to screen",
            "[ESC/Q] Quit",
        ]
        for i, line in enumerate(lines):
            color = (50, 50, 50) if line else (0, 0, 0)
            lbl = self.font_s.render(line, True, color)
            surf.blit(lbl, (10, 10 + i * 18))

    # ── 메인 루프 ────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_d:
                        self.show_dirs = not self.show_dirs
                    elif event.key == pygame.K_r:
                        self._fit_to_screen()

                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    # 마우스 위치 기준 줌
                    wx = mx / self.zoom - self.offset_x
                    wy = my / self.zoom - self.offset_y
                    self.zoom = max(0.1, min(10.0, self.zoom * (1.1 if event.y > 0 else 0.9)))
                    self.offset_x = mx / self.zoom - wx
                    self.offset_y = my / self.zoom - wy

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.dragging = True
                    self.drag_ox, self.drag_oy = pygame.mouse.get_pos()

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging = False

                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    mx, my = pygame.mouse.get_pos()
                    self.offset_x += (mx - self.drag_ox) / self.zoom
                    self.offset_y += (my - self.drag_oy) / self.zoom
                    self.drag_ox, self.drag_oy = mx, my

            # 렌더
            self.screen.fill(BG)
            self._draw_road(self.screen)
            if self.show_dirs:
                self._draw_direction_grid(self.screen)
            self._draw_checkpoints(self.screen)
            self._draw_start_positions(self.screen)
            self._draw_end(self.screen)
            self._draw_traffic_lights(self.screen)
            self._draw_hud(self.screen)
            pygame.display.flip()

        pygame.quit()

    def save_png(self, path):
        """현재 뷰를 PNG로 저장"""
        self.screen.fill(BG)
        self._draw_road(self.screen)
        if self.show_dirs:
            self._draw_direction_grid(self.screen)
        self._draw_checkpoints(self.screen)
        self._draw_start_positions(self.screen)
        self._draw_end(self.screen)
        self._draw_traffic_lights(self.screen)
        pygame.image.save(self.screen, path)
        print(f"Saved: {path}")


# ── 진입점 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track JSON 시각화 도구")
    parser.add_argument("json", nargs="?", default="/home/krw/traffic_control_with_RL/traffic_control_with_RL/traffic_simulation/track_data.json",
                        help="트랙 JSON 파일 경로 (기본: track_data.json)")
    parser.add_argument("--save", action="store_true",
                        help="PNG로 저장 후 종료")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"파일을 찾을 수 없습니다: {args.json}")
        sys.exit(1)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    viewer = Viewer(data)

    if args.save:
        out = os.path.splitext(args.json)[0] + "_preview.png"
        viewer.save_png(out)
    else:
        viewer.run()

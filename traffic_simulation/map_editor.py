import pygame
import numpy as np
import json
import math
import random

class TrackEditor:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Track Editor - Direction Arrow Mode")
        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (60, 60, 60)
        self.PREVIEW_GRAY = (150, 150, 150)
        self.EDGE_COLOR = (40, 40, 40)
        self.YELLOW = (255, 200, 0)
        self.ORANGE = (255, 100, 0)
        self.RED = (255, 50, 50)
        self.GREEN = (0, 200, 0)
        self.BLACK = (0, 0, 0)
        self.BLUE = (50, 100, 255)

        # Traffic light colors
        self.TL_RED    = (255, 30,  30)
        self.TL_YELLOW = (255, 220,  0)
        self.TL_GREEN  = (0,  220,   0)
        self.TL_OFF_R  = (80,   0,   0)
        self.TL_OFF_Y  = (80,  70,   0)
        self.TL_OFF_G  = (0,   70,   0)

        self.track_width = 100
        self.min_width   = 40
        self.max_width   = 200

        self.segments        = []
        self.checkpoints     = []
        self.traffic_lights  = []
        self.start_positions = []   # 다중 시작 위치 (인덱스 = start_point 번호)
        self.start_angles_deg = []  # start_positions와 같은 인덱스의 시작 각도(도)
        self.end_pos         = None

        # 순차 신호등 상태 (0번 → 1번 → 2번 → ... → 0번 순환)
        self.TL_GREEN_MS  = 12000  # 초록 지속 시간 (ms)
        self.TL_YELLOW_MS = 2000   # 노랑 지속 시간 (ms)
        self.tl_seq_idx   = 0      # 현재 활성 신호등 인덱스
        self.tl_seq_timer = 0      # 현재 단계 경과 시간 (ms)
        self.tl_seq_phase = 'green'  # 'green' or 'yellow'

        self.current_tool   = "line"
        self.drag_start     = None
        self.temp_points    = []
        self.pending_start_pos = None

        self.ui_height      = 100
        self.font           = pygame.font.SysFont('arial', 16)
        self.buttons        = self._create_buttons()
        self.slider_rect    = pygame.Rect(self.width - 220, 40, 200, 20)
        self.dragging_slider = False

    # ------------------------------------------------------------------ UI
    def _create_buttons(self):
        btns = []
        labels = [
            ("Line Road",  "line"),  ("Curve(4pt)", "curve"),
            ("Checkpoint", "checkpoint"), ("T-Light", "light"),
            ("Start Point","start"), ("End Point",  "end"),
            ("Undo",       "undo"),  ("Clear",      "clear"), ("Save", "save"),
        ]
        x, y = 10, 10
        for txt, act in labels:
            btns.append({'rect': pygame.Rect(x, y, 90, 35), 'text': txt, 'action': act})
            x += 100
            if x > 800: x = 10; y += 45
        return btns

    # ------------------------------------------------------------------ 기하
    def _get_bezier_points(self, p0, p1, p2, p3, steps=50):
        pts = []
        for t in np.linspace(0, 1, steps):
            x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
            y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
            pts.append((x, y))
        return pts

    def _get_road_polygon(self, points, width):
        if len(points) < 2:
            return [], []
        half_w = width / 2
        left_edge, right_edge = [], []
        for i in range(len(points)):
            p = points[i]
            if i == 0:
                dx, dy = points[1][0]-p[0], points[1][1]-p[1]
            elif i == len(points)-1:
                dx, dy = p[0]-points[i-1][0], p[1]-points[i-1][1]
            else:
                dx, dy = points[i+1][0]-points[i-1][0], points[i+1][1]-points[i-1][1]
            ln = math.hypot(dx, dy)
            if ln == 0: continue
            ux, uy = dx/ln, dy/ln
            px, py = -uy, ux
            left_edge.append( (p[0]+px*half_w, p[1]+py*half_w))
            right_edge.append((p[0]-px*half_w, p[1]-py*half_w))
        return left_edge, right_edge

    def _sample_polyline(self, points, step=10):
        """중심선 폴리라인을 step px 간격으로 샘플링 → [(x, y, ux, uy), ...]"""
        samples = []
        carry   = 0.0
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            dx, dy  = p2[0]-p1[0], p2[1]-p1[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6: continue
            ux, uy = dx/seg_len, dy/seg_len
            t = step - carry
            while t <= seg_len + 1e-6:
                samples.append((p1[0]+ux*t, p1[1]+uy*t, ux, uy))
                t += step
            carry = seg_len - (t - step)
            if carry < 0: carry = 0
        return samples

    # ------------------------------------------------------------------ 통합 도로 렌더링
    def _render_all_roads(self, surface):
        """
        모든 세그먼트를 통합 렌더링.
        겹치는 부분은 단일 도로색으로 표시, 외곽 경계선만 한 번 그림.
        """
        if not self.segments:
            return

        w, h = surface.get_size()

        # ── Pass 1: 모든 fill 을 별도 surface 에 ──────────────────────
        road_surf = pygame.Surface((w, h))
        road_surf.fill((180, 200, 180))   # 배경색과 동일

        for seg in self.segments:
            pts = (self._get_bezier_points(*seg['points'])
                   if seg['type'] == 'curve' else seg['points'])
            left, right = self._get_road_polygon(pts, seg['width'])
            if not left:
                continue
            for i in range(len(left) - 1):
                poly = [left[i], left[i+1], right[i+1], right[i]]
                pygame.draw.polygon(road_surf, self.GRAY, poly)

        surface.blit(road_surf, (0, 0))

        # ── Pass 2: numpy 경계 감지 → 외곽선 한 번만 그리기 ──────────
        arr     = pygame.surfarray.array3d(road_surf)
        is_road = arr[:, :, 0] < 100   # GRAY R=60 < 100, BG R=180

        # 경계 픽셀: 도로이면서 인접 픽셀 중 비도로가 있는 것
        bnd          = np.zeros_like(is_road)
        bnd[:-1, :] |= is_road[:-1, :] & ~is_road[1:, :]
        bnd[1:, :]  |= is_road[1:, :]  & ~is_road[:-1, :]
        bnd[:, :-1] |= is_road[:, :-1] & ~is_road[:, 1:]
        bnd[:, 1:]  |= is_road[:, 1:]  & ~is_road[:, :-1]

        # 2픽셀 팽창 → 경계선 두께 ~4px
        dilated = bnd.copy()
        for _ in range(2):
            tmp        = dilated.copy()
            tmp[1:,  :] |= dilated[:-1, :]
            tmp[:-1, :] |= dilated[1:, :]
            tmp[:,  1:] |= dilated[:, :-1]
            tmp[:, :-1] |= dilated[:, 1:]
            dilated = tmp

        edge_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        ea = pygame.surfarray.pixels3d(edge_surf)
        aa = pygame.surfarray.pixels_alpha(edge_surf)
        ea[dilated] = (40, 40, 40)
        aa[dilated] = 255
        del ea, aa
        surface.blit(edge_surf, (0, 0))

        # ── Pass 3: 중앙 대시선 + 방향 화살표 ───────────────────────
        for seg in self.segments:
            pts = (self._get_bezier_points(*seg['points'])
                   if seg['type'] == 'curve' else seg['points'])
            self._draw_dashed_line(surface, self.YELLOW, pts, 2)
            if seg['type'] == 'line':
                self._draw_arrow(surface, pts[0], pts[-1])
            else:
                for i in range(len(pts) - 1):
                    if i % 10 == 5:
                        self._draw_arrow(surface, pts[i], pts[i+1])

    # ------------------------------------------------------------------ 개별 세그먼트 렌더링 (미리보기용)
    def _draw_arrow(self, surface, p1, p2):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if math.hypot(dx, dy) < 20: return
        angle = math.atan2(dy, dx)
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        sz = 12
        tip   = (mx+math.cos(angle)*sz,       my+math.sin(angle)*sz)
        left  = (mx+math.cos(angle+2.5)*sz,   my+math.sin(angle+2.5)*sz)
        right = (mx+math.cos(angle-2.5)*sz,   my+math.sin(angle-2.5)*sz)
        pygame.draw.polygon(surface, self.ORANGE, [tip, left, right])

    def _draw_dashed_line(self, surface, color, points, width=2, dash=15):
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            dist = math.hypot(dx, dy)
            if dist == 0: continue
            ux, uy = dx/dist, dy/dist
            s = 0
            while s < dist:
                sp = (p1[0]+ux*s,          p1[1]+uy*s)
                ep = (p1[0]+ux*min(s+dash,dist), p1[1]+uy*min(s+dash,dist))
                if (s//dash) % 2 == 0:
                    pygame.draw.line(surface, color, sp, ep, width)
                s += dash

    def _draw_segment(self, surface, seg, preview=False):
        pts = seg['points']
        w   = seg['width']
        draw_pts = self._get_bezier_points(*pts) if seg['type'] == 'curve' else pts
        left_edge, right_edge = self._get_road_polygon(draw_pts, w)
        if not left_edge: return

        col = self.PREVIEW_GRAY if preview else self.GRAY
        for i in range(len(left_edge)-1):
            poly = [left_edge[i], left_edge[i+1], right_edge[i+1], right_edge[i]]
            pygame.draw.polygon(surface, col, poly)

        if preview:
            pygame.draw.lines(surface, self.YELLOW, False, draw_pts, 2)
            if seg['type'] == 'line':
                self._draw_arrow(surface, draw_pts[0], draw_pts[1])
            else:
                for i in range(len(draw_pts)-1):
                    if i % 10 == 5:
                        self._draw_arrow(surface, draw_pts[i], draw_pts[i+1])
        else:
            pygame.draw.lines(surface, self.EDGE_COLOR, False, left_edge,  4)
            pygame.draw.lines(surface, self.EDGE_COLOR, False, right_edge, 4)
            self._draw_dashed_line(surface, self.YELLOW, draw_pts, 2)

    def _draw_start_finish_line(self, surface, pos, is_start=True):
        color = self.GREEN if is_start else self.RED
        label = "S" if is_start else "E"
        pygame.draw.circle(surface, color,      pos, 20)
        pygame.draw.circle(surface, self.BLACK, pos, 20, 3)
        surface.blit(self.font.render(label, True, self.BLACK),
                     self.font.render(label, True, self.BLACK).get_rect(center=pos))

    def _draw_traffic_light(self, surface, tl):
        x, y = int(tl['pos'][0]), int(tl['pos'][1])
        pygame.draw.rect(surface, self.BLACK, (x-14, y-38, 28, 76), border_radius=5)
        pygame.draw.circle(surface,
                           self.TL_RED    if tl['state'] == 'red'    else self.TL_OFF_R,
                           (x, y-25), 9)
        pygame.draw.circle(surface,
                           self.TL_YELLOW if tl['state'] == 'yellow' else self.TL_OFF_Y,
                           (x, y),    9)
        pygame.draw.circle(surface,
                           self.TL_GREEN  if tl['state'] == 'green'  else self.TL_OFF_G,
                           (x, y+25), 9)
        # 방향 화살표
        if 'dir' in tl:
            dx, dy = tl['dir']
            ex = int(x + dx * 35)
            ey = int(y + dy * 35)
            pygame.draw.line(surface, self.ORANGE, (x, y), (ex, ey), 3)
            angle = math.atan2(dy, dx)
            a1 = (int(ex + math.cos(angle+2.5)*8), int(ey + math.sin(angle+2.5)*8))
            a2 = (int(ex + math.cos(angle-2.5)*8), int(ey + math.sin(angle-2.5)*8))
            pygame.draw.polygon(surface, self.ORANGE, [(ex, ey), a1, a2])

    def _draw_traffic_light_ghost(self, surface, pos, dir_vec):
        """반투명 신호등 미리보기 — SRCALPHA surface에 그린다"""
        x, y = int(pos[0]), int(pos[1])
        pygame.draw.rect(surface,   (0,   0,   0, 150), (x-14, y-38, 28, 76), border_radius=5)
        pygame.draw.circle(surface, (80,  0,   0, 150), (x, y-25), 9)
        pygame.draw.circle(surface, (80,  70,  0, 150), (x, y),    9)
        pygame.draw.circle(surface, (0,   70,  0, 150), (x, y+25), 9)
        if dir_vec:
            dx, dy = dir_vec
            ex = int(x + dx * 35)
            ey = int(y + dy * 35)
            pygame.draw.line(surface,    (255, 150, 0, 220), (x, y), (ex, ey), 3)
            angle = math.atan2(dy, dx)
            a1 = (int(ex + math.cos(angle+2.5)*8), int(ey + math.sin(angle+2.5)*8))
            a2 = (int(ex + math.cos(angle-2.5)*8), int(ey + math.sin(angle-2.5)*8))
            pygame.draw.polygon(surface, (255, 150, 0, 220), [(ex, ey), a1, a2])

    def _draw_ghost_preview(self, surface, mouse_pos):
        """현재 툴의 반투명 hover 미리보기"""
        ghost = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        mx, my = mouse_pos

        if self.current_tool == 'line':
            if not self.drag_start:
                # 드래그 전: 도로 폭 원으로 표시
                pygame.draw.circle(ghost, (150, 150, 150,  70), mouse_pos, self.track_width // 2)
                pygame.draw.circle(ghost, (255, 255, 255, 100), mouse_pos, self.track_width // 2, 2)

        elif self.current_tool == 'curve':
            n = len(self.temp_points)
            if n == 3:
                # 4번째 점 = 마우스 → bezier 미리보기
                bpts = self._get_bezier_points(*self.temp_points, mouse_pos)
                left, right = self._get_road_polygon(bpts, self.track_width)
                if left:
                    for i in range(len(left) - 1):
                        pygame.draw.polygon(ghost, (150, 150, 150, 70),
                                            [left[i], left[i+1], right[i+1], right[i]])
                    pygame.draw.lines(ghost, (255, 200, 0, 150), False, bpts, 2)
            elif n == 0:
                pygame.draw.circle(ghost, (150, 150, 150,  70), mouse_pos, self.track_width // 2)
                pygame.draw.circle(ghost, (255, 255, 255, 100), mouse_pos, self.track_width // 2, 2)

        elif self.current_tool == 'checkpoint':
            pygame.draw.circle(ghost, (255, 200, 0, 120), mouse_pos, 14)
            pygame.draw.circle(ghost, (255, 200, 0, 200), mouse_pos, 14, 2)
            lbl = self.font.render(str(len(self.checkpoints)), True, (0, 0, 0, 200))
            ghost.blit(lbl, lbl.get_rect(center=mouse_pos))

        elif self.current_tool == 'light':
            if self.drag_start:
                dx = mx - self.drag_start[0]
                dy = my - self.drag_start[1]
                mag = math.hypot(dx, dy)
                dir_vec = (dx/mag, dy/mag) if mag > 5 else (1.0, 0.0)
                self._draw_traffic_light_ghost(ghost, self.drag_start, dir_vec)
            else:
                self._draw_traffic_light_ghost(ghost, mouse_pos, None)

        elif self.current_tool == 'start':
            if self.pending_start_pos:
                sp = self.pending_start_pos
                pygame.draw.circle(ghost, (0, 200, 0, 120), sp, 20)
                pygame.draw.circle(ghost, (0, 200, 0, 200), sp, 20, 3)
                dx = mx - sp[0]
                dy = my - sp[1]
                mag = math.hypot(dx, dy)
                if mag > 5:
                    ex = int(sp[0] + dx / mag * 35)
                    ey = int(sp[1] + dy / mag * 35)
                    pygame.draw.line(ghost, (255, 150, 0, 220), sp, (ex, ey), 3)
                    angle = math.atan2(dy, dx)
                    a1 = (int(ex + math.cos(angle+2.5)*8), int(ey + math.sin(angle+2.5)*8))
                    a2 = (int(ex + math.cos(angle-2.5)*8), int(ey + math.sin(angle-2.5)*8))
                    pygame.draw.polygon(ghost, (255, 150, 0, 220), [(ex, ey), a1, a2])
                lbl = self.font.render(str(len(self.start_positions)), True, (0, 0, 0, 200))
                ghost.blit(lbl, lbl.get_rect(center=sp))
            else:
                pygame.draw.circle(ghost, (0, 200, 0, 120), mouse_pos, 20)
                pygame.draw.circle(ghost, (0, 200, 0, 200), mouse_pos, 20, 3)
                lbl = self.font.render(str(len(self.start_positions)), True, (0, 0, 0, 200))
                ghost.blit(lbl, lbl.get_rect(center=mouse_pos))

        elif self.current_tool == 'end':
            pygame.draw.circle(ghost, (255,  50, 50, 120), mouse_pos, 20)
            pygame.draw.circle(ghost, (255,  50, 50, 200), mouse_pos, 20, 3)

        surface.blit(ghost, (0, 0))

    # ------------------------------------------------------------------ 신호등 업데이트
    def _update_traffic_lights(self, dt):
        """순차 신호등: 0번 초록→노랑, 1번 초록→노랑, ... 순으로 동일 시간 작동"""
        n = len(self.traffic_lights)
        if n == 0:
            return

        self.tl_seq_timer += dt
        duration = self.TL_GREEN_MS if self.tl_seq_phase == 'green' else self.TL_YELLOW_MS

        if self.tl_seq_timer >= duration:
            self.tl_seq_timer -= duration
            if self.tl_seq_phase == 'green':
                self.tl_seq_phase = 'yellow'
            else:
                # 노랑 끝 → 다음 신호등으로
                self.tl_seq_idx   = (self.tl_seq_idx + 1) % n
                self.tl_seq_phase = 'green'

        # 모든 신호등 상태 일괄 갱신
        for i, tl in enumerate(self.traffic_lights):
            tl['state'] = self.tl_seq_phase if i == self.tl_seq_idx else 'red'

    # ------------------------------------------------------------------ 저장용 데이터 빌드
    def _build_direction_grid(self):
        """
        10x10 px 셀마다 방향 벡터 저장
        - 단일 세그먼트 셀:
          중앙선을 기준으로
          * 그린 방향의 오른쪽 차로 = 정방향
          * 왼쪽 차로 = 역방향
        - 교차(2개+ 세그먼트) 셀: {"is_intersection": 1, "dir": [0.0, 0.0]}
        """
        cell_map = {}   # key -> list of (seg_id, lane_side, dir_x, dir_y)
        cell_size = 10.0

        for seg_id, seg in enumerate(self.segments):
            center_pts = (self._get_bezier_points(*seg['points'])
                          if seg['type'] == 'curve' else seg['points'])
            half_w = seg['width'] / 2
            lateral_offsets = np.arange(-half_w + cell_size / 2, half_w, cell_size)
            if len(lateral_offsets) == 0:
                lateral_offsets = np.array([0.0])

            for (cx, cy, ux, uy) in self._sample_polyline(center_pts, step=10):
                perp_x, perp_y = -uy, ux
                for offset in lateral_offsets:
                    px = cx + perp_x * offset
                    py = cy + perp_y * offset
                    key = f"{int(px/10)}_{int(py/10)}"

                    # perp = (-uy, ux) 은 pygame 좌표(y↓)에서 진행 방향의 오른쪽 수직벡터.
                    # 따라서 offset > 0 이 운전자 기준 오른쪽 차로 (한국 우측통행 정방향),
                    #         offset < 0 이 왼쪽 차로 (역방향 / 맞은편 차선).
                    if offset > 0:
                        lane_side = 'right'
                        dir_x, dir_y = ux, uy
                    else:
                        lane_side = 'left'
                        dir_x, dir_y = -ux, -uy

                    cell_map.setdefault(key, []).append((seg_id, lane_side, dir_x, dir_y))

        grid = {}
        for key, entries in cell_map.items():
            seg_ids = {e[0] for e in entries}
            if len(seg_ids) >= 2:
                grid[key] = {"is_intersection": 1, "dir": [0.0, 0.0]}
            else:
                lane_sides = {e[1] for e in entries}
                if len(lane_sides) >= 2:
                    # 중앙선에 걸친 셀은 방향 중립 처리
                    grid[key] = {"is_intersection": 0, "dir": [0.0, 0.0]}
                    continue

                avg_x = sum(e[2] for e in entries) / len(entries)
                avg_y = sum(e[3] for e in entries) / len(entries)
                mag   = math.hypot(avg_x, avg_y)
                if mag > 0:
                    avg_x /= mag; avg_y /= mag
                grid[key] = {"is_intersection": 0,
                             "dir": [round(avg_x, 4), round(avg_y, 4)]}
        return grid

    def _build_lane_data(self):
        """
        세그먼트별 좌측/중앙/우측 차선 포인트 저장
        """
        lane_data = []
        for seg_id, seg in enumerate(self.segments):
            center_pts = (self._get_bezier_points(*seg['points'])
                          if seg['type'] == 'curve' else seg['points'])
            left_edge, right_edge = self._get_road_polygon(center_pts, seg['width'])
            lane_data.append({
                'seg_id':      seg_id,
                'left_lane':   [[round(p[0],2), round(p[1],2)] for p in left_edge],
                'center_lane': [[round(p[0],2), round(p[1],2)] for p in center_pts],
                'right_lane':  [[round(p[0],2), round(p[1],2)] for p in right_edge],
            })
        return lane_data

    # ------------------------------------------------------------------ 저장
    def save(self):
        direction_grid = self._build_direction_grid()
        lane_data      = self._build_lane_data()

        tl_serial = [{
            'pos': [tl['pos'][0], tl['pos'][1]],
            'dir':  tl.get('dir', [1.0, 0.0]),
        } for tl in self.traffic_lights]

        data = {
            'segments':       [{'type': s['type'],
                                 'points': [[p[0],p[1]] for p in s['points']],
                                 'width':  s['width']} for s in self.segments],
            'direction_grid':  direction_grid,
            'lane_data':       lane_data,
            'traffic_lights':  tl_serial,
            'tl_green_ms':     self.TL_GREEN_MS,
            'tl_yellow_ms':    self.TL_YELLOW_MS,
            'checkpoints':      [[p[0],p[1]] for p in self.checkpoints],
            'start_positions':  [[p[0],p[1]] for p in self.start_positions],
            'start_angles_deg': [round(a, 2) for a in self.start_angles_deg],
            'end_pos':           list(self.end_pos) if self.end_pos else None,
        }
        with open("track_data.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved — cells: {len(direction_grid)}, segments: {len(self.segments)}")

    # ------------------------------------------------------------------ 메인 루프
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)            # ms
            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEWHEEL:
                    self.track_width = max(self.min_width,
                                           min(self.max_width, self.track_width + event.y*5))

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.slider_rect.collidepoint((mx, my)):
                        self.dragging_slider = True
                    elif my < self.ui_height:
                        for btn in self.buttons:
                            if btn['rect'].collidepoint((mx, my)):
                                act = btn['action']
                                if   act == 'save':  self.save()
                                elif act == 'clear':
                                    self.segments=[]; self.checkpoints=[]
                                    self.traffic_lights=[]
                                    self.start_positions=[]; self.start_angles_deg=[]; self.end_pos=None
                                    self.tl_seq_idx=0; self.tl_seq_timer=0
                                    self.tl_seq_phase='green'
                                elif act == 'undo':
                                    if   self.segments:         self.segments.pop()
                                    elif self.start_positions:
                                        self.start_positions.pop()
                                        if self.start_angles_deg:
                                            self.start_angles_deg.pop()
                                else:
                                    self.current_tool = act
                                    self.temp_points  = []
                                    self.drag_start   = None
                                    self.pending_start_pos = None
                    else:
                        t_pos = (mx, my - self.ui_height)
                        if   self.current_tool == 'line':
                            self.drag_start = t_pos
                        elif self.current_tool == 'curve':
                            self.temp_points.append(t_pos)
                            if len(self.temp_points) == 4:
                                self.segments.append({'type': 'curve',
                                                      'points': self.temp_points,
                                                      'width':  self.track_width})
                                self.temp_points = []
                        elif self.current_tool == 'checkpoint':
                            self.checkpoints.append(t_pos)
                        elif self.current_tool == 'light':
                            self.drag_start = t_pos   # 드래그로 방향 설정
                        elif self.current_tool == 'start':
                            if self.pending_start_pos is None:
                                self.pending_start_pos = t_pos
                            else:
                                dx = t_pos[0] - self.pending_start_pos[0]
                                dy = t_pos[1] - self.pending_start_pos[1]
                                angle_deg = math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 5 else 0.0
                                self.start_positions.append(self.pending_start_pos)
                                self.start_angles_deg.append(angle_deg)
                                self.pending_start_pos = None
                        elif self.current_tool == 'end':
                            self.end_pos = t_pos

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_slider = False
                    if self.current_tool == 'line' and self.drag_start and my > self.ui_height:
                        t_pos = (mx, my - self.ui_height)
                        if math.hypot(t_pos[0]-self.drag_start[0],
                                      t_pos[1]-self.drag_start[1]) > 10:
                            self.segments.append({'type':   'line',
                                                  'points': [self.drag_start, t_pos],
                                                  'width':  self.track_width})
                        self.drag_start = None
                    elif self.current_tool == 'light' and self.drag_start and my > self.ui_height:
                        t_pos = (mx, my - self.ui_height)
                        dx = t_pos[0] - self.drag_start[0]
                        dy = t_pos[1] - self.drag_start[1]
                        mag = math.hypot(dx, dy)
                        dir_vec = [round(dx/mag, 4), round(dy/mag, 4)] if mag > 5 else [1.0, 0.0]
                        self.traffic_lights.append({
                            'pos':   list(self.drag_start),
                            'dir':   dir_vec,
                            'state': 'red',
                        })
                        self.drag_start = None

            if self.dragging_slider:
                ratio = max(0, min(1, (mx-self.slider_rect.x)/self.slider_rect.width))
                self.track_width = int(self.min_width + ratio*(self.max_width-self.min_width))

            self._update_traffic_lights(dt)

            # ---- 렌더링 ----
            self.screen.fill((180, 200, 180))
            sub = self.screen.subsurface(
                pygame.Rect(0, self.ui_height, self.width, self.height-self.ui_height))

            self._render_all_roads(sub)

            real_mouse = (mx, my - self.ui_height)

            # Line 드래그 미리보기
            if self.current_tool == 'line' and self.drag_start:
                self._draw_segment(sub,
                    {'type':'line','points':[self.drag_start, real_mouse],'width':self.track_width},
                    preview=True)

            # Curve 제어점 표시 (3점 미만일 때만 연결선 표시 — 3점이면 ghost가 bezier 처리)
            if self.current_tool == 'curve' and self.temp_points:
                for p in self.temp_points:
                    pygame.draw.circle(sub, self.RED, p, 5)
                if len(self.temp_points) < 3:
                    pygame.draw.line(sub, self.RED, self.temp_points[-1], real_mouse, 1)

            # Light 드래그 중 방향선
            if self.current_tool == 'light' and self.drag_start:
                pygame.draw.line(sub, self.ORANGE, self.drag_start, real_mouse, 2)

            # Ghost 미리보기 (모든 툴 공통)
            if my > self.ui_height:
                self._draw_ghost_preview(sub, real_mouse)

            for idx, sp in enumerate(self.start_positions):
                pygame.draw.circle(sub, self.GREEN,  sp, 20)
                pygame.draw.circle(sub, self.BLACK,  sp, 20, 3)
                lbl = self.font.render(str(idx), True, self.BLACK)
                sub.blit(lbl, lbl.get_rect(center=sp))
                if idx < len(self.start_angles_deg):
                    angle = math.radians(self.start_angles_deg[idx])
                    ex = int(sp[0] + math.cos(angle) * 35)
                    ey = int(sp[1] + math.sin(angle) * 35)
                    pygame.draw.line(sub, self.ORANGE, sp, (ex, ey), 3)
                    a1 = (int(ex + math.cos(angle+2.5)*8), int(ey + math.sin(angle+2.5)*8))
                    a2 = (int(ex + math.cos(angle-2.5)*8), int(ey + math.sin(angle-2.5)*8))
                    pygame.draw.polygon(sub, self.ORANGE, [(ex, ey), a1, a2])
            if self.end_pos: self._draw_start_finish_line(sub, self.end_pos, False)
            for idx, cp in enumerate(self.checkpoints):
                pygame.draw.circle(sub, self.YELLOW, cp, 14)
                pygame.draw.circle(sub, self.BLACK,  cp, 14, 2)
                lbl = self.font.render(str(idx), True, self.BLACK)
                sub.blit(lbl, lbl.get_rect(center=cp))
            for tl in self.traffic_lights:
                self._draw_traffic_light(sub, tl)

            # ---- UI 패널 ----
            pygame.draw.rect(self.screen, (230,230,230), (0,0,self.width,self.ui_height))
            for btn in self.buttons:
                col = self.BLUE if btn['action'] == self.current_tool else self.WHITE
                if btn['action'] in ['save','undo','clear']: col = (200,255,200)
                pygame.draw.rect(self.screen, col, btn['rect'])
                pygame.draw.rect(self.screen, self.BLACK, btn['rect'], 2)
                self.screen.blit(
                    self.font.render(btn['text'], True,
                                     self.BLACK if col != self.BLUE else self.WHITE),
                    (btn['rect'].x+5, btn['rect'].y+9))

            pygame.draw.rect(self.screen, self.GRAY, self.slider_rect)
            ratio = (self.track_width-self.min_width)/(self.max_width-self.min_width)
            pygame.draw.circle(self.screen, self.BLUE,
                               (int(self.slider_rect.x+ratio*self.slider_rect.width),
                                self.slider_rect.centery), 10)
            self.screen.blit(
                self.font.render(f"Width: {self.track_width}px", True, self.BLACK),
                (self.slider_rect.x, self.slider_rect.y-25))

            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    TrackEditor().run()

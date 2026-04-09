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

        self.segments       = []
        self.checkpoints    = []
        self.traffic_lights = []
        self.start_pos      = None
        self.end_pos        = None

        self.current_tool   = "line"
        self.drag_start     = None
        self.temp_points    = []

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

    # ------------------------------------------------------------------ 렌더링
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

    # ------------------------------------------------------------------ 신호등 업데이트
    def _update_traffic_lights(self, dt):
        next_state = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
        for tl in self.traffic_lights:
            tl['timer'] += dt
            if tl['timer'] >= tl[f"{tl['state']}_duration"]:
                tl['timer'] = 0
                tl['state'] = next_state[tl['state']]
                if tl['state'] == 'green':          # 초록이 될 때마다 랜덤 지속시간 재설정
                    tl['green_duration'] = random.randint(5000, 15000)

    # ------------------------------------------------------------------ 저장용 데이터 빌드
    def _build_direction_grid(self):
        """
        10x10 px 셀마다 방향 벡터 저장
        - 단일 세그먼트 셀: {"is_intersection": 0, "dir": [dx, dy]}
        - 교차(2개+ 세그먼트) 셀: {"is_intersection": 1, "dir": [0.0, 0.0]}
        """
        cell_map = {}   # key -> list of (seg_id, ux, uy)

        for seg_id, seg in enumerate(self.segments):
            center_pts = (self._get_bezier_points(*seg['points'])
                          if seg['type'] == 'curve' else seg['points'])
            for (px, py, ux, uy) in self._sample_polyline(center_pts, step=10):
                key = f"{int(px/10)}_{int(py/10)}"
                cell_map.setdefault(key, []).append((seg_id, ux, uy))

        grid = {}
        for key, entries in cell_map.items():
            seg_ids = {e[0] for e in entries}
            if len(seg_ids) >= 2:
                grid[key] = {"is_intersection": 1, "dir": [0.0, 0.0]}
            else:
                avg_x = sum(e[1] for e in entries) / len(entries)
                avg_y = sum(e[2] for e in entries) / len(entries)
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
            'pos':              [tl['pos'][0], tl['pos'][1]],
            'state':             tl['state'],
            'green_duration':    tl['green_duration'],
            'yellow_duration':   tl['yellow_duration'],
            'red_duration':      tl['red_duration'],
        } for tl in self.traffic_lights]

        data = {
            'segments':       [{'type': s['type'],
                                 'points': [[p[0],p[1]] for p in s['points']],
                                 'width':  s['width']} for s in self.segments],
            'direction_grid':  direction_grid,
            'lane_data':       lane_data,
            'traffic_lights':  tl_serial,
            'checkpoints':    [[p[0],p[1]] for p in self.checkpoints],
            'start_pos':       list(self.start_pos) if self.start_pos else None,
            'end_pos':         list(self.end_pos)   if self.end_pos   else None,
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
                                    self.start_pos=None; self.end_pos=None
                                elif act == 'undo':
                                    if self.segments: self.segments.pop()
                                else:
                                    self.current_tool = act
                                    self.temp_points  = []
                                    self.drag_start   = None
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
                            self.traffic_lights.append({
                                'pos':             t_pos,
                                'state':          'red',
                                'timer':           0,
                                'green_duration':  random.randint(5000, 15000),
                                'yellow_duration': 3000,
                                'red_duration':    7000,
                            })
                        elif self.current_tool == 'start': self.start_pos = t_pos
                        elif self.current_tool == 'end':   self.end_pos   = t_pos

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

            if self.dragging_slider:
                ratio = max(0, min(1, (mx-self.slider_rect.x)/self.slider_rect.width))
                self.track_width = int(self.min_width + ratio*(self.max_width-self.min_width))

            self._update_traffic_lights(dt)

            # ---- 렌더링 ----
            self.screen.fill((180, 200, 180))
            sub = self.screen.subsurface(
                pygame.Rect(0, self.ui_height, self.width, self.height-self.ui_height))

            for seg in self.segments:
                self._draw_segment(sub, seg, preview=False)

            real_mouse = (mx, my - self.ui_height)
            if self.current_tool == 'line' and self.drag_start:
                self._draw_segment(sub,
                    {'type':'line','points':[self.drag_start, real_mouse],'width':self.track_width},
                    preview=True)
            elif self.current_tool == 'curve' and self.temp_points:
                for p in self.temp_points:
                    pygame.draw.circle(sub, self.RED, p, 5)
                pygame.draw.line(sub, self.RED, self.temp_points[-1], real_mouse, 1)

            if self.start_pos: self._draw_start_finish_line(sub, self.start_pos, True)
            if self.end_pos:   self._draw_start_finish_line(sub, self.end_pos,   False)
            for cp in self.checkpoints:
                pygame.draw.circle(sub, self.YELLOW, cp, 12)
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

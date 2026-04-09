import pygame
import numpy as np
import json
from typing import List, Tuple, Optional
import math

class TrackEditor:
    """
    트랙 에디터
    - 마우스 드래그로 자유 곡선 그리기
    - 직선 툴
    - 베지어 곡선 툴
    - 트랙 굵기 조절
    - 시작점/끝점 설정
    - 저장/불러오기
    """
    
    def __init__(self, width=1200, height=800):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("트랙 에디터")
        self.clock = pygame.time.Clock()
        
        # 색상
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        self.BLUE = (0, 100, 255)
        self.YELLOW = (255, 255, 0)
        self.LIGHT_BLUE = (173, 216, 230)
        
        # 트랙 데이터
        self.track_surface = pygame.Surface((width, height))
        self.track_surface.fill(self.WHITE)
        
        # 그리기 설정
        self.track_width = 60  # 트랙 굵기
        self.min_track_width = 30
        self.max_track_width = 150
        
        # 툴 설정
        self.current_tool = "free"  # free, line, curve, eraser
        self.is_drawing = False
        self.last_pos = None
        
        # 직선 툴용
        self.line_start = None
        
        # 베지어 곡선 툴용
        self.bezier_points = []
        
        # 시작점/끝점
        self.start_pos = None
        self.end_pos = None
        self.placing_start = False
        self.placing_end = False
        
        self.checkpoints = [] # 수정
        self.placing_checkpoint = False # 수정
        
        # UI 설정
        self.ui_height = 100
        self.buttons = self._create_buttons()
        
        # 폰트
        try:
            # 시스템 폰트 시도 (한글 지원)
            self.font = pygame.font.SysFont('malgungothic,nanumgothic,applegothic,sans-serif', 20)
            self.small_font = pygame.font.SysFont('malgungothic,nanumgothic,applegothic,sans-serif', 16)
        except:
            # 기본 폰트 사용 (영문만)
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 20)
        
    def _create_buttons(self):
        """UI 버튼 생성"""
        buttons = []
        button_width = 100
        button_height = 40
        spacing = 10
        y = 10
        
        # 툴 버튼들
        tools = [
            ("Free Draw", "free"),
            ("Line", "line"),
            ("Curve", "curve"),
            ("Eraser", "eraser"),
        ]
        
        x = 10
        for i, (name, tool) in enumerate(tools):
            buttons.append({
                'rect': pygame.Rect(x, y, button_width, button_height),
                'text': name,
                'action': 'tool',
                'value': tool
            })
            x += button_width + spacing
        
        # 기능 버튼들
        x = 10
        y += button_height + spacing
        
        functions = [
            ("Start", "start"),
            ("End", "end"),
            ("Checkpoint", "checkpoint"), # 수정
            ("Clear", "clear"),
            ("Save", "save"),
            ("Load", "load"),
        ]
        
        for name, action in functions:
            buttons.append({
                'rect': pygame.Rect(x, y, button_width, button_height),
                'text': name,
                'action': action,
                'value': None
            })
            x += button_width + spacing
        
        return buttons
    
    def _draw_bezier_curve(self, points: List[Tuple[int, int]], color, width):
        """베지어 곡선 그리기 (4점 기준)"""
        if len(points) < 2:
            return
        
        # 3차 베지어 곡선
        if len(points) >= 4:
            p0, p1, p2, p3 = points[:4]
            prev_point = p0
            
            for t in np.linspace(0, 1, 50):
                # 베지어 공식
                x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
                y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
                
                curr_point = (int(x), int(y))
                pygame.draw.line(self.track_surface, color, prev_point, curr_point, width)
                prev_point = curr_point
        else:
            # 점이 부족하면 직선으로
            for i in range(len(points) - 1):
                pygame.draw.line(self.track_surface, color, points[i], points[i+1], width)
    
    def _handle_free_draw(self, pos):
        """자유 그리기"""
        if self.last_pos:
            pygame.draw.line(self.track_surface, self.DARK_GRAY, self.last_pos, pos, self.track_width)
            pygame.draw.circle(self.track_surface, self.DARK_GRAY, pos, self.track_width // 2)
        self.last_pos = pos
    
    def _handle_line_tool(self, pos, mouse_pressed):
        """직선 툴"""
        if mouse_pressed:
            if self.line_start is None:
                self.line_start = pos
            else:
                pygame.draw.line(self.track_surface, self.DARK_GRAY, self.line_start, pos, self.track_width)
                self.line_start = None
    
    def _handle_curve_tool(self, pos, mouse_pressed):
        """베지어 곡선 툴 (4점)"""
        if mouse_pressed:
            self.bezier_points.append(pos)
            
            if len(self.bezier_points) == 4:
                self._draw_bezier_curve(self.bezier_points, self.DARK_GRAY, self.track_width)
                self.bezier_points = []
    
    def _handle_eraser(self, pos):
        """지우개"""
        if self.last_pos:
            pygame.draw.line(self.track_surface, self.WHITE, self.last_pos, pos, self.track_width)
            pygame.draw.circle(self.track_surface, self.WHITE, pos, self.track_width // 2)
        self.last_pos = pos
        
    def _draw_ui(self):
        """UI 그리기"""
        # UI 배경
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, (0, 0, self.width, self.ui_height))
        
        # 버튼 그리기
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            # 버튼 색상 (현재 툴이면 하이라이트)
            if button['action'] == 'tool' and button['value'] == self.current_tool:
                color = self.BLUE
            elif button['action'] == 'checkpoint' and self.placing_checkpoint:  # 수정: checkpoint 활성화 시 하이라이트
                color = self.BLUE
            elif button['rect'].collidepoint(mouse_pos):
                color = self.GRAY
            else:
                color = self.WHITE
            
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.BLACK, button['rect'], 2)
            
            # 버튼 텍스트
            text = self.small_font.render(button['text'], True, self.BLACK)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
        
        slider_x = self.width - 300
        slider_y = 30
        slider_width = 200
        # 슬라이더 배경
        pygame.draw.rect(self.screen, self.GRAY, (slider_x, slider_y, slider_width, 10))
        
        # 슬라이더 핸들
        slider_pos = slider_x + (self.track_width - self.min_track_width) / (self.max_track_width - self.min_track_width) * slider_width
        pygame.draw.circle(self.screen, self.BLUE, (int(slider_pos), slider_y + 5), 8)
        
        # 슬라이더 라벨
        label = self.small_font.render(f"Track Width: {self.track_width}", True, self.BLACK)
        self.screen.blit(label, (slider_x, slider_y - 20))
        
        # 현재 툴 표시
        tool_names = {
            "free": "Free Draw",
            "line": "Line Tool",
            "curve": "Curve Tool (4 clicks)",
            "eraser": "Eraser"
        }
        tool_text = self.small_font.render(f"Tool: {tool_names[self.current_tool]}", True, self.BLACK)
        self.screen.blit(tool_text, (10, self.ui_height - 25))
        
        # 수정: checkpoint 활성화 상태 표시
        if self.placing_checkpoint:
            checkpoint_text = self.small_font.render("Checkpoint Mode: ON (Click to place)", True, self.RED)
            self.screen.blit(checkpoint_text, (200, self.ui_height - 25))
        
        # 도움말
        if self.current_tool == "curve":
            help_text = self.small_font.render(f"Points: {len(self.bezier_points)}/4", True, self.RED)
            self.screen.blit(help_text, (200, self.ui_height - 25))
        elif self.current_tool == "line" and self.line_start:
            help_text = self.small_font.render("Click end point", True, self.RED)
            self.screen.blit(help_text, (200, self.ui_height - 25))
    
    # def _draw_ui(self):
    #     """UI 그리기"""
    #     # UI 배경
    #     pygame.draw.rect(self.screen, self.LIGHT_BLUE, (0, 0, self.width, self.ui_height))
        
    #     # 버튼 그리기
    #     mouse_pos = pygame.mouse.get_pos()
        
    #     for button in self.buttons:
    #         # 버튼 색상 (현재 툴이면 하이라이트)
    #         if button['action'] == 'tool' and button['value'] == self.current_tool:
    #             color = self.BLUE
    #         elif button['rect'].collidepoint(mouse_pos):
    #             color = self.GRAY
    #         else:
    #             color = self.WHITE
            
    #         pygame.draw.rect(self.screen, color, button['rect'])
    #         pygame.draw.rect(self.screen, self.BLACK, button['rect'], 2)
            
    #         # 버튼 텍스트
    #         text = self.small_font.render(button['text'], True, self.BLACK)
    #         text_rect = text.get_rect(center=button['rect'].center)
    #         self.screen.blit(text, text_rect)
        
    #     # 트랙 굵기 슬라이더
    #     slider_x = self.width - 300
    #     slider_y = 30
    #     slider_width = 200
        
    #     # 슬라이더 배경
    #     pygame.draw.rect(self.screen, self.GRAY, (slider_x, slider_y, slider_width, 10))
        
    #     # 슬라이더 핸들
    #     slider_pos = slider_x + (self.track_width - self.min_track_width) / (self.max_track_width - self.min_track_width) * slider_width
    #     pygame.draw.circle(self.screen, self.BLUE, (int(slider_pos), slider_y + 5), 8)
        
    #     # 슬라이더 라벨
    #     label = self.small_font.render(f"Track Width: {self.track_width}", True, self.BLACK)
    #     self.screen.blit(label, (slider_x, slider_y - 20))
        
    #     # 현재 툴 표시
    #     tool_names = {
    #         "free": "Free Draw",
    #         "line": "Line Tool",
    #         "curve": "Curve Tool (4 clicks)",
    #         "eraser": "Eraser"
    #     }
    #     tool_text = self.small_font.render(f"Tool: {tool_names[self.current_tool]}", True, self.BLACK)
    #     self.screen.blit(tool_text, (10, self.ui_height - 25))
        
    #     # 도움말
    #     if self.current_tool == "curve":
    #         help_text = self.small_font.render(f"Points: {len(self.bezier_points)}/4", True, self.RED)
    #         self.screen.blit(help_text, (200, self.ui_height - 25))
    #     elif self.current_tool == "line" and self.line_start:
    #         help_text = self.small_font.render("Click end point", True, self.RED)
    #         self.screen.blit(help_text, (200, self.ui_height - 25))
    
    def _draw_track_area(self):
        """트랙 영역 그리기"""
        # 트랙 표면을 화면에 그리기
        self.screen.blit(self.track_surface, (0, self.ui_height))
        
        # 시작점 표시
        if self.start_pos:
            pygame.draw.circle(self.screen, self.GREEN, 
                             (self.start_pos[0], self.start_pos[1] + self.ui_height), 15)
            text = self.small_font.render("S", True, self.WHITE)
            text_rect = text.get_rect(center=(self.start_pos[0], self.start_pos[1] + self.ui_height))
            self.screen.blit(text, text_rect)
        
        # 끝점 표시
        if self.end_pos:
            pygame.draw.circle(self.screen, self.RED, 
                             (self.end_pos[0], self.end_pos[1] + self.ui_height), 15)
            text = self.small_font.render("E", True, self.WHITE)
            text_rect = text.get_rect(center=(self.end_pos[0], self.end_pos[1] + self.ui_height))
            self.screen.blit(text, text_rect)
        
        # 수정
        for i, cp in enumerate(self.checkpoints):
            pygame.draw.circle(self.screen, self.YELLOW,
                            (cp[0], cp[1] + self.ui_height), 12)
            text = self.small_font.render(str(i+1), True, self.BLACK)
            text_rect = text.get_rect(center=(cp[0], cp[1] + self.ui_height))
            self.screen.blit(text, text_rect)
        
        # 베지어 곡선 점들 표시 (미리보기)
        for i, point in enumerate(self.bezier_points):
            pygame.draw.circle(self.screen, self.YELLOW, 
                             (point[0], point[1] + self.ui_height), 5)
            text = self.small_font.render(str(i+1), True, self.BLACK)
            self.screen.blit(text, (point[0] + 10, point[1] + self.ui_height - 10))
        
        # 직선 툴 미리보기
        if self.current_tool == "line" and self.line_start:
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[1] > self.ui_height:
                preview_end = (mouse_pos[0], mouse_pos[1] - self.ui_height)
                pygame.draw.line(self.screen, self.YELLOW, 
                               (self.line_start[0], self.line_start[1] + self.ui_height),
                               mouse_pos, 2)
    
    def _handle_slider(self, mouse_pos):
        """슬라이더 드래그 처리"""
        slider_x = self.width - 300
        slider_width = 200
        
        if slider_x <= mouse_pos[0] <= slider_x + slider_width:
            ratio = (mouse_pos[0] - slider_x) / slider_width
            self.track_width = int(self.min_track_width + ratio * (self.max_track_width - self.min_track_width))
    
    def save_track(self, filename="track.json"):
        """트랙 저장"""
        # 트랙 픽셀 데이터를 배열로 변환
        track_array = pygame.surfarray.array3d(self.track_surface)
        track_array = np.transpose(track_array, (1, 0, 2))  # (height, width, channels)
        
        # 그레이스케일로 변환 (트랙 = 검은색 부분)
        gray = np.mean(track_array, axis=2)
        track_mask = (gray < 200).astype(np.uint8) * 255  # 흰색이 아닌 부분
        
        data = {
            'width': self.width,
            'height': self.height - self.ui_height,
            'track_mask': track_mask.tolist(),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'checkpoints': self.checkpoints # 수정
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print(f"Track saved: {filename}")
    
    def load_track(self, filename="track.json"):
        """트랙 불러오기"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # 트랙 마스크 복원
            track_mask = np.array(data['track_mask'], dtype=np.uint8)
            
            # 트랙 표면에 그리기
            self.track_surface.fill(self.WHITE)
            
            for y in range(track_mask.shape[0]):
                for x in range(track_mask.shape[1]):
                    if track_mask[y, x] > 0:
                        self.track_surface.set_at((x, y), self.DARK_GRAY)
            
            self.start_pos = data.get('start_pos')
            self.end_pos = data.get('end_pos')
            
            self.checkpoints = data.get('checkpoints', []) # 수정
            
            print(f"Track loaded: {filename}")
        except Exception as e:
            print(f"Load failed: {e}")
    
    def run(self):
        """에디터 실행"""
        running = True
        dragging_slider = False
        
        print("=" * 60)
        print("Track Editor Controls")
        print("=" * 60)
        print("Free Draw: Drag to draw track")
        print("Line: Click start and end points")
        print("Curve: Click 4 points in order (Bezier curve)")
        print("Eraser: Drag to erase")
        print("Mouse Wheel: Adjust track width")
        print("=" * 60)
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 왼쪽 클릭
                        # UI 영역 클릭 체크
                        if mouse_pos[1] < self.ui_height:
                            # 버튼 클릭 체크
                            for button in self.buttons:
                                if button['rect'].collidepoint(mouse_pos):
                                    if button['action'] == 'tool':
                                        self.current_tool = button['value']
                                        self.line_start = None
                                        self.bezier_points = []
                                        self.placing_checkpoint = False  # 수정: 툴 변경 시 checkpoint 모드 해제
                                    elif button['action'] == 'start':
                                        self.placing_start = True
                                        self.placing_checkpoint = False  # 수정: 다른 기능 사용 시 checkpoint 모드 해제
                                    elif button['action'] == 'end':
                                        self.placing_end = True
                                        self.placing_checkpoint = False  # 수정: 다른 기능 사용 시 checkpoint 모드 해제
                                        
                                    elif button['action'] == 'checkpoint': 
                                        self.placing_checkpoint = not self.placing_checkpoint  # 수정: 토글 방식으로 변경
                                        
                                    elif button['action'] == 'clear':
                                        self.track_surface.fill(self.WHITE)
                                        self.start_pos = None
                                        self.end_pos = None
                                        self.placing_checkpoint = False  # 수정: clear 시 checkpoint 모드 해제
                                    elif button['action'] == 'save':
                                        self.save_track()
                                        self.placing_checkpoint = False  # 수정: save 시 checkpoint 모드 해제
                                    elif button['action'] == 'load':
                                        self.load_track()
                                        self.placing_checkpoint = False  # 수정: load 시 checkpoint 모드 해제
                            
                            # 슬라이더 클릭 체크
                            slider_x = self.width - 300
                            slider_y = 20
                            slider_width = 200
                            if slider_x <= mouse_pos[0] <= slider_x + slider_width and \
                            slider_y <= mouse_pos[1] <= slider_y + 30:
                                dragging_slider = True
                                self._handle_slider(mouse_pos)
                        else:
                            # 트랙 영역 클릭
                            track_pos = (mouse_pos[0], mouse_pos[1] - self.ui_height)
                            
                            if self.placing_start:
                                self.start_pos = track_pos
                                self.placing_start = False
                            elif self.placing_end:
                                self.end_pos = track_pos
                                self.placing_end = False
                                
                            elif self.placing_checkpoint: 
                                self.checkpoints.append(track_pos)
                                # 수정: placing_checkpoint = False 제거 (계속 찍을 수 있도록)
                            
                            else:
                                if self.current_tool in ["free", "eraser"]:
                                    self.is_drawing = True
                                    self.last_pos = track_pos
                                elif self.current_tool == "line":
                                    self._handle_line_tool(track_pos, True)
                                elif self.current_tool == "curve":
                                    self._handle_curve_tool(track_pos, True)
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     if event.button == 1:  # 왼쪽 클릭
                #         # UI 영역 클릭 체크
                #         if mouse_pos[1] < self.ui_height:
                #             # 버튼 클릭 체크
                #             for button in self.buttons:
                #                 if button['rect'].collidepoint(mouse_pos):
                #                     if button['action'] == 'tool':
                #                         self.current_tool = button['value']
                #                         self.line_start = None
                #                         self.bezier_points = []
                #                     elif button['action'] == 'start':
                #                         self.placing_start = True
                #                     elif button['action'] == 'end':
                #                         self.placing_end = True
                                        
                #                     elif button['action'] == 'checkpoint': 
                #                         self.placing_checkpoint = True # 수정
                                        
                #                     elif button['action'] == 'clear':
                #                         self.track_surface.fill(self.WHITE)
                #                         self.start_pos = None
                #                         self.end_pos = None
                #                     elif button['action'] == 'save':
                #                         self.save_track()
                #                     elif button['action'] == 'load':
                #                         self.load_track()
                            
                #             # 슬라이더 클릭 체크
                #             slider_x = self.width - 300
                #             slider_y = 20
                #             slider_width = 200
                #             if slider_x <= mouse_pos[0] <= slider_x + slider_width and \
                #                slider_y <= mouse_pos[1] <= slider_y + 30:
                #                 dragging_slider = True
                #                 self._handle_slider(mouse_pos)
                #         else:
                #             # 트랙 영역 클릭
                #             track_pos = (mouse_pos[0], mouse_pos[1] - self.ui_height)
                            
                #             if self.placing_start:
                #                 self.start_pos = track_pos
                #                 self.placing_start = False
                #             elif self.placing_end:
                #                 self.end_pos = track_pos
                #                 self.placing_end = False
                                
                #             elif self.placing_checkpoint: 
                #                 self.checkpoints.append(track_pos)
                #                 self.placing_checkpoint = False # 수정
                            
                #             else:
                #                 if self.current_tool in ["free", "eraser"]:
                #                     self.is_drawing = True
                #                     self.last_pos = track_pos
                #                 elif self.current_tool == "line":
                #                     self._handle_line_tool(track_pos, True)
                #                 elif self.current_tool == "curve":
                #                     self._handle_curve_tool(track_pos, True)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.is_drawing = False
                        self.last_pos = None
                        dragging_slider = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if dragging_slider:
                        self._handle_slider(mouse_pos)
                    elif self.is_drawing and mouse_pos[1] > self.ui_height:
                        track_pos = (mouse_pos[0], mouse_pos[1] - self.ui_height)
                        if self.current_tool == "free":
                            self._handle_free_draw(track_pos)
                        elif self.current_tool == "eraser":
                            self._handle_eraser(track_pos)
                
                elif event.type == pygame.MOUSEWHEEL:
                    # 마우스 휠로 트랙 굵기 조절
                    self.track_width += event.y * 5
                    self.track_width = max(self.min_track_width, min(self.max_track_width, self.track_width))
            
            # 화면 그리기
            self.screen.fill(self.WHITE)
            self._draw_track_area()
            self._draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    editor = TrackEditor()
    editor.run()

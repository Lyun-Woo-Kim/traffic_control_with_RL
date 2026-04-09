# Traffic Simulation with Multi-Agent Reinforcement Learning

2D 교통 시뮬레이션 환경에서 **멀티 에이전트 자율 주행**을 강화학습으로 학습시키는 프로젝트입니다.  
신호등, 교차로, 차선 방향 정보가 포함된 도로 환경에서 에이전트가 교통 규칙을 스스로 학습합니다.

---

## 프로젝트 구조

```
traffic_control_with_RL/
├── traffic_simulation/          # 메인 코드 (현재 진행 중)
│   ├── map_editor.py            # 트랙 편집기 (신호등, 차선 방향, 교차로 지원)
│   ├── traffic_env.py           # 강화학습 환경 (RacingGame 클래스)
│   ├── train_dddqn.py           # DDDQN Dual Head 학습 스크립트
│   └── vehicle_config.json      # 차량 물리 파라미터 설정
│
├── racing_simulation/           # 기존 단일 에이전트 드리프트 레이싱 (참고용)
│   ├── env/                     # 환경 코드
│   ├── train/                   # 학습 스크립트
│   ├── evaluation/              # 평가 스크립트
│   ├── reward_function.py       # 보상 함수
│   └── README.md                # 기존 프로젝트 설명
│
├── example_models/              # 사전 학습된 모델 가중치
├── log_files/                   # 학습 로그
├── results/                     # 학습 결과 그래프 (기존 프로젝트)
├── analysis.py                  # 로그 분석 스크립트
└── requirements.txt             # 의존성 목록
```

---

## 핵심 기능

### 도로 환경 (`map_editor.py`)
- 직선/곡선 차선을 자유롭게 그리는 트랙 편집기
- **10×10 픽셀 그리드** 기반 방향 벡터 저장 (전체 맵 커버)
- 교차로 자동 감지: 두 차선이 겹치는 셀 → `is_intersection=1`, 방향 벡터=`[0.0, 0.0]`
- **신호등 시스템**: 각 차선에 신호등 배치 가능
  - 🔴 Red (고정 7초) → 🟢 Green (랜덤 5~15초) → 🟡 Yellow (고정 3초) → 🔴 Red
- 저장 포맷 (`track_data.json`): `direction_grid`, `lane_data`, `traffic_lights`, `checkpoints`

### 강화학습 환경 (`traffic_env.py`)
- **8방향 레이캐스팅 센서**: 차량 중심 기준 8방향으로 차선 경계까지 거리 측정
- **고정 크기 상태 벡터 (Input Size = 29)**:

| 인덱스 | 항목 | 설명 |
|--------|------|------|
| 0~7 | 레이캐스팅 센서 | 8방향 차선 경계까지의 거리 (정규화) |
| 8~15 | 차량 상태 | 속도, 각속도, sin/cos(heading), 드리프트 여부, 체크포인트 진행도, 스티어링 입력, 가속 입력 |
| 16~23 | 차량 특성 | 물리 파라미터 (최대속도, 가속도, 마찰 등) |
| 24~25 | 도로 방향 벡터 | 현재 위치의 `dir_x`, `dir_y` (교차로 = `[0.0, 0.0]`) |
| 26 | 교차로 여부 | `0` = 일반 도로, `1` = 교차로 |
| 27 | 신호등 존재 여부 | `0` = 없음, `1` = 있음 |
| 28 | 신호등 상태 | `0` = 빨강, `1` = 노랑, `2` = 초록 |

### 학습 알고리즘 (`train_dddqn.py`)
- **Dueling Double DQN (DDDQN) + Dual Head**
  - Action Head: 어떤 조작을 할지 결정
  - Duration Head: 그 조작을 얼마나 유지할지 결정
- 체크포인트 순서를 **인덱스로 직접 지정** (`CHECKPOINT_ORDER = [0, 1, 2, ...]`)
- 세그먼트별 학습: 이전 구간 성공 후 다음 구간으로 점진적 확장

---

## 스케일 기준

| 픽셀 | 실제 거리 |
|------|----------|
| 1px | 약 5cm |
| 100px | 5m (도로 폭 기준) |
| 20px | 1m |

---

## 설치 및 실행

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 트랙 편집기 실행
```bash
python traffic_simulation/map_editor.py
```

### 학습 실행
```bash
python traffic_simulation/train_dddqn.py
```

---

## 차량 설정 (`vehicle_config.json`)

차량의 물리 파라미터는 `vehicle_config.json`에서 조정할 수 있습니다.  
주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `max_speed` | 300 | 최대 속도 (px/s) ≈ 54 km/h |
| `acceleration_force` | 200 | 가속력 |
| `brake_force` | 300 | 제동력 |
| `width` | 18 | 차량 너비 (px) ≈ 0.9m |
| `length` | 36 | 차량 길이 (px) ≈ 1.8m |

---

## 기존 프로젝트

단일 에이전트 드리프트 레이싱 기반 프로젝트는 [`racing_simulation/`](./racing_simulation/README.md) 폴더를 참고하세요.

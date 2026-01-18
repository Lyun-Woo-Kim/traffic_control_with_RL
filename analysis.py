import matplotlib.pyplot as plt
import numpy as np
import os
import re

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    'red', 'blue', 'green', 'black', 'orange', 
    'purple', 'brown', 'pink', 'gray', 'olive'
]

MARKERS = [
    'o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h',
    'H', 'X', 'x', 'P', '8', '1', '2', '3', '4', '+'
]

# -----------------------------
# 정규식 유틸(숫자 패턴)
# -----------------------------
NUM = r'[-+]?\d+(?:\.\d+)?'

# -----------------------------
# 훈련 로그: [Ep 1200] ... Reward: -160.3 ... Avg Speed: 49.31 km/h
# -----------------------------
re_train_ep = re.compile(
    rf'\[Ep\s*(\d+)\].*?Reward:\s*({NUM}).*?Avg Speed:\s*({NUM})\s*km/h',
    re.IGNORECASE
)

# -----------------------------
# 세그먼트 종료: ★★★ Segment 3 학습 완료! (Episode: 3333) ★★★
# -----------------------------
re_segment = re.compile(
    r'Segment\s*(\d+).*?\(Episode:\s*(\d+)\)',
    re.IGNORECASE
)

# -----------------------------
# GOAL: ✓ GOAL! Episode 51172, Avg Speed 0.2989km/h), Total Goals: 50
# -----------------------------
re_goal = re.compile(
    rf'GOAL!\s*Episode\s*(\d+),\s*Avg Speed\s*({NUM})\s*km/h',
    re.IGNORECASE
)

# -----------------------------
# Evaluation
# -----------------------------
re_eval_base = re.compile(
    r'Evaluation:\s*(\d+)\s*/\s*(\d+)\s*success',
    re.IGNORECASE
)
re_eval_avg_speed = re.compile(
    rf'Avg Speed\s*({NUM})\s*km/h',
    re.IGNORECASE
)
re_eval_avg_time = re.compile(
    rf'Avg Time:\s*({NUM})s',
    re.IGNORECASE
)

# -----------------------------
# best model saved 이벤트
# -----------------------------
re_best_saved = re.compile(
    rf'New best model saved!\s*\(Episode:\s*(\d+),\s*Avg Speed\s*({NUM})\s*km/h',
    re.IGNORECASE
)

def parse_logs(log_dir: str):
    txt_list = [f for f in os.listdir(log_dir) if f.endswith('.txt')]
    file_paths = [os.path.join(log_dir, f) for f in txt_list]
    all_results = {}

    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        all_results[file_name] = {
            '100ep_rewards': {}, '100ep_avg_speed': {},
            'segment_ep': {}, 'goal_avg_speed_by_ep': {},
            'eval_success_rate': {}, 'eval_success': {}, 'eval_total': {},
            'eval_avg_speed': {}, 'eval_avg_time': {}, 'best_saved': {}
        }
        eval_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line: continue
                
                m = re_train_ep.search(line)
                if m:
                    ep, reward, avg_speed = int(m.group(1)), float(m.group(2)), float(m.group(3))
                    if ep % 100 == 0:
                        all_results[file_name]['100ep_rewards'][ep] = reward
                        all_results[file_name]['100ep_avg_speed'][ep] = avg_speed
                    continue

                m = re_segment.search(line)
                if m:
                    all_results[file_name]['segment_ep'][int(m.group(1))] = int(m.group(2))
                    continue

                m = re_goal.search(line)
                if m:
                    all_results[file_name]['goal_avg_speed_by_ep'][int(m.group(1))] = float(m.group(2))
                    continue

                if 'Evaluation:' in line:
                    mb = re_eval_base.search(line)
                    if mb:
                        eval_count += 1
                        success, total = int(mb.group(1)), int(mb.group(2))
                        all_results[file_name]['eval_success'][eval_count] = success
                        all_results[file_name]['eval_total'][eval_count] = total
                        all_results[file_name]['eval_success_rate'][eval_count] = (success / total) * 100.0
                        ms = re_eval_avg_speed.search(line)
                        all_results[file_name]['eval_avg_speed'][eval_count] = float(ms.group(1)) if ms else None
                        mt = re_eval_avg_time.search(line)
                        all_results[file_name]['eval_avg_time'][eval_count] = float(mt.group(1)) if mt else None
                    continue

                m = re_best_saved.search(line)
                if m:
                    all_results[file_name]['best_saved'][int(m.group(1))] = float(m.group(2))
                    continue
    return all_results

def show_plots(all_results, plot_keys=['reward_100', 'reward_2000', 'speed_100', 'speed_2000', 'checkpoint'], avg_size=2000, save_plots=False):
    tick_size = int(avg_size/100)
    
    if save_plots:
        os.makedirs('./results', exist_ok=True)
    
    for key in plot_keys:
        plt.figure(figsize=(10, 6))
        
        for i, (file_name, results) in enumerate(all_results.items()):
            label = " ".join(file_name.split('_')[:2]) if 'DualHead' in file_name else file_name.split('_')[0]
            
            ### 에피소드 별 reward 변화량 그래프(100 episode)
            if key == 'reward_100':
                plt.plot(list(results['100ep_rewards']), list(results['100ep_rewards'].values()), linewidth=0.5, label=label)
                plt.title('Rewards (100 episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Rewards')

            ### 2000에피소드 마다 평균 낸 reward 변화량 그래프
            elif key == 'reward_2000':
                data = list(results['100ep_rewards'].values())
                x_list = [x*avg_size for x in range(len(data)//tick_size)]
                y_data = [np.mean(data[j*tick_size:(j+1)*tick_size]) for j in range(len(x_list))]
                plt.plot(x_list, y_data, linewidth=1, label=label, marker=MARKERS[i%len(MARKERS)], markersize=3)
                plt.title(f'Avg Rewards ({avg_size} episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Rewards')

            ### 에피소드 별 평균 속도 변화량 그래프(100 episode)
            elif key == 'speed_100':
                plt.plot(list(results['100ep_avg_speed']), list(results['100ep_avg_speed'].values()), linewidth=0.5, label=label)
                plt.title('Speed (100 episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Speed (km/h)')

            ### 2000에피소드 마다 평균 낸 평균 속도 변화량 그래프
            elif key == 'speed_2000':
                data = list(results['100ep_avg_speed'].values())
                x_list = [x*avg_size for x in range(len(data)//tick_size)]
                y_data = [np.mean(data[j*tick_size:(j+1)*tick_size]) for j in range(len(x_list))]
                plt.plot(x_list, y_data, linewidth=1, label=label, marker=MARKERS[i%len(MARKERS)], markersize=3)
                plt.title(f'Avg Speed ({avg_size} episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Speed (km/h)')

            ### 각 Checkpoint를 도달한 episode
            elif key == 'checkpoint':
                plt.plot(list(results['segment_ep']), list(results['segment_ep'].values()), linewidth=1, label=label, marker=MARKERS[i%len(MARKERS)], markersize=4)
                plt.title('Episodes with 100 Checkpoint Successes')
                plt.xlabel('Checkpoint')
                plt.xticks(np.array(list(results['segment_ep'].keys())))
                plt.ylabel('Episodes')

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_plots:
            save_path = f'./results/{key}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved: {save_path}")
        
        plt.show()
        plt.close()

# -----------------------------
# 실행부
# -----------------------------
LOG_DIR = './log_files/'
results_data = parse_logs(LOG_DIR)

# save_plots=True 를 추가하여 ./results 폴더에 자동 저장
show_plots(results_data, 
           plot_keys=['reward_100', 'reward_2000', 'speed_100', 'speed_2000', 'checkpoint'], 
           save_plots=True)
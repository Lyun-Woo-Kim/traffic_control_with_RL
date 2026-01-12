#%%
import os
import re

all_results = {}

txt_list = os.listdir('/rwkim/AI-DEV/IoT-S_xqbot/files/RLs')
txt_list = [txt for txt in txt_list if txt.endswith('.txt')]
file_paths = [f'/rwkim/AI-DEV/IoT-S_xqbot/files/RLs/{txt}' for txt in txt_list]

for file_path in file_paths: 
    file_name = file_path.split('/')[-1].split('.')[0]
    all_results[file_name] = {
        '100ep_rewards': {},
        '100ep_avg_speed': {},
        'segment_ep': {},
        'eval_success_ep': {},
        'eval_avg_time': {},
        'eval_success_rate': {}
    }
    
    eval_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 패턴 1: [Ep XXXXX] ... Reward: ... Avg Speed: ...
            if '[Ep' in line and 'Reward:' in line:
                ep_match = re.search(r'\[Ep\s+(\d+)\]', line)
                reward_match = re.search(r'Reward:\s+([-\d.]+)', line)
                speed_match = re.search(r'Avg Speed:\s+([\d.]+)', line)
                
                if ep_match and reward_match and speed_match:
                    ep = int(ep_match.group(1))
                    if ep % 100 == 0:
                        all_results[file_name]['100ep_rewards'][ep] = float(reward_match.group(1))
                        all_results[file_name]['100ep_avg_speed'][ep] = float(speed_match.group(1))
            
            # 패턴 2: ★★★ Segment XX 학습 완료! (Episode: XXXXX) ★★★
            elif 'Segment' in line and '학습 완료' in line:
                match = re.search(r'Segment\s+(\d+).*Episode:\s+(\d+)', line)
                if match:
                    segment = int(match.group(1))
                    episode = int(match.group(2))
                    all_results[file_name]['segment_ep'][segment] = episode
            
            # 패턴 3: ✓ GOAL! Episode XXXXX, Time: X.XXs
            elif 'GOAL!' in line and 'Episode' in line:
                match = re.search(r'Episode\s+(\d+),\s+Time:\s+([\d.]+)s', line)
                if match:
                    episode = int(match.group(1))
                    time = float(match.group(2))
                    all_results[file_name]['eval_success_ep'][episode] = time
            
            # 패턴 4 & 5: Evaluation 결과
            elif 'Evaluation:' in line:
                eval_count += 1
                success_match = re.search(r'(\d+)/(\d+)\s+success', line)
                if success_match:
                    success = int(success_match.group(1))
                    total = int(success_match.group(2))
                    all_results[file_name]['eval_success_rate'][eval_count] = (success / total) * 100
                
                time_match = re.search(r'Avg Time:\s+([\d.]+)s', line)
                if time_match:
                    all_results[file_name]['eval_avg_time'][eval_count] = float(time_match.group(1))
#%%
import matplotlib.pyplot as plt

# 결과 확인
for file_name, results in all_results.items():
    print(f"\n=== {file_name} ===")
    print(f"100ep_rewards: {len(results['100ep_rewards'])}개")
    print(f"segment_ep: {len(results['segment_ep'])}개")
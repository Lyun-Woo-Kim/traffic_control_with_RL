def get_frame_rewards(state, coll, goal, curr_distance, curr_time, max_time, 
                          cp_r, dis_gap, action_index, max_speed, prev_distance): 
        """
        [수정됨] 단일 프레임 보상 계산
        - 직진 구간(전방 센서값 높음)에서 드리프트 시 강력한 페널티 적용
        - 과도한 드리프트 보너스 계수 하향 조정 (1.0 -> 0.5)
        """
        # state 구조: [sensors(12), cos, sin, speed, vel_x, vel_y, dist, angle, is_drifting, ...]
        # get_sensors 로직상 index 0이 정면(0도), 1이 +30도, 11이 -30도(330도)입니다.
        front_sensors = [state[0], state[1], state[11]]
        front_clearance = sum(front_sensors) / 3.0  # 전방 3개 센서의 평균 거리 (0~1)

        speed_norm = state[14]
        is_drifting = state[19] > 0.5
        is_turning = action_index in [0, 1, 2]
        is_brake_only = action_index == 5
        
        # 1. 기본 이벤트 보상
        if coll: 
            # 충돌은 가장 치명적이므로 -200 유지
            return -200
        elif goal: 
            time_ratio = curr_time / (max_time * 1000)
            # 빨리 도착할수록 보상 (최대 600)
            r = 100 + (1 - time_ratio) ** 2 * 600.0
        else: 
            r = 0
        
        # 2. 진행 보상 (목표지점 접근 시 +)
        r_dis = (prev_distance - curr_distance) * 0.1
        
        # 3. 시간 패널티 (빨리 가도록 유도)
        r_time = 0.01
        
        # 4. 속도 보상 (기본 주행 보상)
        r_speed = speed_norm * 0.3
        
        # 저속 패널티 (지나치게 느리면 감점)
        if speed_norm < 0.15:
            r_speed -= 0.5
        
        # 5. [핵심 수정] 드리프트 보상 및 페널티 로직
        drift_reward = 0
        if is_drifting:
            # (A) 직진 구간 드리프트 페널티
            if front_clearance > 0.7:
                drift_reward = -1.0
                
            # (B) 올바른 드리프트 (코너링 상황 + 고속)
            elif speed_norm > 0.4:
                # 계수를 1.0에서 0.5로 하향 (속도 유지 보조 역할만 수행)
                drift_reward = speed_norm * 0.5
                
            # (C) 저속 드리프트 페널티 (제자리 뱅글뱅글 방지)
            else:
                drift_reward = -0.2
        
        # 6. 브레이크만 밟고 저속일 때 페널티 (멈춤 방지)
        if is_brake_only and speed_norm < 0.2:
            r_speed -= 0.5
        
        # 7. 체크포인트 보상 (드리프트에 따른 가산점)
        if cp_r > 0:
            if is_drifting and speed_norm > 0.4 and front_clearance <= 0.7:
                cp_r *= 2.0 
            elif speed_norm > 0.3:
                cp_r *= 1.5
            elif speed_norm < 0.2:
                cp_r *= 0.3
        
        # 최종 리워드 합산
        reward = r + r_dis + r_speed - r_time + cp_r + drift_reward
        return reward
import numpy as np

# 定义TD3模型的奖励函数
def calculate_reward_global(state, next_state, action, prev_action, params):
    alpha, beta, gamma, T_ideal, H_ideal, T_max, H_max, E, C = params  # 数据网格搜索获得
    E_max = 85  # 数据中获得
    C_max = 4  # 同时调整四个空调状态

    '''
    计算奖励函数
    R(s,a,v) = \alpha*(1-- 0.05 * d_T^normalized - 0.05 * d_H^normalized) - \beta * E_total - \gamma * C_switch)
    d_T^normalized, d_H^normalized:  分别为14个传感器温度、湿度偏离理想范围的程度
    E_total: 4个空调设备的能耗水平(与空调设定温度直接相关)
    C_switch： 4台空调状态切换成本
    '''

    # 14个传感器偏离理想范围,非线性温湿度偏离惩罚 + 标准化温度和湿度偏离值
    d_T = sum((sensor_T - T_ideal) ** 2 for sensor_T in next_state[13:40][::2])  # 温度偏离理想范围的程度
    d_H = sum((sensor_H - H_ideal) ** 2 for sensor_H in next_state[12:40][::2])  # 湿度偏离理想范围的程度
    d_T_normalized = d_T / (14 * T_max ** 2)    # 14 个传感器，每个传感器最大偏离 T_max
    d_H_normalized = d_H / (14 * H_max ** 2)   # 14 个传感器，每个传感器最大偏离 H_max  

    
    E_total = sum(next_state[40+i] for i in range(4))  # 当前能耗水平
    E_penalty = beta * (1 - np.exp(-E_total / E_max))  # 非线性能耗惩罚
    C_switch = sum(abs(next_state[i] - state[i]) * C for i in range(0, 12, 3))  # 空调切换状态成本
    C_penalty = gamma * (1 - np.exp(-C_switch / C_max))   # 非线性切换成本惩罚

    
    # 温度和湿度惩罚项
    temp_variance_penalty = np.var(next_state[13:40][::2]) / T_max
    hum_variance_penalty = np.var(next_state[12:40][::2]) / H_max

    # 计算奖励
    reward = alpha * (1 - 0.05 * d_T_normalized - 0.05 * d_H_normalized) - E_penalty - C_penalty
    reward -= temp_variance_penalty + hum_variance_penalty


    # 增加正奖励，鼓励达到理想状态 24-28, 逐步增加正奖励 0.1 
    positive_reward = 0
    for temp, hum in zip(next_state[13:40][::2], next_state[12:40][::2]):
        if 24 <= temp <= 28 and 40 <= hum <= 60:
            positive_reward += 0.1  # 每个传感器满足条件增加0.1奖励

    reward += positive_reward
    
    return reward


# 定义TD3模型的状态空间
def get_state_global(row):
    return np.array([
        row['ac1_status'], row['ac1_set_T'], row['ac1_return_T'],
        row['ac2_status'], row['ac2_set_T'], row['ac2_return_T'],
        row['ac3_status'], row['ac3_set_T'], row['ac3_return_T'],
        row['ac4_status'], row['ac4_set_T'], row['ac4_return_T'],
        row['sensor1_H'], row['sensor1_T'],
        row['sensor2_H'], row['sensor2_T'],
        row['sensor3_H'], row['sensor3_T'],
        row['sensor4_H'], row['sensor4_T'],
        row['sensor5_H'], row['sensor5_T'],
        row['sensor6_H'], row['sensor6_T'],
        row['sensor7_H'], row['sensor7_T'],
        row['sensor11_H'], row['sensor11_T'],
        row['sensor12_H'], row['sensor12_T'],
        row['sensor13_H'], row['sensor13_T'],
        row['sensor14_H'], row['sensor14_T'],
        row['sensor15_H'], row['sensor15_T'],
        row['sensor16_H'], row['sensor16_T'],
        row['sensor17_H'], row['sensor17_T'],
        row['ac1_P'],row['ac2_P'],row['ac3_P'],row['ac4_P']
    ], dtype=np.float32)





# 定义VDN模型的单个空调局部状态
def get_state_local(row, ac_index):
    if ac_index==1:
        return np.array([
            row[f'ac{ac_index}_status'],
            row[f'ac{ac_index}_set_T'],
            row[f'ac{ac_index}_return_T'],
            row[f'sensor1_T'], row[f'sensor1_H'],
            row[f'sensor2_T'], row[f'sensor2_H'],
            row[f'sensor11_T'], row[f'sensor11_H'],
            row[f'sensor12_T'], row[f'sensor12_H'],
            row[f'ac{ac_index}_P']
        ], dtype=np.float32)      
      
    elif ac_index==2:
        return np.array([
            row[f'ac{ac_index}_status'],
            row[f'ac{ac_index}_set_T'],
            row[f'ac{ac_index}_return_T'],
            row[f'sensor3_T'], row[f'sensor3_H'],
            row[f'sensor4_T'], row[f'sensor4_H'],
            row[f'sensor13_T'], row[f'sensor13_H'],
            row[f'sensor14_T'], row[f'sensor14_H'],
            row[f'ac{ac_index}_P']
        ], dtype=np.float32)

    elif ac_index==3:
        return np.array([
            row[f'ac{ac_index}_status'],
            row[f'ac{ac_index}_set_T'],
            row[f'ac{ac_index}_return_T'],
            row[f'sensor5_T'], row[f'sensor5_H'],
            row[f'sensor6_T'], row[f'sensor6_H'],
            row[f'sensor15_T'], row[f'sensor15_H'],
            row[f'sensor16_T'], row[f'sensor16_H'],
            row[f'ac{ac_index}_P']
        ], dtype=np.float32)

    elif ac_index==4:
        return np.array([
            row[f'ac{ac_index}_status'],
            row[f'ac{ac_index}_set_T'],
            row[f'ac{ac_index}_return_T'],
            row[f'sensor6_T'], row[f'sensor6_H'],
            row[f'sensor7_T'], row[f'sensor7_H'],
            row[f'sensor16_T'], row[f'sensor16_H'],
            row[f'sensor17_T'], row[f'sensor17_H'],
            row[f'ac{ac_index}_P']
        ], dtype=np.float32)

    else:
        return ValueError('Air Condition index not exists!')



# VDN单个空调的奖励函数
def calculate_reward_local(state, next_state, actions, prev_state, params):
    '''
    计算单个空调的奖励函数值
    '''
    
    alpha, beta, gamma, T_ideal, H_ideal, T_max, H_max, E, C = params
    E_max = 30  # 数据中获得
    C_max = 1  # 同时调整四个空调状态
    # 提取状态和动作
    ac_status = state[0]
    ac_set_temp = state[1]
    ax_return_temp =  state[2]
    sensor_humidity = state[4:11][::2]
    sensor_temp = state[3:11][::2]
    ac_P  = state[11]
    ac_status_next = next_state[0]
    

    # 标准化温度和湿度偏离值
    d_T = sum((sensor_T - T_ideal)**2 for sensor_T in sensor_temp)  # 温度偏离理想范围的程度
    d_H = sum((sensor_H - H_ideal)**2 for sensor_H in sensor_humidity) # 湿度偏离理想范围的程度
    d_T_normalized = d_T / (4 * T_max ** 2)    # 4 个传感器，每个传感器最大偏离 T_max
    d_H_normalized = d_H / (4 * H_max ** 2)   # 4 个传感器，每个传感器最大偏离 H_max  

    E_total = ac_status * ac_P   # 当前能耗水平
    E_penalty = beta * (1 - np.exp(-E_total / E_max))  # 非线性能耗惩罚
    
    C_switch = abs(ac_status_next - ac_status) * C   # 空调切换状态成本
    C_penalty = gamma * (1 - np.exp(-C_switch / C_max))   # 非线性切换成本惩罚


    temp_variance_penalty = np.var(next_state[3:11][::2]) / T_max
    hum_variance_penalty = np.var(next_state[4:11][::2]) / H_max


    # 计算奖励
    reward = alpha * (1 - 0.05 * d_T_normalized - 0.05 * d_H_normalized) - E_penalty - C_penalty
    reward -= temp_variance_penalty + hum_variance_penalty

    #额外激励
    positive_reward = 0
    for temp, hum in zip(next_state[4:11][::2], next_state[3:11][::2]):
        if 24 <= temp <= 28 and 40 <= hum <= 60:
            positive_reward += 0.1  # 每个传感器满足条件增加0.1奖励

    reward += positive_reward    

    
    return reward
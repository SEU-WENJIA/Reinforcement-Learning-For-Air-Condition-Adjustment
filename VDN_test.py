import numpy as np
import torch
from models import VDN
from utils import calculate_reward_local, get_state_local
import matplotlib.pyplot as plt



#初始化模型
num_agents = 4    
obs_dim = 12    # 每个智能体的局部状态维度
action_dim = 2  # 每个智能体的局部动作维度
vdns = [VDN(obs_dim, action_dim) for _ in range(num_agents)]


# 测试案例
states = [ [ 1.  , 25.  , 24.  , 26.37, 47.75, 26.86, 45.92, 28.1 , 47.8 , 27.7 , 49.5 ,  3.59],
          [ 1.  , 26.  , 26.9 , 26.05, 47.63, 23.46, 54.39, 25.3 , 49.5 ,  27.  , 44.5 , 29.28],
          [ 1.  , 24.  , 25.3 , 24.59, 50.65, 24.56, 50.69, 25.8 , 58.2 ,25.4 , 55.  , 11.85],
          [ 1.  , 25.  , 26.7 , 24.56, 50.69, 26.08, 44.92, 25.4 , 55.  ,28.8 , 48.3 ,  3.55]]


# 加载训练完成模型
print(f"VDN 模型策略，如下: \n   ")
filename = 'VDN_finished\\'

# 给出四个环境调节策略
for i, vdn in enumerate(vdns):
    vdn_filename = f'{filename}local_value_net_ac{i + 1}.pth'
    state_dict = torch.load(vdn_filename, map_location=torch.device('cpu'))
    vdn.local_value_net.load_state_dict(state_dict)

    #单个环境调节策略
    air_state = states[i]
    actions = vdn.select_action(air_state)
    actions[0] = int((actions[0] + 1) / 2 >= 0.5)
    print(f" 空调 {i+1}: 状态: {actions[0]}, 设定温度: {actions[1]:6.2f}")
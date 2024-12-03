import pandas as pd
import numpy as np
import torch
import datetime
import os
from models import TD3

# 初始化TD3模型
state_dim = 44  # 状态维度
action_dim = 8  # 动作维度（每个空调的状态和设定温度）
max_action = 1.0  # 最大动作值
agent = TD3(state_dim, action_dim, max_action)


# 加载训练完成的TD3模型参数
filename = f'results\\TD3_finished/'

# 加载Actor模型
actor_path = f'{filename}actor_model.pth'
agent.actor.load_state_dict(torch.load(actor_path))

# 加载Critic 1模型
critic1_path = f'{filename}critic1_model.pth'
agent.critic_1.load_state_dict(torch.load(critic1_path))

# 加载Critic 2模型
critic2_path = f'{filename}critic2_model.pth'
agent.critic_2.load_state_dict(torch.load(critic2_path))



# 环境当前状态
state= np.array([ 1.  , 25.  , 24.  ,  1.  , 26.  , 26.9 ,  1.  , 24.  , 25.3 ,
        1.  , 25.  , 26.7 , 47.75, 26.37, 45.92, 26.86, 47.63, 26.05,
       54.39, 23.46, 50.65, 24.59, 50.69, 24.56, 44.92, 26.08, 47.8 ,
       28.1 , 49.5 , 27.7 , 49.5 , 25.3 , 44.5 , 27.  , 58.2 , 25.8 ,
       55.  , 25.4 , 48.3 , 28.8 ,  3.59, 29.28, 11.85,  3.55])


# 给出环境调节策略

action = agent.select_action(state)
action = np.clip(action + np.random.normal(0, 0.1, size=action.shape), -1, 1)        
action_mapped = np.zeros(8)
for i in range(4):
    action_mapped[i * 2] = int((action[i * 2] + 1) / 2 >= 0.5) #(action[i * 2] + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]
    set_T = (action[i * 2 + 1] + 1) * 4 + 18  # 将 [-1, 1] 映射到 [18, 26]
    action_mapped[i * 2 + 1] = np.clip(set_T, 18, 28)  # 确保设定温度在 [18, 26] 范围内

print(f"TD3 模型策略，如下: \n   ")
for i in range(4):
    status = int(action_mapped[i * 2])
    set_temperature = float(action_mapped[i * 2 + 1])
    
    print(f" 空调 {i+1}: 状态: {status}, 设定温度: {set_temperature:6.2f}")
    print()
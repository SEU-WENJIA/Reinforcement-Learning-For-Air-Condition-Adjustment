import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import random
import pandas as pd
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from collections import namedtuple
from itertools import count

import argparse
import gym
import os
import sys
from models import VDN
from utils import calculate_reward_local, get_state_local
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('data\\data01.csv')





# 参数
best_params = (5.74040941e-02,  2.60904007e-04, -1.78204446e+01,  2.52571830e+01,
  6.81185427e+01,  1.99094691e+02,  9.87365351e+02,  1.00000000e+01, -3.35105392e-04)



num_agents = 4   # 初始化环境和代理
obs_dim = 12    # 每个智能体的局部观察维度
action_dim = 2  # 每个智能体的局部动作维度

vdns = [VDN(obs_dim, action_dim) for _ in range(num_agents)]

num_episodes = 1
max_steps = 10000

# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")




filename = f'results\\{current_time}_VDN\\'
if not os.path.exists(filename):
    os.mkdir(filename)


# 打开输出文件，文件名加上当前时间
with open(f'{filename}VDN_training_log_{current_time}.txt', 'w') as log_file:
    prev_states = [None] * num_agents  # 初始化每个空调的上一个状态
    for episode in range(num_episodes):
        # log_file.write(f"Episode {episode + 1}/{num_episodes}\n")
        rewards = []
        states_list = []
        actions_list = []
        for t in range(max_steps):
            temp_reward = []
            temp_states = []
            temp_actions = []

            for i in range(num_agents):
                # 获取当前状态
                states = get_state_local(data.iloc[t], i + 1)


                # 选择动作
                actions = vdns[i].select_action(states)
                actions[0] = int((actions[0] + 1) / 2 >= 0.5)
               
                # 执行动作，获取下一个状态
                next_states = get_state_local(data.iloc[t + 1], i + 1)

                # 计算奖励
                reward = calculate_reward_local(states, next_states, actions, prev_states[i], best_params)
                
                # 存储经验
                vdns[i].memory.push(states, actions, reward, next_states, 0 if t < max_steps - 1 else 1)

                # 更新网络
                if vdns[i].memory.size() > 1000:
                    vdns[i].update(100)


                temp_reward.append(reward)
                temp_states.extend(states)
                temp_actions.extend(actions)

                # 更新上一个状态
                prev_states[i] = states
            
            log_file.write(f"Step {t + 1}/{max_steps}: Rewards {temp_reward}, States {temp_states}, Actions {temp_actions}\n")

            actions_list.append(temp_actions)
            rewards.append(temp_reward)
            states_list.append(temp_states)

            np.save(f'{filename}actions.npy',actions_list)
            np.save(f'{filename}rewards.npy',rewards)
            np.save(f'{filename}next_states.npy',states_list)
            
            
            # 可视化结果
            if (t+1)%200==0:
                
                actions_plot = np.array(actions_list)

                fig, axs = plt.subplots(2, 4, figsize=(20, 10),dpi=600)  # 注意这里的figsize调整为15x10更合适
               
                # 绘制4个空调的动作
                for i in range(4):
                    # 获取当前空调的状态和温度子图
                    ax_state = axs[0, i]
                    ax_T = axs[1, i]

                    # 绘制状态
                    ax_state.plot(actions_plot[:, i * 2], label=f'AC{i+1} Status')
                    ax_state.set_title(f'AC{i+1} State Over Time')
                    ax_state.set_xlabel('Time Step')
                    ax_state.set_ylabel('Status (On/Off)')
                    
                    # 绘制设定温度
                    ax_T.plot(actions_plot[:, i * 2 + 1], label=f'AC{i+1} Set Temperature')
                    ax_T.set_title(f'AC{i+1} Set Temperature Over Time')
                    ax_T.set_xlabel('Time Step')
                    ax_T.set_ylabel('Temperature (°C)')

                plt.tight_layout()
                fig.savefig(f'{filename}ac_status_and_Teratures.png')
                plt.close()        


                state_plot = np.array(states_list)  # (T,48)
                # 四个空调的附近传感器湿度和温度
                fig_H, axs_H = plt.subplots(1, 4, figsize=(20, 5), dpi=600)
                fig_T, axs_T = plt.subplots(1, 4, figsize=(20, 5), dpi=600)

                for i in range(4):
                    # 提取湿度数据
                    data_H = state_plot[:, 4 + i * 12:11 + i * 12:2]  # (T, 4) 每个空调的4个湿度传感器数据
                    
                    # 提取温度数据
                    data_T = state_plot[:, 3 + i * 12:11 + i * 12:2]  # (T, 4) 每个空调的4个温度传感器数据
                    
                    # 绘制湿度数据
                    ax_H = axs_H[i]
                    for j in range(data_H.shape[1]):
                        ax_H.plot(data_H[:, j],  linestyle='-', label=f'Sensor {j+1}')
                    ax_H.set_title(f'Air Condition {i+1} Humidity')
                    ax_H.set_xlabel('Time Step')
                    ax_H.set_ylabel('Humidity (%)')
                    ax_H.set_ylim([20,80])
                    ax_H.legend()

                    
                    # 绘制温度数据
                    ax_T = axs_T[i]
                    for j in range(data_T.shape[1]):
                        ax_T.plot(data_T[:, j], linestyle='-', label=f'Sensor {j+1}')
                    ax_T.set_title(f'Air Condition {i+1} Temperature')
                    ax_T.set_xlabel('Time Step')
                    ax_T.set_ylabel('Temperature (°C)')
                    ax_T.legend()
                    ax_T.set_ylim([22,32])




                # 调整布局以避免重叠
                plt.tight_layout()
               
                # 保存图像（可选）
                fig_H.savefig(f'{filename}sensors_humidity.png', bbox_inches='tight')
                fig_T.savefig(f'{filename}sensors_temperature.png', bbox_inches='tight')
                plt.close()
                plt.close()

                fig, axs = plt.subplots(1,1, figsize=(3, 2.4),dpi=300)  # 注意这里的figsize调整为15x10更合适
                rewards_plot = np.array(rewards)
                axs.plot(rewards, label='Reward')
                axs.set_title('Rewards Over Time')
                axs.set_xlabel('Time Step')
                axs.set_ylabel('Reward Value')
                plt.tight_layout()
                fig.savefig(f'{filename}reward_status_and_Teratures.png')

                plt.close()

                print(f"Step {t + 1}/{max_steps}: Action {temp_actions}, Reward {temp_reward}, Next State {temp_states} \n")



            # 保存模型
            for i, vdn in enumerate(vdns):
                torch.save(vdn.local_value_net.state_dict(), f'{filename}local_value_net_ac{i + 1}.pth')

print("Training complete!")
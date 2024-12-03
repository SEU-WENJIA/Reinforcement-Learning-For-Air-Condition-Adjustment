import pandas as pd
import numpy as np
import torch
import datetime
import os
from models import TD3
from utils import get_state_global, calculate_reward_global
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data\\data01.csv')


# 参数初始化
best_params = (5.74040941e-02,  2.60904007e-04, -1.78204446e+01,  1.82571830e+00,
  6.81185427e+01,  1.99094691e+02,  9.87365351e+02,  1.00000000e+01, -3.35105392e-04)


# 初始化环境和代理
state_dim = 44  # 状态维度
action_dim = 8  # 动作维度（每个空调的状态和设定温度）
max_action = 1.0  # 最大动作值

agent = TD3(state_dim, action_dim, max_action)
num_episodes = 2
max_steps = len(data) - 1
prev_action = None  # 用于存储上一个动作

# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
actions  = []
rewards = []
next_states = []



filename = f'results\\TD3_{current_time}/'
if not os.path.exists(filename):
    os.mkdir(filename)


# 打开输出文件，文件名加上当前时间
with open(f'{filename}TD3_plus_training_log.txt', 'w') as log_file:
    for episode in range(num_episodes):
        state = get_state_global(data.iloc[0])

        for t in range(max_steps):
            # 选择动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = agent.select_action(state)
            action = np.clip(action + np.random.normal(0, 0.1, size=action.shape), -1, 1)
            
            # 将动作转换为实际的空调状态和设定温度
            action_mapped = np.zeros(8)
            for i in range(4):
                action_mapped[i * 2] = int((action[i * 2] + 1) / 2 >= 0.5) #(action[i * 2] + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]
                set_T = (action[i * 2 + 1] + 1) * 4 + 18  # 将 [-1, 1] 映射到 [18, 28]
                action_mapped[i * 2 + 1] = np.clip(set_T, 18, 28)  # 确保设定温度在 [18, 28] 范围内

            # 执行动作，获取下一个状态
            next_state = get_state_global(data.iloc[t + 1])
            
            # 计算奖励
            reward = calculate_reward_global(state, next_state, action_mapped, prev_action, best_params)
            
            # 存储经验
            agent.memory.push(state, action, next_state, reward, 0 if t < max_steps - 1 else 1)
            
            # 更新网络
            if agent.memory.size() > 1000:
                agent.update(100)
            
            # 输出当前步骤的信息
            log_file.write(f"Step {t + 1}/{max_steps}: Action {action_mapped}, Reward {reward:.10f} Next State {next_state} \n")
            # log_file.write(f"Step {t + 1}/{max_steps}: Action {action_mapped}, Reward {reward}, end\n")
            actions.append(action_mapped)
            rewards.append(reward)
            next_states.append(next_state)
            

            np.save(f'{filename}actions.npy',actions)
            np.save(f'{filename}rewards.npy',rewards)
            np.save(f'{filename}next_states.npy',next_states)

            state = next_state
            prev_action = action_mapped  # 更新上一个动作


            # 可视化结果
            if (t+1)%200==0:
                
                fig, axs = plt.subplots(2, 4, figsize=(16, 8),dpi=50)  # 注意这里的figsize调整为15x10更合适
                actions_plot = np.array(actions)
                # 绘制4个空调的状态
                for i in range(4):
                    # 获取当前空调的状态和温度子图
                    ax_state = axs[0, i]
                    ax_T = axs[1, i]

                    # 绘制状态
                    ax_state.plot(actions_plot[:, i * 2], label=f'AC{i+1} Status')
                    ax_state.set_title(f'AC{i+1} State Over Time')
                    ax_state.set_xlabel('Time Step')
                    ax_state.set_ylabel('Status (On/Off)')
                    ax_state.set_ylim([-0.1,1.1])
                    
                    # 绘制设定温度
                    ax_T.plot(actions_plot[:, i * 2 + 1], label=f'AC{i+1} Set Temperature')
                    ax_T.set_title(f'AC{i+1} Set Temperature Over Time')
                    ax_T.set_xlabel('Time Step')
                    ax_T.set_ylabel('Temperature (°C)')
                    

                plt.tight_layout()
                fig.savefig(f'{filename}ac_status_and_Teratures.png')
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
                print(f"Step {t + 1}/{max_steps}: Action {action_mapped}, Reward {reward:.4f} Next State {next_state} \n")


                next_states_array = np.array(next_states)
                sensor_temperatures = next_states_array[:, 13:40:2]
                sensor_humidities = next_states_array[:, 12:40:2]

                fig_temperature, axs_T = plt.subplots(2, 7, figsize=(28, 8), dpi=50)

                for i in range(14):
                    ax = axs_T[i // 7, i % 7]
                    ax.plot(sensor_temperatures[:,i], label=f'Sensor {i+1} Temperature')
                    ax.set_title(f'Sensor {i+1} Temperature Over Time')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Temperature (°C)')
                    ax.set_ylim([22,32])

                plt.tight_layout()
                fig_temperature.savefig(f'{filename}sensors_temperatures.png')
                plt.close()
               
                


                fig_humidity, axs_H = plt.subplots(2, 7, figsize=(28, 8), dpi=50)
                for i in range(14):
                    ax = axs_H[i // 7, i % 7]
                    ax.plot(sensor_humidities[:,i], label=f'Sensor {i+1} Humidity')
                    ax.set_title(f'Sensor {i+1} Humidity Over Time')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Humidity (%)')
                    ax.set_ylim([20,80])
                    # ax.legend()

                plt.tight_layout()
                fig_humidity.savefig(f'{filename}sensors_humidities.png')
                plt.close()
               



        # 每个 episode 保存模型
        torch.save(agent.actor.state_dict(), f'{filename}actor_model.pth')
        torch.save(agent.critic_1.state_dict(), f'{filename}critic1_model.pth')
        torch.save(agent.critic_2.state_dict(), f'{filename}critic2_model.pth')

print("Training complete!")
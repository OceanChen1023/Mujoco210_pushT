import gym
import random
import torch
import torch.distributions
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pushT_env import PushEnv
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多個 GPU

# 定義 Actor-Critic 網路，共享部分層，然後分別輸出動作均值和狀態價值
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, action_std=0.1):
        super(ActorCritic, self).__init__()
        self.action_std = action_std
        # 共享層
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        # Actor：輸出動作均值
        self.actor = nn.Linear(256, action_dim)
        
        # Critic：輸出狀態價值
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.shared(x)
        action_mean = self.actor(x)
        state_value = self.critic(x)
        return action_mean, state_value

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_rewards.insert(0, G)
    return torch.tensor(discounted_rewards, dtype=torch.float32)

# Actor-Critic 訓練函數
def train_actor_critic(num_episodes=1000, gamma=0.99, value_coef=0.5, entropy_coef=0.01, plot_interval=100):
    # 用於紀錄每個 episode 的 loss 與 reward
    loss_history = []
    reward_history = []
    
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize = (12, 5))
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        done = False
        total_reward = 0
        
        while not done:
            # 從 Actor-Critic 網路取得動作均值及狀態價值
            action_mean, state_value = ac_net(state)
            # 以固定標準差建立正態分布
            dist = torch.distributions.Normal(action_mean, ac_net.action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()
            
            next_state, reward, done, _ = env.step(action.detach().numpy())
            
            log_probs.append(log_prob)
            values.append(state_value.squeeze())
            rewards.append(reward)
            entropies.append(entropy)
            
            total_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32)
        
        # 計算折扣回報並標準化
        returns = compute_discounted_rewards(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        
        # 計算 advantage：回報與估計值之差
        advantages = returns - values
        
        # Actor 損失：用負的 log probability 乘以 advantage
        actor_loss = (-log_probs * advantages.detach()).mean()
        # Critic 損失：均方誤差
        critic_loss = advantages.pow(2).mean()
        # 熵損失：鼓勵探索
        entropy_loss = -entropies.mean()
        
        total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        reward_history.append(total_reward)
        
        if (episode+1) % plot_interval == 0:
            ax[0].clear()
            ax[0].plot(loss_history, label="Total Loss")
            ax[0].set_title("Loss over Episodes")
            ax[0].set_xlabel("Episode")
            ax[0].set_ylabel("Loss")
            ax[0].legend()
            
            ax[1].clear()
            ax[1].plot(reward_history, label="Total Reward", color='orange')
            ax[1].set_title("Reward over Episodes")
            ax[1].set_xlabel("Episode")
            ax[1].set_ylabel("Reward")
            ax[1].legend()
            
            plt.pause(0.001)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig("/home/iris/TINA/tinafu/diffusion/diffusion_policy/mujoco/src/mocap_test/mocap_test/plot/Training_Metrics_29_obs_new_reward_and_newGoal2.png")
            
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Loss: {total_loss.item():.4f}")
    
    torch.save(ac_net.state_dict(), "/home/iris/TINA/tinafu/diffusion/diffusion_policy/mujoco/src/mocap_test/mocap_test/models/ac_net_weights_new_reward_and_newGoal.pth")
    plt.ioff()
    plt.show()
    
env = PushEnv("/home/iris/TINA/tinafu/diffusion/diffusion_policy/mujoco/src/mocap_test/mocap_test/UR5_pole.xml")  # 加載 PushT 環境
env.reset()
input_dim = env.observation_space.shape[0]  # 觀察維度（例如9維）
action_dim = env.action_space.shape[0]       # 動作維度（例如3維）

ac_net = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(ac_net.parameters(), lr=1e-4)

train_actor_critic(num_episodes = 30000)

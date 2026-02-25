"""
DQN (Deep Q-Network) baseline for JOPACS problem - Parallel Training Version
Trains all scenarios in parallel using multiprocessing
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(root_dir)

from settings import scenarios, T, I_0, c_a, c_h, gamma, max_acquisition
from mutils.env import LeasingEnv


def mlp(input_dim, output_dim, hidden_dims=[256, 256], activation=nn.ReLU):
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
            layers.append(nn.LayerNorm(dims[i + 1]))
    
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.network = mlp(state_dim, n_actions, hidden_dims=[512, 256])
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


def state_to_features(state, env):
    clock, T, inventory_now, inventory_all, arrival_his, service_his, holding = state
    
    arr_mean = [np.mean(arrival_his[p]) if arrival_his[p] else 0 for p in env.p_list]
    ser_mean = [np.mean(service_his[p]) if service_his[p] else 0 for p in env.p_list]
    holding_list = [holding[p] for p in env.p_list]
    
    time_ratio = clock / T if T > 0 else 0
    inventory_ratio = inventory_now / inventory_all if inventory_all > 0 else 0
    
    obs = np.array([
        time_ratio,
        inventory_ratio,
        np.log(inventory_now + 1),
    ] + arr_mean + ser_mean + holding_list, dtype=np.float32)
    
    return obs


class DQNAgent:
    def __init__(self, state_dim, n_a, n_p, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config is None:
            config = {}
        

        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 50000)
        self.target_update_freq = config.get('target_update_freq', 500)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 20000)
        self.reward_scale = config.get('reward_scale', 0.01)
        self.learning_starts = config.get('learning_starts', 500)
        
        self.state_dim = state_dim
        self.n_a = n_a
        self.n_p = n_p
        self.n_actions = n_a * n_p
        

        self.q_network = QNetwork(state_dim, self.n_actions).to(self.device)
        self.target_network = QNetwork(state_dim, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        

        self.episode_rewards = deque(maxlen=200)
        self.total_steps = 0
    
    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-1.0 * self.total_steps / self.epsilon_decay)
    
    def select_action(self, state, env, training=True):
        epsilon = self.get_epsilon() if training else 0.0
        
        if training and random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
        
        a = action // self.n_p
        p_idx = action % self.n_p
        return action, a, env.p_list[p_idx]
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, env_fn, max_episodes=500, verbose=False):
        best_avg_reward = float('-inf')
        patience = 80
        no_improve_count = 0
        
        for episode in range(max_episodes):
            env = env_fn()
            state = env.reset()[1]
            state_features = state_to_features(state, env)
            
            episode_reward = 0
            
            while True:
                action, a, p = self.select_action(state_features, env, training=True)
                done, next_state, reward = env.step((a, p))
                next_state_features = state_to_features(next_state, env)
                
                scaled_reward = reward * self.reward_scale
                self.replay_buffer.push(state_features, action, scaled_reward, 
                                        next_state_features, float(done))
                
                episode_reward += scaled_reward
                self.total_steps += 1
                
                if self.total_steps > self.learning_starts:
                    self.update()
                    if self.total_steps % self.target_update_freq == 0:
                        self.update_target_network()
                
                state = next_state
                state_features = next_state_features
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            
            if len(self.episode_rewards) >= 30:
                recent_avg = np.mean(list(self.episode_rewards)[-30:])
                if recent_avg > best_avg_reward:
                    best_avg_reward = recent_avg
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if verbose and episode % 50 == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                print(f"Episode {episode:4d} | Reward: {episode_reward:7.2f} | Avg10: {avg_reward:7.2f}")
            
            if no_improve_count >= patience:
                break
        
        return best_avg_reward
    
    def test_model(self, env_fn, num_tests=10):
        test_rewards = []
        self.q_network.eval()
        
        for _ in range(num_tests):
            env = env_fn()
            state = env.reset()[1]
            state_features = state_to_features(state, env)
            
            total_reward = 0
            
            while True:
                _, a, p = self.select_action(state_features, env, training=False)
                done, next_state, reward = env.step((a, p))
                total_reward += reward
                
                state = next_state
                state_features = state_to_features(next_state, env)
                
                if done:
                    break
            
            test_rewards.append(total_reward)
        
        return test_rewards


def create_env_fn(scenario_config):
    def env_fn():
        return LeasingEnv(
            lambda_dict=scenario_config["lambda_dict"],
            mu_dict=scenario_config["mu_dict"],
            p_list=scenario_config["price"],
            a_max=max_acquisition,
            T=T,
            I0=I_0,
            c_a=c_a,
            c_h=c_h,
            chance_cost=False
        )
    return env_fn


def calculate_state_dim():
    base_features = 3
    price_list = scenarios[1]["price"]
    price_features = len(price_list) * 3
    return base_features + price_features


def train_single_scenario(args):
    train_scenario_id, state_dim, n_a, n_p, dqn_config, max_episodes = args
    
    np.random.seed(42 + train_scenario_id)
    torch.manual_seed(42 + train_scenario_id)
    random.seed(42 + train_scenario_id)
    
    print(f"[场景 {train_scenario_id}] 开始训练...")
    
    train_config = scenarios[train_scenario_id]
    train_env_fn = create_env_fn(train_config)
    
    agent = DQNAgent(state_dim, n_a, n_p, dqn_config)
    agent.train(train_env_fn, max_episodes=max_episodes, verbose=False)
    

    results = {}
    for test_scenario_id in scenarios.keys():
        test_config = scenarios[test_scenario_id]
        test_env_fn = create_env_fn(test_config)
        
        test_rewards = agent.test_model(test_env_fn, num_tests=10)
        results[test_scenario_id] = {
            'mean': np.mean(test_rewards),
            'std': np.std(test_rewards)
        }
        print(f"[场景 {train_scenario_id}] 测试场景 {test_scenario_id}: 均值={np.mean(test_rewards):.2f}")
    
    print(f"[场景 {train_scenario_id}] 训练完成!")
    return train_scenario_id, results


if __name__ == "__main__":
    print("开始DQN并行实验")
    print(f"实验时间: {datetime.now()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CPU核心数: {cpu_count()}")
    

    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    

    scenario_ids = list(scenarios.keys())
    num_scenarios = len(scenario_ids)
    max_train_episodes = 2000
    

    state_dim = calculate_state_dim()
    n_a = max_acquisition + 1
    n_p = len(scenarios[1]["price"])
    
    print(f"状态维度: {state_dim}")
    print(f"动作空间: 补货数量={n_a}, 价格数量={n_p}")
    

    dqn_config = {
        'gamma': gamma,
        'learning_rate': 1e-4,
        'batch_size': 64,
        'buffer_size': 50000,
        'target_update_freq': 500,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 20000,
        'reward_scale': 0.1,
        'learning_starts': 500
    }
    

    train_args = [
        (scenario_id, state_dim, n_a, n_p, dqn_config, max_train_episodes)
        for scenario_id in scenario_ids
    ]
    

    print(f"\n开始并行训练 {num_scenarios} 个场景...")
    start_time = datetime.now()
    

    num_workers = min(num_scenarios, cpu_count())
    print(f"使用 {num_workers} 个工作进程")
    
    with Pool(num_workers) as pool:
        results_list = pool.map(train_single_scenario, train_args)
    
    end_time = datetime.now()
    print(f"\n总训练时间: {end_time - start_time}")
    

    mean_results = np.zeros((num_scenarios, num_scenarios))
    std_results = np.zeros((num_scenarios, num_scenarios))
    
    scenario_index = {sid: idx for idx, sid in enumerate(scenario_ids)}
    for train_scenario_id, results in results_list:
        for test_scenario_id, metrics in results.items():
            train_idx = scenario_index[train_scenario_id]
            test_idx = scenario_index[test_scenario_id]
            mean_results[train_idx, test_idx] = metrics['mean']
            std_results[train_idx, test_idx] = metrics['std']
    

    mean_df = pd.DataFrame(mean_results,
                          index=[f"Train_Scenario_{i}" for i in scenario_ids],
                          columns=[f"Test_Scenario_{i}" for i in scenario_ids])
    mean_csv_path = os.path.join(results_dir, "dqn_results_mean.csv")
    mean_df.to_csv(mean_csv_path)
    print(f"\n均值结果已保存到: {mean_csv_path}")
    
    std_df = pd.DataFrame(std_results,
                         index=[f"Train_Scenario_{i}" for i in scenario_ids],
                         columns=[f"Test_Scenario_{i}" for i in scenario_ids])
    std_csv_path = os.path.join(results_dir, "dqn_results_std.csv")
    std_df.to_csv(std_csv_path)
    print(f"标准差结果已保存到: {std_csv_path}")
    
    print(f"\n{'='*60}")
    print("DQN实验结果总结:")
    print("\n均值结果:")
    print(mean_df.round(2))
    

    ppo_mean_path = os.path.join(results_dir, "drl_results_mean.csv")
    if os.path.exists(ppo_mean_path):
        ppo_mean_df = pd.read_csv(ppo_mean_path, index_col=0)
        print("\n\nDQN vs PPO 对比 (对角线元素 - On-Policy Performance):")
        for i, scenario_id in enumerate(scenario_ids):
            dqn_val = mean_results[i, i]
            ppo_val = ppo_mean_df.iloc[i, i]
            diff_pct = (ppo_val - dqn_val) / ppo_val * 100 if ppo_val != 0 else 0
            print(f"  场景 {scenario_id}: DQN={dqn_val:.2f}, PPO={ppo_val:.2f}, PPO优势={diff_pct:.1f}%")
    
    print(f"\n实验完成!")

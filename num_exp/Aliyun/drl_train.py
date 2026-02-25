import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from settings import scenarios, T, I_0, c_a, c_h, gamma, max_acquisition
from mutils.env import LeasingEnv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class VecEnv:
    def __init__(self, env_fn, num_envs):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        
    def reset(self):
        return [env.reset()[1] for env in self.envs]
        
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        dones, next_states, rewards = zip(*results)
        return list(dones), list(next_states), list(rewards)


def mlp(input_dim, output_dim, hidden_dims=[256, 256], activation=nn.ReLU):
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
            layers.append(nn.LayerNorm(dims[i + 1]))
    
    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_a, n_p):
        super().__init__()

        self.actor = mlp(state_dim, n_a * n_p, hidden_dims=[512, 256])
        self.critic = mlp(state_dim, 1, hidden_dims=[512, 256])
        self.n_a = n_a
        self.n_p = n_p
        self.apply(self._init_weights)

        print(f"Actor network: {self.actor}")
        print(f"Critic network: {self.critic}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        actor_logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return actor_logits, value
        
    def get_action_and_value(self, x):
        actor_logits, value = self.forward(x)
        probs = torch.softmax(actor_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        a = action // self.n_p
        p = action % self.n_p
        
        return action, dist.log_prob(action), dist.entropy(), value, a, p


def states_to_tensor(states, env):
    obs_list = []
    for state in states:
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
        
        obs_list.append(obs)
        
    obs_arr = np.stack(obs_list)
    return torch.tensor(obs_arr, dtype=torch.float32, device=DEVICE)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = values[t+1]
            
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        
    returns = advantages + values
    return advantages, returns


class PPOAgent:
    def __init__(self, state_dim, n_a, n_p, config=None):

        self.device = DEVICE
        

        if config is None:
            config = {}
        
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.learning_rate = config.get('learning_rate', 2.5e-4)
        self.update_epochs = config.get('update_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        self.num_envs = config.get('num_envs', 8)
        self.reward_scale = config.get('reward_scale', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.target_kl = config.get('target_kl', 0.01)
        

        self.state_dim = state_dim
        self.n_a = n_a
        self.n_p = n_p
        

        self.model = ActorCritic(state_dim, n_a, n_p).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=800
        )
        

        self.reward_normalizer = RunningMeanStd()
        

        self.episode_rewards = deque(maxlen=1000)
        self.loss_history = []
        
        print(f"PPO Agent initialized on device: {self.device}")
        print(f"State dim: {state_dim}, Action num: {n_a}, Price num: {n_p}")
    
    def get_action(self, state, env):
        s_tensor = states_to_tensor([state], env)
        with torch.no_grad():
            action, log_prob, entropy, value, a, p = self.model.get_action_and_value(s_tensor)
        
        a_val = int(a.cpu().numpy()[0])
        p_val = int(p.cpu().numpy()[0])
        return a_val, env.p_list[p_val]
    
    def train(self, env_fn, max_episodes=2000, rollout_len=None, save_path='models', verbose=True):
        dummy_env = env_fn()
        if rollout_len is None:
            rollout_len = min(128, dummy_env.T // 4)
        
        vec_env = VecEnv(env_fn, self.num_envs)
        self.model.train()
        

        best_avg_reward = float('-inf')
        patience = 150
        no_improve_count = 0
        
        if verbose:
            print(f"开始训练...")
            print(f"Rollout长度: {rollout_len}, 并行环境数: {self.num_envs}")
        
        for episode in range(max_episodes):
            states = vec_env.reset()
            ep_rewards = np.zeros(self.num_envs)
            

            all_states, all_actions, all_log_probs, all_values = [], [], [], []
            all_rewards, all_dones = [], []
            

            for t in range(rollout_len):
                s_tensor = states_to_tensor(states, dummy_env)
                
                with torch.no_grad():
                    action, log_prob, entropy, value, a, p = self.model.get_action_and_value(s_tensor)
                

                action_cpu = action.cpu().numpy()
                a_cpu = (action_cpu // self.n_p).tolist()
                p_cpu = (action_cpu % self.n_p).tolist()
                actions = [(a_, dummy_env.p_list[p_]) for a_, p_ in zip(a_cpu, p_cpu)]
                
                dones, next_states, rewards = vec_env.step(actions)
                

                rewards = np.array(rewards) * self.reward_scale
                

                all_states.append(s_tensor)
                all_actions.append(action)
                all_log_probs.append(log_prob)
                all_values.append(value)
                all_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self.device))
                all_dones.append(torch.tensor(dones, dtype=torch.float32, device=self.device))
                
                ep_rewards += rewards
                states = next_states
            

            with torch.no_grad():
                next_value = self.model.forward(states_to_tensor(states, dummy_env))[1]
            

            all_states = torch.stack(all_states)
            all_actions = torch.stack(all_actions)
            all_log_probs = torch.stack(all_log_probs)
            all_values = torch.stack(all_values)
            all_rewards = torch.stack(all_rewards)
            all_dones = torch.stack(all_dones)
            

            advantages, returns = [], []
            for env_idx in range(self.num_envs):
                adv, ret = compute_gae(
                    all_rewards[:, env_idx].cpu().numpy(),
                    all_values[:, env_idx].cpu().numpy(),
                    all_dones[:, env_idx].cpu().numpy(),
                    next_value[env_idx].cpu().numpy(),
                    self.gamma, self.lam
                )
                advantages.append(torch.tensor(adv, dtype=torch.float32, device=self.device))
                returns.append(torch.tensor(ret, dtype=torch.float32, device=self.device))
            
            advantages = torch.stack(advantages, dim=1)
            returns = torch.stack(returns, dim=1)
            

            flat_states = all_states.reshape(-1, self.state_dim)
            flat_actions = all_actions.reshape(-1)
            flat_log_probs = all_log_probs.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            

            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            

            self.reward_normalizer.update(all_rewards.cpu().numpy().flatten())
            

            episode_losses = []
            approx_kl_divs = []
            
            for epoch in range(self.update_epochs):

                indices = torch.randperm(flat_states.size(0), device=self.device)
                
                for start in range(0, flat_states.size(0), self.batch_size):
                    end = min(start + self.batch_size, flat_states.size(0))
                    batch_indices = indices[start:end]
                    
                    batch_states = flat_states[batch_indices]
                    batch_actions = flat_actions[batch_indices]
                    batch_old_log_probs = flat_log_probs[batch_indices]
                    batch_returns = flat_returns[batch_indices]
                    batch_advantages = flat_advantages[batch_indices]
                    

                    logits, values = self.model(batch_states)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                    

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self.value_loss_coef * (batch_returns - values).pow(2).mean()
                    entropy_loss = -self.entropy_coef * entropy.mean()
                    
                    total_loss = policy_loss + value_loss + entropy_loss
                    

                    with torch.no_grad():
                        approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                        approx_kl_divs.append(approx_kl)
                    
                    episode_losses.append(total_loss.item())
                    

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                

                if np.mean(approx_kl_divs) > self.target_kl:
                    break
            

            self.scheduler.step()
            

            if episode_losses:
                self.loss_history.append(np.mean(episode_losses))
            
            mean_reward = np.mean(ep_rewards)
            self.episode_rewards.append(mean_reward)
            

            if len(self.episode_rewards) >= 50:
                recent_avg = np.mean(list(self.episode_rewards)[-50:])
                if recent_avg > best_avg_reward:
                    best_avg_reward = recent_avg
                    no_improve_count = 0

                    try:
                        os.makedirs(save_path, exist_ok=True)
                        self.save_model(os.path.join(save_path, 'best_model.pth'))
                        if verbose:
                            print(f"新的最佳模型保存! 平均奖励: {recent_avg:.2f}")
                    except Exception as e:
                        if verbose:
                            print(f"保存最佳模型失败: {e}")
                else:
                    no_improve_count += 1
            

            if verbose and episode % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else mean_reward
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Episode {episode:4d} | Reward: {mean_reward:7.2f} | "
                      f"Avg10: {avg_reward:7.2f} | Loss: {self.loss_history[-1] if self.loss_history else 0:.4f} | "
                      f"KL: {np.mean(approx_kl_divs):.4f} | LR: {current_lr:.2e}")
            

            if no_improve_count >= patience:
                if verbose:
                    print(f"早停于第 {episode} 轮，无改善轮数: {no_improve_count}")
                break
        

        try:
            os.makedirs(save_path, exist_ok=True)
            self.save_model(os.path.join(save_path, 'final_model.pth'))
        except Exception as e:
            if verbose:
                print(f"保存最终模型失败: {e}")
        
        final_avg = np.mean(list(self.episode_rewards)[-50:]) if len(self.episode_rewards) >= 50 else np.mean(list(self.episode_rewards))
        if verbose:
            print(f"训练完成! 最终平均奖励: {final_avg:.2f}")
        
        return final_avg
    
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        self.model.eval()
    
    def test_model(self, env_fn, num_tests=5, verbose=True):
        if verbose:
            print(f"开始进行 {num_tests} 次测试...")
        
        test_results = []
        self.model.eval()
        
        for test_id in range(num_tests):
            if verbose:
                print(f"Running test {test_id + 1}/{num_tests}")
            
            env = env_fn()
            state = env.reset()[1]
            test_data = {
                'inventories': [], 
                'actions_a': [], 
                'actions_p': [], 
                'rewards': [], 
                'periods': []
            }
            
            period = 0
            total_reward = 0
            
            while True:
                _, _, inventory_now, _, _, _, _ = state
                test_data['inventories'].append(inventory_now)
                test_data['periods'].append(period)
                

                a, p = self.get_action(state, env)
                test_data['actions_a'].append(a)
                test_data['actions_p'].append(p)
                

                done, next_state, reward = env.step((a, p))
                test_data['rewards'].append(reward)
                total_reward += reward
                
                state = next_state
                period += 1
                
                if done:
                    break
            
            # env.plot_history()
            test_data['total_reward'] = total_reward
            test_results.append(test_data)
            
            if verbose:
                print(f"Test {test_id + 1} completed: Total reward = {total_reward:.2f}, Periods = {period}")
        
        return test_results

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

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    print("开始深度强化学习数值实验")
    print(f"实验时间: {datetime.now()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    

    num_scenarios = len(scenarios)
    num_tests_per_env = 10
    max_train_episodes = 2000
    

    state_dim = calculate_state_dim()
    n_a = max_acquisition + 1
    n_p = len(scenarios[1]["price"])
    
    print(f"状态维度: {state_dim}")
    print(f"动作空间: 补货数量={n_a}, 价格数量={n_p}")
    print(f"总动作空间大小: {n_a * n_p}")
    
    
    

    mean_results = np.zeros((num_scenarios, num_scenarios))
    std_results = np.zeros((num_scenarios, num_scenarios))
    
    print("\n开始训练和测试各个场景的DRL智能体...")
    

    for train_scenario_id in range(1, num_scenarios + 1):
        print(f"\n{'='*60}")
        print(f"训练场景 {train_scenario_id}")
        print(f"{'='*60}")
        

        train_config = scenarios[train_scenario_id]
        train_env_fn = create_env_fn(train_config)
        
        ppo_config = {
        'gamma': gamma,
        'lam': 0.95 if train_scenario_id != 4 else 0.99,
        'clip_eps': 0.2,
        'learning_rate': 2.5e-4,  
        'update_epochs': 4,
        'batch_size': 64,
        'num_envs': 16,  
        'reward_scale': 0.001 if train_scenario_id in [1,2] else 1e-2,  
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'target_kl': 0.01
    }
        
        agent = PPOAgent(state_dim, n_a, n_p, ppo_config)
        

        print(f"开始训练智能体 (场景 {train_scenario_id})...")
        model_save_path = os.path.join(models_dir, f"scenario_{train_scenario_id}")
        os.makedirs(model_save_path, exist_ok=True)
        

        dummy_env = train_env_fn()
        rollout_len =  min(128, dummy_env.T)
        
        try:
            avg_train_reward = agent.train(
                env_fn=train_env_fn,
                max_episodes=max_train_episodes,
                rollout_len=rollout_len,
                save_path=model_save_path,
                verbose=True
            )
            print(f"训练完成! 平均训练奖励: {avg_train_reward:.2f}")
            
            
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
        
        
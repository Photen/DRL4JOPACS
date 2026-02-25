import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import math
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from settings import scenarios, T, I_0, c_a, c_h, gamma, max_acquisition
from mutils.env import LeasingEnv
from mutils.baseline import Clairvoyant
class BTSAgent:
    """
    Batch Thompson Sampling (BTS) 算法智能体
    基于论文 "Online Learning and Pricing for Service Systems with Reusable Resources" 实现
    
    该算法使用贝叶斯推断来学习未知的到达率和服务率参数，
    并使用Thompson采样来平衡探索和利用。
    """
    
    def __init__(self, price_list, T_horizon, I0, c_a, c_h, r_max, a_max, alpha_prior=1.0, beta_prior=1.0, verbose=True):
        """
        初始化BTS智能体
        
        Args:
            price_list: 候选价格列表
            capacity: 系统容量（用于稳定性检查）
            T_horizon: 规划期限
            I0: 初始库存
            c_a: 获取成本
            c_h: 持有成本
            alpha_prior: Gamma分布的形状参数（先验）
            beta_prior: Gamma分布的速率参数（先验）
            verbose: 是否输出详细信息
        """
        self.price_list = sorted(price_list, reverse=True)
        self.P = len(self.price_list)
        self.T_horizon = T_horizon
        self.I0 = I0
        self.c_a = c_a
        self.c_h = c_h
        self.verbose = verbose
        self.epsilon_bar = 0.1
        self.m = 1
        self.r_max = r_max
        self.c = a_max
        self.a_max = a_max
        self.T = T_horizon

        self.epsilon_T = self.epsilon_bar/np.log(T_horizon)
        self.delta = self.epsilon_bar/(2 * (1 + self.r_max/self.c) * np.log(T_horizon))
        print(f"初始化BTS智能体: ε={self.epsilon_bar}, δ={self.delta:.4f}, ε_T={self.epsilon_T:.4f}")
        self.N_W = max(8*np.log(T_horizon), np.log(T_horizon)/(2 * self.delta **2 ))
        print(f"预热阶段观测次数 N_W={self.N_W:.2f}")
        self.tau = (np.log(T_horizon))**2
        self.rho_W = 1 - self.delta * (1 + self.r_max/self.c) / (1+self.delta)

        self.lambda_hat = {p: 0.0 for p in self.price_list}
        self.mu_hat = {p: 0.0 for p in self.price_list}
        

        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        

        self.lambda_alpha = {p: alpha_prior for p in self.price_list}
        self.lambda_beta = {p: beta_prior for p in self.price_list}
        self.mu_alpha = {p: alpha_prior for p in self.price_list}
        self.mu_beta = {p: beta_prior for p in self.price_list}
        

        self.lambda_sample = {p: 0.0 for p in self.price_list}
        self.mu_sample = {p: 0.0 for p in self.price_list}
        
        self.warm_up_completed = False
        self.warm_up_price_index = 0
        self.warm_up_observations = {p: 0 for p in self.price_list}
        

        self.batch_size = 2**self.m
        self.current_batch = 0
        self.periods_in_current_batch = 0
        self.current_price = None
        

        self.learning_prices = set(self.price_list)
        

        self.total_periods = 0
        
        if self.verbose:
            print(f"BTS初始化参数:")
            print(f"  价格列表: {self.price_list}")
            print(f"  批次大小: {self.batch_size}")
            print(f"  先验参数: α={self.alpha_prior}, β={self.beta_prior}")
    
    def update_posterior(self, price, arrival_history, service_history):
        """
        基于观测到的到达和服务历史更新后验分布
        
        Args:
            price: 当前价格
            arrival_history: 到达间隔时间历史
            service_history: 服务时间历史
        """
        if price not in self.price_list:
            return
            


        if arrival_history:
            n_arrivals = len(arrival_history)
            total_time = sum(arrival_history)
            
            self.lambda_alpha[price] = self.alpha_prior + n_arrivals
            self.lambda_beta[price] = self.beta_prior + total_time
            self.lambda_hat[price] = 1/np.mean(arrival_history) if n_arrivals > 0 else 0.0
        


        if service_history:
            n_services = len(service_history)
            total_service_time = sum(service_history)
            self.mu_alpha[price] = self.alpha_prior + n_services
            self.mu_beta[price] = self.beta_prior + total_service_time
            self.mu_hat[price] = 1/np.mean(service_history) if n_services > 0 else 0.0

    
    def update_batch_size(self):
        self.m += 1
        self.batch_size = 2 ** self.m * self.tau


    def sample_parameters(self):
        """
        从后验分布中采样参数
        """
        for price in self.learning_prices:

            self.lambda_sample[price] = np.random.gamma(
                self.lambda_alpha[price], 1.0 / self.lambda_beta[price]
            )
            self.mu_sample[price] = np.random.gamma(
                self.mu_alpha[price], 1.0 / self.mu_beta[price]
            ) 

    
    def prune_unstable_prices(self):
        """
        剪枝导致系统不稳定的价格
        """
        prices_to_remove = []
        
        for price in list(self.learning_prices):
            rho_p = self.lambda_hat[price] / self.mu_hat[price] if self.mu_hat[price] > 0 else float('inf')
            if rho_p > self.rho_W:
                prices_to_remove.append(price)
                if self.verbose:
                    print(f"  剪枝价格 {price} (ρ={rho_p:.4f})")

        

        if len(prices_to_remove) >= len(self.learning_prices):
            if prices_to_remove:
                prices_to_remove = prices_to_remove[:-1]
        
        for price in prices_to_remove:
            self.learning_prices.discard(price)
        
        if self.verbose and prices_to_remove:
            print(f"  剪枝后学习价格集合: {sorted(list(self.learning_prices), reverse=True)}")
    
    def select_price_thompson_sampling(self):
        """
        使用Thompson采样选择价格
        """
        if not self.learning_prices:
            return self.price_list[0]
        

        self.sample_parameters()
        
        best_price = None
        best_profit = -float('inf')
        
        for price in self.learning_prices:
            expected_profit = price * self.lambda_sample[price]/ self.mu_sample[price] 
            if expected_profit > best_profit:
                best_profit = expected_profit
                best_price = price
        
        return best_price if best_price is not None else self.price_list[0]
    
    def get_action(self, period, env_state):
        """
        获取当前周期的动作（获取量和价格）
        
        Args:
            period: 当前周期
            env_state: 环境状态 (clock, T, inventory_now, inventory_all, arrival_his, service_his, holding)
            
        Returns:
            (acquisition, price): 获取量和价格的元组
        """
        clock, T_remaining, inventory_now, inventory_all, arrival_his, service_his, holding = env_state

        self.c = inventory_all
        

        for price in self.price_list:
            if price in arrival_his and price in service_his:
                self.update_posterior(price, arrival_his[price], service_his[price])
        
        

        if not self.warm_up_completed:

            if self.warm_up_price_index < len(self.price_list):
                current_price = self.price_list[self.warm_up_price_index]
                

                if self.warm_up_observations[current_price] >= self.N_W:
                    self.warm_up_price_index += 1
                    if self.warm_up_price_index >= len(self.price_list):
                        self.warm_up_completed = True
                        self.prune_unstable_prices()
                        self.current_batch = 1
                        self.periods_in_current_batch = 0
                        current_price = self.select_price_thompson_sampling()
                        if self.verbose:
                            print(f"\n预热阶段完成，开始学习阶段")
                            print(f"批次 {self.current_batch}")
                    else:
                        current_price = self.price_list[self.warm_up_price_index]
                        
                self.warm_up_observations[current_price] += 1
                
            else:
                current_price = self.price_list[0]
        else:

            if self.periods_in_current_batch == 0:

                current_price = self.select_price_thompson_sampling()
                if self.verbose:
                    print(f"\n批次 {self.current_batch}, 选择价格: {current_price}")
            else:

                current_price = self.current_price
            
            self.periods_in_current_batch += 1
            

            if self.periods_in_current_batch >= self.batch_size:
                self.current_batch += 1
                self.periods_in_current_batch = 0
        
        self.current_price = current_price
        self.total_periods += 1
        if clock == 0:
            acquisition = self.a_max
        else:

            values = (self.lambda_hat, self.mu_hat, self.price_list, self.a_max, self.T, self.I0, self.c_a, self.c_h)
            self.clairvoyant = Clairvoyant(values)

            acquisition = self.clairvoyant.get_acquisition(current_price, inventory_all)
        return acquisition, current_price
    
    def test_model(self, env, num_tests=1, verbose=True):
        """
        测试BTS智能体的性能
        
        Args:
            env: 环境实例
            num_tests: 测试次数
            verbose: 是否输出详细信息
            
        Returns:
            test_results: 测试结果列表
        """
        if verbose:
            print(f"开始进行 {num_tests} 次BTS测试...")
        
        test_results = []
        
        for test_id in range(num_tests):
            if verbose and num_tests > 1:
                print(f"Running test {test_id + 1}/{num_tests}")
            

            self.reset()
            

            state = env.reset()[1]
            test_data = {
                'periods': [],
                'acquisitions': [],
                'prices': [],
                'rewards': [],
                'inventories': [],
                'batch_info': [],
                'lambda_estimates': [],
                'mu_estimates': []
            }
            
            period = 0
            total_reward = 0
            
            while period < self.T_horizon:

                clock, T_remaining, inventory_now, inventory_all, arrival_his, service_his, holding = state
                test_data['periods'].append(period)
                test_data['inventories'].append(inventory_now)
                

                acquisition, selected_price = self.get_action(period, state)
                test_data['acquisitions'].append(acquisition)
                test_data['prices'].append(selected_price)
                

                if self.warm_up_completed:
                    batch_info = f"B{self.current_batch}"
                else:
                    batch_info = f"W{self.warm_up_price_index}"
                test_data['batch_info'].append(batch_info)
                

                if selected_price in self.lambda_alpha:
                    lambda_est = self.lambda_alpha[selected_price] / self.lambda_beta[selected_price]
                    mu_est = self.mu_alpha[selected_price] / self.mu_beta[selected_price]
                    test_data['lambda_estimates'].append(lambda_est)
                    test_data['mu_estimates'].append(mu_est)
                else:
                    test_data['lambda_estimates'].append(0.0)
                    test_data['mu_estimates'].append(0.0)
                

                done, next_state, reward = env.step((acquisition, selected_price))
                test_data['rewards'].append(reward)
                total_reward += reward
                
                state = next_state
                period += 1
                
                if done or period >= self.T_horizon:
                    break
            


            
            test_data['total_reward'] = total_reward # type: ignore
            test_results.append(test_data)
            
            if verbose and num_tests > 1:
                print(f"Test {test_id + 1} completed: Total reward = {total_reward:.2f}")

            # env.plot_action()
        
        return test_results
    
    def reset(self):
        self.lambda_alpha = {p: self.alpha_prior for p in self.price_list}
        self.lambda_beta = {p: self.beta_prior for p in self.price_list}
        self.mu_alpha = {p: self.alpha_prior for p in self.price_list}
        self.mu_beta = {p: self.beta_prior for p in self.price_list}
        

        self.lambda_sample = {p: 0.0 for p in self.price_list}
        self.mu_sample = {p: 0.0 for p in self.price_list}
        

        self.learning_prices = set(self.price_list)
        self.warm_up_completed = False
        self.warm_up_price_index = 0
        self.warm_up_observations = {p: 0 for p in self.price_list}
        self.current_batch = 0
        self.periods_in_current_batch = 0
        self.current_price = None
        self.total_periods = 0

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

if __name__ == "__main__":
    np.random.seed(42)
    print("开始BTS算法数值实验")
    print(f"实验时间: {datetime.now()}")
    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    

    num_scenarios = len(scenarios)
    num_tests_per_scenario = 10
    
    print(f"场景数量: {num_scenarios}")
    print(f"每个场景测试次数: {num_tests_per_scenario}")
    

    results = []
    
    print(f"\n{'='*60}")
    print("开始各个场景的BTS测试...")
    

    for scenario_id in scenarios.keys():
        print(f"\n{'='*40}")
        print(f"测试场景 {scenario_id}")
        print(f"{'='*40}")
        

        scenario_config = scenarios[scenario_id]
        
        try:

            bts_agent = BTSAgent(
                price_list=scenario_config["price"],
                T_horizon=T,
                I0=I_0,
                c_a=c_a,
                c_h=c_h,
                alpha_prior=1.0,
                beta_prior=1.0,
                r_max = 200,
                a_max=max_acquisition,
                verbose=False
            )
            

            env_fn = create_env_fn(scenario_config)
            

            scenario_rewards = []
            
            print(f"开始进行 {num_tests_per_scenario} 次测试...")
            
            for test_run in range(num_tests_per_scenario):
                try:

                    test_env = env_fn()
                    

                    test_results = bts_agent.test_model(test_env, num_tests=1, verbose=False)
                    test_reward = test_results[0]['total_reward']
                    scenario_rewards.append(test_reward)
                    
                    if (test_run + 1) % 5 == 0 or test_run == 0:
                        print(f"  完成测试 {test_run + 1}/{num_tests_per_scenario}, "
                              f"当前奖励: {test_reward:.2f}")
                        
                except Exception as e:
                    print(f"  测试运行 {test_run + 1} 出现错误: {e}")
                    continue
            
            if scenario_rewards:
                mean_reward = np.mean(scenario_rewards)
                std_reward = np.std(scenario_rewards)
                
                results.append({
                    'scenario_id': scenario_id,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'num_tests': len(scenario_rewards)
                })
                
                print(f"场景 {scenario_id} 测试完成:")
                print(f"  平均奖励: {mean_reward:.2f}")
                print(f"  标准差: {std_reward:.2f}")
                print(f"  有效测试次数: {len(scenario_rewards)}")
            else:
                print(f"场景 {scenario_id}: 所有测试都失败了")
                
        except Exception as e:
            print(f"场景 {scenario_id} 初始化失败: {e}")
            continue
    

    print(f"\n{'='*60}")
    print("保存实验结果...")
    
    if results:

        results_df = pd.DataFrame(results)
        

        detailed_csv_path = os.path.join(results_dir, "bts_detailed_results.csv")
        results_df.to_csv(detailed_csv_path, index=False)
        print(f"详细结果已保存到: {detailed_csv_path}")
        

        mean_results = np.zeros((num_scenarios, 1))
        std_results = np.zeros((num_scenarios, 1))
        
        for i, result in enumerate(results):
            mean_results[i, 0] = result['mean_reward']
            std_results[i, 0] = result['std_reward']
        

        mean_df = pd.DataFrame(mean_results, 
                              index=[f"Scenario_{i+1}" for i in range(num_scenarios)],
                              columns=["BTS"])
        mean_csv_path = os.path.join(results_dir, "bts_results_mean.csv")
        mean_df.to_csv(mean_csv_path)
        print(f"均值结果已保存到: {mean_csv_path}")
        

        std_df = pd.DataFrame(std_results,
                             index=[f"Scenario_{i+1}" for i in range(num_scenarios)], 
                             columns=["BTS"])
        std_csv_path = os.path.join(results_dir, "bts_results_std.csv")
        std_df.to_csv(std_csv_path)
        print(f"标准差结果已保存到: {std_csv_path}")
        

        print(f"\n{'='*60}")
        print("BTS实验结果总结:")
        print("\n均值结果:")
        print(mean_df.round(2))
        print("\n标准差结果:")
        print(std_df.round(2))
        
    else:
        print("没有获得有效的实验结果!")
    
    print(f"\n实验完成! 总耗时: {datetime.now()}")
    print(f"结果文件保存在 {results_dir} 目录中")

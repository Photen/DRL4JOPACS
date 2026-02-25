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


class BUCBAgent:
    """
    Batch Upper Confidence Bound (BUCB) 算法智能体
    基于论文 "Online Learning and Pricing for Service Systems with Reusable Resources" 实现
    """

    def __init__(self, price_list, inventory_all, r_max):
        """
        初始化BUCB智能体
        
        Args:
            price_list: 可选的价格列表
        """
        self.price_list = sorted(price_list, reverse=True)
        self.P = len(self.price_list)
        self.T_horizon = T
        self.I0 = I_0
        self.c_a = c_a
        self.c_h = c_h
        self.epsilon_bar = 0.1
        self.verbose = True
        self.a_max = max_acquisition
        self.T = T
        self.r_max = r_max
        self.inventory_all = inventory_all
        

        self.epsilon_T = self.epsilon_bar / math.log(T)
        self.delta = self.epsilon_bar / (2 * (1 + r_max / self.inventory_all) * math.log(T))
        print("初始化BUCB智能体参数:")
        print(f"  epsilon_T: {self.epsilon_T:.4f}")
        print(f"  delta: {self.delta:.4f}")

        self.N_W = max(8 * np.log(T), np.log(T)/ (2*self.delta ** 2))
        print(f"  N_W: {self.N_W:.0f} (预热观测数)")
        self.rho_W = 1 - self.delta * (1 + r_max / self.inventory_all) / (1 + self.delta)
        self.tau = (np.log(T))**2
        

        self.m = 1
        

        self.arrival_times = {p: [] for p in self.price_list}
        self.service_times = {p: [] for p in self.price_list}
        self.n_arrivals = {p: 0 for p in self.price_list}
        self.n_services = {p: 0 for p in self.price_list}
        self.total_periods = {p: 0 for p in self.price_list}
        

        self.lambda_hat = {p: 0 for p in self.price_list}
        self.mu_hat = {p: 0 for p in self.price_list}

        

        self.learning_prices = set(self.price_list)
        

        self.current_batch = 0
        self.periods_in_current_batch = 0
        self.current_batch_size = 0
        self.current_price = None
        self.warm_up_completed = False
        self.warm_up_price_index = 0
        self.warm_up_observations = {p: 0 for p in self.price_list}
        

        self.upper_confidence_bounds = {p: float('inf') for p in self.price_list}

        if self.verbose:
            print(f"BUCB初始化参数:")
            print(f"  价格列表: {self.price_list}")
            print(f"  候选价格数: {self.P}")
            print(f"  规划期限: {self.T_horizon}")
            print(f"  资源容量: {self.r_max}")
            print(f"  预热观测数: {self.N_W:.0f}")
            print(f"  批次数: {self.m}")
            print(f"  tau: {self.tau:.0f}")
            print(f"  epsilon_T: {self.epsilon_T:.4f}")
    
    def warm_up_phase(self, price, arrival_time, service_time):
        self.warm_up_observations[price] += 1
        if arrival_time is not None:
            self.arrival_times[price].append(arrival_time)
        if service_time is not None:
            self.service_times[price].append(service_time)



        all_enough = all(self.warm_up_observations[p] >= self.N_W for p in self.price_list)
        if all_enough and not self.warm_up_completed:

            self.prune_unstable_prices()

    def get_current_batch_size(self, remain_periods):
        batch_size = min([self.tau * (2 ** self.m), remain_periods])
        self.m += 1
        return batch_size
    
    def update_empirical_estimates(self, price, arrival_time, service_time):
        if arrival_time is not None:
            self.arrival_times[price] += arrival_time
            self.n_arrivals[price] += len(arrival_time)
            
        if service_time is not None:
            self.service_times[price] += service_time
            self.n_services[price] += len(service_time)
        

        if self.arrival_times[price]:

            self.lambda_hat[price] = 1/np.mean(self.arrival_times[price]) if len(self.arrival_times[price]) > 0 else 10 # type: ignore
        
        if self.service_times[price]:

            self.mu_hat[price] = 1/np.mean(self.service_times[price]) if len(self.service_times[price]) > 0 else 10

    def estimate_utilization(self, price):
        if self.mu_hat[price] <= 0:
            return float('inf')
        return self.lambda_hat[price] / self.mu_hat[price]

    
    def update_confidence_bounds(self):
        for price in self.learning_prices:
            self.upper_confidence_bounds[price] = (1/self.mu_hat[price] * self.lambda_hat[price] + 4 * (self.r_max + 1/self.mu_hat[price] * self.lambda_hat[price]) * np.sqrt(2 * np.log(self.T)/ len(self.service_times[price]))) * price
        
    
    def prune_unstable_prices(self):
        prices_to_remove = []
        for price in self.learning_prices:
            rho_est = self.estimate_utilization(price)
            if rho_est > self.rho_W:
                prices_to_remove.append(price)
                if self.verbose:
                    print(f"  剪枝价格 {price}, 估计利用率: {rho_est:.4f} > {self.rho_W:.4f}")
        

        if len(prices_to_remove) >= len(self.learning_prices):
            if self.verbose:
                print(f"  所有价格都将被剪枝，保留利用率最小的价格")

            min_utilization = float('inf')
            price_to_keep = None
            for price in self.learning_prices:
                rho = self.estimate_utilization(price)
                if rho < min_utilization:
                    min_utilization = rho
                    price_to_keep = price
            

            if price_to_keep in prices_to_remove:
                prices_to_remove.remove(price_to_keep)
        

        for price in prices_to_remove:
            self.learning_prices.discard(price)
        
        if self.verbose and prices_to_remove:
            print(f"  剪枝后学习价格集合: {sorted(list(self.learning_prices), reverse=True)}")
    
    def select_price_bucb(self):
        best_price = None
        best_ucb = -float('inf')
        
        for price in self.learning_prices:
            ucb = self.upper_confidence_bounds[price]
            if ucb > best_ucb:
                best_ucb = ucb
                best_price = price
        
        if self.verbose:
            print(f"    UCB值: {', '.join([f'{p}:{self.upper_confidence_bounds[p]:.2f}' for p in sorted(self.learning_prices, reverse=True)])}")
            print(f"    选择价格: {best_price} (UCB: {best_ucb:.2f})")
        
        return best_price
    
    def select_price_ucb(self):
        """
        学习阶段根据上置信界选价格
        """
        best_price = None
        best_ucb = float('-inf')
        for p in self.learning_prices:
            ucb_value = self.upper_confidence_bounds[p]
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_price = p
        
        if self.verbose:
            print(f"    UCB值: {', '.join([f'{p}:{self.upper_confidence_bounds[p]:.2f}' for p in sorted(self.learning_prices, reverse=True)])}")
            print(f"    选择价格: {best_price} (UCB: {best_ucb:.2f})")
        
        return best_price if best_price is not None else self.price_list[0]

    
    def get_action(self, period, env_state=None):
        """
        获取当前周期的动作（价格）
        
        Args:
            period: 当前周期
            env_state: 环境状态（可选）
            
        Returns:
            price: 选择的价格
        """
        self.inventory_all = env_state[3] # type: ignore # type: ignore
        if not self.warm_up_completed:

            if self.warm_up_price_index < len(self.price_list):
                current_price = self.price_list[self.warm_up_price_index]
                

                if self.warm_up_observations[current_price] >= self.N_W:
                    self.warm_up_price_index += 1
                    if self.warm_up_price_index < len(self.price_list):
                        current_price = self.price_list[self.warm_up_price_index]
                    else:

                        self.warm_up_completed = True
                        self.prune_unstable_prices()
                        self.current_batch = 1
                        self.periods_in_current_batch = 0
                        self.current_batch_size = self.get_current_batch_size(self.T - period)

                        self.update_confidence_bounds()
                        current_price = self.select_price_bucb()
                        if self.verbose:
                            print(f"\n预热阶段完成，开始学习阶段")
                            print(f"批次 {self.current_batch}, 批次大小: {self.current_batch_size}")
                
                self.current_price = current_price
                return current_price
            else:

                self.current_price = self.price_list[0]
                return self.current_price
        else:

            if self.periods_in_current_batch == 0:

                self.update_confidence_bounds()
                self.current_price = self.select_price_bucb()
                if self.verbose:
                    print(f"\n批次 {self.current_batch}, 批次大小: {self.current_batch_size}, 选择价格: {self.current_price}")
            
            self.periods_in_current_batch += 1
            

            if self.periods_in_current_batch >= self.current_batch_size:
                self.current_batch += 1
                self.periods_in_current_batch = 0
                self.current_batch_size = self.get_current_batch_size(self.T - period)
            
            return self.current_price
    
    def update_observations(self, price, env_state):
        """
        更新观测数据（基于环境反馈）
        
        Args:
            price: 当前价格
            env_state: 环境状态
        """

        arrival_time = env_state[4][price]
        service_time = env_state[5][price]
        self.update_empirical_estimates(price, arrival_time=arrival_time, service_time=service_time)
        
        if not self.warm_up_completed:
            if price in self.warm_up_observations:
                if isinstance(arrival_time, list):
                    self.warm_up_observations[price] += len(arrival_time)
                else:
                    self.warm_up_observations[price] += 1
    
    def test_model(self, env, num_tests=10, verbose=True):
        """
        测试BUCB智能体的性能
        
        Args:
            env: 环境实例
            num_tests: 测试次数
            verbose: 是否输出详细信息
            
        Returns:
            test_results: 测试结果列表
        """
        self.verbose = verbose
        if verbose:
            print(f"开始进行 {num_tests} 次BUCB测试...")
        
        test_results = []
        
        for test_id in range(num_tests):
            if verbose and num_tests > 1:
                print(f"Running test {test_id + 1}/{num_tests}")
            

            self.reset()
            

            state = env.reset()[1]
            test_data = {
                'periods': [],
                'prices': [],
                'actions_a': [],
                'actions_p': [],
                'rewards': [],
                'inventories': [],
                'batch_info': [],
                'ucb_values': [],
            }
            
            period = 0
            total_reward = 0
            
            while True:

                clock, _, inventory_now, inventory_all, _, _, _ = state
                test_data['inventories'].append(inventory_now)
                test_data['periods'].append(period)
                self.inventory_all = inventory_all if clock !=0 else self.a_max
                self.delta = self.epsilon_bar / (2 * (1 + self.r_max / self.inventory_all) * math.log(self.T))
                self.rho_W = 1 - self.delta * (1 + self.r_max / self.inventory_all) / (1 + self.delta)
                

                selected_price = self.get_action(period, state)
                test_data['prices'].append(selected_price)
                if clock == 0:
                    acquisition = self.a_max
                else:

                    values = (self.lambda_hat, self.mu_hat, self.price_list, self.a_max, self.T, self.I0, self.c_a, self.c_h)
                    self.clairvoyant = Clairvoyant(values)

                    acquisition = self.clairvoyant.get_acquisition(selected_price, inventory_all)
                test_data['actions_a'].append(acquisition)
                test_data['actions_p'].append(selected_price)
                

                batch_info = f"B{self.current_batch}" if self.warm_up_completed else f"W{self.warm_up_price_index}"
                test_data['batch_info'].append(batch_info)
                

                if selected_price in self.upper_confidence_bounds:
                    test_data['ucb_values'].append(self.upper_confidence_bounds[selected_price])
                else:
                    test_data['ucb_values'].append(0.0)
                
                

                done, next_state, reward = env.step((acquisition, selected_price))
                test_data['rewards'].append(reward)
                total_reward += reward
                

                self.update_observations(selected_price, next_state)

                state = next_state
                period += 1
                
                if done:
                    break
            
            test_data['total_reward'] = total_reward
            test_results.append(test_data)
            
            if verbose and num_tests > 1:
                print(f"Test {test_id + 1} completed: Total reward = {total_reward:.2f}, "
                      f"Periods = {period}")
        
        return test_results
    
    def reset(self):
        self.arrival_times = {p: [] for p in self.price_list}
        self.service_times = {p: [] for p in self.price_list}
        self.n_arrivals = {p: 0 for p in self.price_list}
        self.n_services = {p: 0 for p in self.price_list}
        self.total_periods = {p: 0 for p in self.price_list}
        

        self.lambda_hat = {p: 0.0 for p in self.price_list}
        self.mu_hat = {p: 0.0 for p in self.price_list}


        self.upper_confidence_bounds = {p: float('inf') for p in self.price_list}
        self.lower_confidence_bounds = {p: 0.0 for p in self.price_list}
        

        self.learning_prices = set(self.price_list)
        self.current_batch = 0
        self.periods_in_current_batch = 0
        self.current_batch_size = 0
        self.current_price = None
        self.warm_up_completed = False
        self.warm_up_price_index = 0
        self.warm_up_observations = {p: 0 for p in self.price_list}
    
    def get_learning_statistics(self):
        stats = {
            'learning_prices': list(self.learning_prices),
            'revenue_rates': {p: self.revenue_rate_hat[p] for p in self.learning_prices}, # type: ignore
            'ucb_values': {p: self.upper_confidence_bounds[p] for p in self.learning_prices},
            'utilization_rates': {p: self.estimate_utilization(p) for p in self.learning_prices},
            'observations': {p: self.n_arrivals[p] + self.n_services[p] for p in self.learning_prices}
        }
        return stats

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
    print("开始BUCB算法数值实验")
    print(f"实验时间: {datetime.now()}")
    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    

    num_scenarios = len(scenarios)
    num_tests_per_scenario = 10
    
    print(f"场景数量: {num_scenarios}")
    print(f"每个场景测试次数: {num_tests_per_scenario}")
    

    results = []
    
    print(f"\n{'='*60}")
    print("开始各个场景的BUCB测试...")
    

    for scenario_id in scenarios.keys():
        print(f"\n{'='*40}")
        print(f"测试场景 {scenario_id}")
        print(f"{'='*40}")
        

        scenario_config = scenarios[scenario_id]
        
        try:

            bucb_agent = BUCBAgent(
                price_list=scenario_config["price"],
                inventory_all=max_acquisition,
                r_max=200.0
            )
            

            env_fn = create_env_fn(scenario_config)
            

            scenario_rewards = []
            
            print(f"开始进行 {num_tests_per_scenario} 次测试...")
            
            for test_run in range(num_tests_per_scenario):
                try:

                    test_env = env_fn()
                    

                    test_results = bucb_agent.test_model(test_env, num_tests=1, verbose=False)
                    test_reward = test_results[0]['total_reward']
                    scenario_rewards.append(test_reward)
                    
                    if (test_run + 1) % 5 == 0 or test_run == 0:
                        print(f"  完成测试 {test_run + 1}/{num_tests_per_scenario}, "
                              f"当前奖励: {test_reward:.2f}")
                        
                except Exception as e:
                    print(f"  测试运行 {test_run + 1} 出现错误: {e}")

                    import traceback
                    traceback.print_exc()
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
        

        detailed_csv_path = os.path.join(results_dir, "bucb_detailed_results.csv")
        results_df.to_csv(detailed_csv_path, index=False)
        print(f"详细结果已保存到: {detailed_csv_path}")
        

        mean_results = np.zeros((num_scenarios, 1))
        std_results = np.zeros((num_scenarios, 1))
        
        for i, result in enumerate(results):
            mean_results[i, 0] = result['mean_reward']
            std_results[i, 0] = result['std_reward']
        

        mean_df = pd.DataFrame(mean_results, 
                              index=[f"Scenario_{i+1}" for i in range(num_scenarios)],
                              columns=["BUCB"])
        mean_csv_path = os.path.join(results_dir, "bucb_results_mean.csv")
        mean_df.to_csv(mean_csv_path)
        print(f"均值结果已保存到: {mean_csv_path}")
        

        std_df = pd.DataFrame(std_results,
                             index=[f"Scenario_{i+1}" for i in range(num_scenarios)], 
                             columns=["BUCB"])
        std_csv_path = os.path.join(results_dir, "bucb_results_std.csv")
        std_df.to_csv(std_csv_path)
        print(f"标准差结果已保存到: {std_csv_path}")
        

        print(f"\n{'='*60}")
        print("BUCB实验结果总结:")
        print("\n均值结果:")
        print(mean_df.round(2))
        print("\n标准差结果:")
        print(std_df.round(2))
        
    else:
        print("没有获得有效的实验结果!")
    
    print(f"\n实验完成! 总耗时: {datetime.now()}")
    print(f"结果文件保存在 {results_dir} 目录中")

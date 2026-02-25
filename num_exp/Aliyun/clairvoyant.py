import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import math


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from settings import scenarios, T, I_0, c_a, c_h, gamma, max_acquisition
from mutils.env_old import LeasingEnv
from mutils.baseline import Clairvoyant

class ClairvoyantAgent:
    """
    Clairvoyant算法智能体
    使用baseline中的Clairvoyant类实现
    """
    def __init__(self, lambda_dict, mu_dict, p_list, a_max, T, I0, c_a, c_h):
        self.lambda_dict = lambda_dict
        self.mu_dict = mu_dict
        self.p_list = p_list
        self.a_max = a_max
        self.T = T
        self.I0 = I0
        self.c_a = c_a
        self.c_h = c_h
        

        values = (lambda_dict, mu_dict, p_list, a_max, T, I0, c_a, c_h)
        self.clairvoyant = Clairvoyant(values)
        

        self.optimal_price = self.clairvoyant.optimal_price
        self.optimal_inventory = self.clairvoyant.optimal_inventory
        
        print(f"使用baseline Clairvoyant类计算得到:")
        print(f"最优价格: {self.optimal_price}, 最优库存: {self.optimal_inventory}")
    
    def get_action(self, PI):
        """
        根据当前物理库存PI决定动作
        使用baseline中的solve方法
        """
        price, acquisition = self.clairvoyant.solve(PI)

        acquisition = min(acquisition, self.a_max)
        return price, acquisition
    
    def test_model(self, env, num_tests=1, verbose=True):
        """
        测试Clairvoyant智能体的性能
        """
        if verbose:
            print(f"开始进行 {num_tests} 次Clairvoyant测试...")
        
        test_results = []
        
        for test_id in range(num_tests):
            if verbose and num_tests > 1:
                print(f"Running test {test_id + 1}/{num_tests}")
            

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
            PI = self.I0
            
            while True:

                _, _, inventory_now, _, _, _, _ = state
                test_data['inventories'].append(inventory_now)
                test_data['periods'].append(period)
                

                p, a = self.get_action(PI)
                test_data['actions_a'].append(a)
                test_data['actions_p'].append(p)
                

                done, next_state, reward = env.step((a, p))
                test_data['rewards'].append(reward)
                total_reward += reward
                

                PI += a
                
                state = next_state
                period += 1
                
                if done:
                    break
            # env.plot_history()
            test_data['total_reward'] = total_reward
            test_results.append(test_data)
            
            if verbose and num_tests > 1:
                print(f"Test {test_id + 1} completed: Total reward = {total_reward:.2f}, "
                      f"Periods = {period}, Final PI = {PI}")
        
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

if __name__ == "__main__":
    np.random.seed(42)
    print("开始Clairvoyant算法数值实验")
    print(f"实验时间: {datetime.now()}")
    print("使用baseline中的Clairvoyant类进行计算")
    

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
    print("开始各个场景的Clairvoyant测试...")
    

    for scenario_id in scenarios.keys():
        print(f"\n{'='*40}")
        print(f"测试场景 {scenario_id}")
        print(f"{'='*40}")
        

        scenario_config = scenarios[scenario_id]
        
        try:

            clairvoyant_agent = ClairvoyantAgent(
                lambda_dict=scenario_config["lambda_dict"],
                mu_dict=scenario_config["mu_dict"],
                p_list=scenario_config["price"],
                a_max=max_acquisition,
                T=T,
                I0=I_0,
                c_a=c_a,
                c_h=c_h
            )
            

            env_fn = create_env_fn(scenario_config)
            

            scenario_rewards = []
            
            print(f"开始进行 {num_tests_per_scenario} 次测试...")
            
            for test_run in range(num_tests_per_scenario):
                try:

                    test_env = env_fn()
                    

                    test_results = clairvoyant_agent.test_model(test_env, num_tests=1, verbose=False)
                    test_reward = test_results[0]['total_reward']
                    scenario_rewards.append(test_reward)
                    
                    if (test_run + 1) % 5 == 0 or test_run == 0:
                        print(f"  完成测试 {test_run + 1}/{num_tests_per_scenario}, "
                              f"当前奖励: {test_reward:.6f}")
                        print(f"  当前场景奖励列表: {[f'{r:.6f}' for r in scenario_rewards]}")

                        if len(scenario_rewards) > 1:
                            unique_rewards = len(set(scenario_rewards))
                            print(f"  唯一奖励值数量: {unique_rewards}")
                            if unique_rewards == 1:
                                print(f"  警告: 所有奖励值都相同! 这可能导致方差为0")
                        
                except Exception as e:
                    print(f"  测试运行 {test_run + 1} 出现错误: {e}")
                    continue
            
            if scenario_rewards:
                mean_reward = np.mean(scenario_rewards)

                unique_rewards = set(scenario_rewards)
                print(f"场景 {scenario_id} 唯一奖励值: {len(unique_rewards)} 个")
                print(f"场景 {scenario_id} 奖励值范围: {min(scenario_rewards):.6f} 到 {max(scenario_rewards):.6f}")
                
                if len(unique_rewards) == 1:
                    print(f"场景 {scenario_id} 警告: 所有测试产生相同奖励值，方差将为0")
                    std_reward = 0.0
                else:
                    std_reward = np.std(scenario_rewards, ddof=1) if len(scenario_rewards) > 1 else 0.0
                
                print(f"场景 {scenario_id} 原始奖励数据: {[f'{r:.6f}' for r in scenario_rewards]}")
                print(f"场景 {scenario_id} 奖励统计: 均值={mean_reward:.6f}, 标准差={std_reward:.6f}")
                
                results.append({
                    'scenario_id': scenario_id,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'optimal_price': clairvoyant_agent.optimal_price,
                    'optimal_inventory': clairvoyant_agent.optimal_inventory,
                    'num_tests': len(scenario_rewards),
                    'all_rewards': scenario_rewards,
                    'unique_rewards': len(unique_rewards)
                })
                
                print(f"场景 {scenario_id} 测试完成:")
                print(f"  最优价格: {clairvoyant_agent.optimal_price}")
                print(f"  最优库存: {clairvoyant_agent.optimal_inventory}")
                print(f"  平均奖励: {mean_reward:.6f}")
                print(f"  标准差: {std_reward:.6f}")
                print(f"  有效测试次数: {len(scenario_rewards)}")
                print(f"  唯一奖励值数量: {len(unique_rewards)}")
            else:
                print(f"场景 {scenario_id}: 所有测试都失败了")
                
        except Exception as e:
            print(f"场景 {scenario_id} 初始化失败: {e}")
            continue
    

    print(f"\n{'='*60}")
    print("保存实验结果...")
    
    if results:

        results_df = pd.DataFrame([{
            'scenario_id': r['scenario_id'],
            'mean_reward': r['mean_reward'],
            'std_reward': r['std_reward'],
            'optimal_price': r['optimal_price'],
            'optimal_inventory': r['optimal_inventory'],
            'num_tests': r['num_tests']
        } for r in results])
        

        detailed_csv_path = os.path.join(results_dir, "clairvoyant_detailed_results.csv")
        results_df.to_csv(detailed_csv_path, index=False)
        print(f"详细结果已保存到: {detailed_csv_path}")
        

        mean_results = np.zeros((num_scenarios, 1))
        std_results = np.zeros((num_scenarios, 1))
        

        for result in results:
            scenario_idx = result['scenario_id'] - 1
            mean_results[scenario_idx, 0] = result['mean_reward']
            std_results[scenario_idx, 0] = result['std_reward']
        

        mean_df = pd.DataFrame(mean_results, 
                              index=[f"Scenario_{i+1}" for i in range(num_scenarios)],
                              columns=["Clairvoyant"])
        mean_csv_path = os.path.join(results_dir, "clairvoyant_results_mean.csv")
        mean_df.to_csv(mean_csv_path)
        print(f"均值结果已保存到: {mean_csv_path}")
        

        std_df = pd.DataFrame(std_results,
                             index=[f"Scenario_{i+1}" for i in range(num_scenarios)], 
                             columns=["Clairvoyant"])
        std_csv_path = os.path.join(results_dir, "clairvoyant_results_std.csv")
        std_df.to_csv(std_csv_path)
        print(f"标准差结果已保存到: {std_csv_path}")
        

        print(f"\n{'='*60}")
        print("Clairvoyant实验结果总结:")
        print("\n均值结果:")
        print(mean_df.round(2))
        print("\n标准差结果:")
        print(std_df.round(2))
        

        print("\n最优策略总结:")
        for result in sorted(results, key=lambda x: x['scenario_id']):
            print(f"场景 {result['scenario_id']}: "
                  f"最优价格={result['optimal_price']}, "
                  f"最优库存={result['optimal_inventory']}, "
                  f"平均奖励={result['mean_reward']:.2f}±{result['std_reward']:.2f}")
            print(f"  原始奖励数据: {result.get('all_rewards', 'N/A')}")
        
    else:
        print("没有获得有效的实验结果!")
    
    print(f"\n实验完成! 总耗时: {datetime.now()}")
    print(f"结果文件保存在 {results_dir} 目录中")

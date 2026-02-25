import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import math
from collections import defaultdict



sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from settings import scenarios, T, I_0, c_a, c_h, max_acquisition
from mutils.env import LeasingEnv
from mutils.baseline import Clairvoyant

class POAgent:
    """
    在线 Predict-then-Optimize (PO) 算法智能体。
    在每个时间步，该算法首先使用至今为止的所有历史数据来更新系统参数（到达率和服务率）的估计，
    然后使用这些最新的估计值在一个优化模型中做出当前最优的决策。
    """
    
    def __init__(self, price_list, T_horizon, I0, c_a, c_h, a_max, verbose=True):
        """
        初始化在线 PO 智能体
        
        Args:
            price_list (list): 候选价格列表
            T_horizon (int): 规划期限
            I0 (int): 初始库存
            c_a (float): 单位采购成本
            c_h (float): 单位持有成本
            a_max (int): 最大采购量
            verbose (bool): 是否输出详细信息
        """
        self.price_list = sorted(price_list)
        self.T_horizon = T_horizon
        self.I0 = I0
        self.c_a = c_a
        self.c_h = c_h
        self.a_max = a_max
        self.verbose = verbose
        self.learning = True
        self.p_now_test = 0

        self.lambda_hat = {p: 10 for p in self.price_list}
        self.mu_hat = {p: 10 for p in self.price_list}

    def _update_estimates(self, arrival_his, service_his):
        """
        使用完整的历史数据在线更新参数估计。
        
        Args:
            arrival_his (dict): 每个价格的到达间隔时间历史
            service_his (dict): 每个价格的服务时间历史
        """
        for p in self.price_list:

            if arrival_his.get(p) and len(arrival_his[p]) > 0:
                mean_inter_arrival = np.mean(arrival_his[p])
                self.lambda_hat[p] = 1.0 / mean_inter_arrival if mean_inter_arrival > 0 else 10
            

            if service_his.get(p) and len(service_his[p]) > 0:
                mean_service_time = np.mean(service_his[p])
                self.mu_hat[p] = 1.0 / mean_service_time if mean_service_time > 0 else 10


    def get_action(self, period, env_state):
        """
        基于当前状态和历史数据，在线进行预测和优化。
        
        Args:
            period (int): 当前周期
            env_state (tuple): 当前环境状态
            
        Returns:
            tuple: 最佳动作 (a, p)
        """
        clock, _, inventory_now, inventory_all, arrival_his, service_his, holding = env_state


        values = [
            self.lambda_hat,
            self.mu_hat,
            self.price_list,
            self.a_max,
            self.T_horizon,
            inventory_all,
            c_a, 
            c_h
        ]

        if self.learning:
            self._update_estimates(arrival_his, service_his)
            if self.p_now_test == len(self.price_list):
                self.learning = False
            else:

                current_price = self.price_list[self.p_now_test]
                self.p_now_test += 1
                self.clairvoyant = Clairvoyant(values)
                if clock == 0:
                    a = self.clairvoyant.solve(inventory_all)[1]
                else:
                    a = 0
                # print(a, current_price)
                return (a, current_price)

        if not self.learning:
            self._update_estimates(arrival_his, service_his)
            self.clairvoyant = Clairvoyant(values)
            p,a = self.clairvoyant.solve(inventory_all) 
            
            return (a, p)
    

    def test_model(self, env_factory, num_tests=1, verbose=True):
        """
        测试在线 PO 智能体的性能。
        
        Args:
            env_factory (function): 创建环境实例的工厂函数
            num_tests (int): 测试次数
            verbose (bool): 是否输出详细信息
            
        Returns:
            list: 测试结果列表
        """
        if verbose:
            print(f"正在测试在线 PO 智能体，共 {num_tests} 次试验。")

        test_results = []
        for test_id in range(num_tests):
            env = env_factory()
            self.reset()
            
            done = False
            total_profit = 0
            period = 0
            
            _, state, _ = env.reset()

            while not done and period < self.T_horizon:
                action = self.get_action(period, state)
                # print(action)
                done, state, reward = env.step(action)
                total_profit += reward
                period += 1
            
            # print(self.lambda_hat, self.mu_hat,self.clairvoyant.solve(state[3]) )
            
            # env.plot_history()
            # env.plot_action()
            
            test_results.append({
                'test_id': test_id + 1,
                'total_profit': total_profit,
                'avg_profit': total_profit / period if period > 0 else 0
            })
            if verbose:
                print(f"  试验 {test_id + 1}/{num_tests}: 总利润 = {total_profit:.2f}")
        
        return test_results

    def reset(self):
        self.lambda_hat = {p: np.random.rand() for p in self.price_list}
        self.mu_hat = {p: np.random.rand() for p in self.price_list}
        self.learning = True
        self.p_now_test = 0

def main():
    start_time = datetime.now()
    np.random.seed(42)
    print("开始在线 Predict-then-Optimize (PO) 算法数值实验")
    print(f"实验开始时间: {start_time}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    num_scenarios = len(scenarios)
    num_tests_per_scenario = 10

    print(f"场景数量: {num_scenarios}")
    print(f"每个场景测试次数: {num_tests_per_scenario}")
    
    results = []
    
    print(f"\n{'='*60}")
    print("开始各个场景的在线 PO 测试...")
    
    for scenario_id, scenario_config in scenarios.items():
        print(f"\n{'='*40}")
        print(f"测试场景 {scenario_id}")
        print(f"{'='*40}")
        
        def env_factory():
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

        try:
            agent = POAgent(
                price_list=scenario_config["price"],
                T_horizon=T,
                I0=I_0,
                c_a=c_a,
                c_h=c_h,
                a_max=max_acquisition,
                verbose=False
            )

            # print(agent.price_list)

            scenario_results = agent.test_model(env_factory, num_tests=num_tests_per_scenario, verbose=True)
            
            for res in scenario_results:
                results.append({
                    'scenario': scenario_id,
                    'algorithm': 'PO',
                    'test_id': res['test_id'],
                    'total_profit': res['total_profit'],
                    'avg_profit': res['avg_profit']
                })
            
            print(f"场景 {scenario_id} 测试完成。")

        except Exception as e:
            print(f"场景 {scenario_id} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    

    print(f"\n{'='*60}")
    print("保存实验结果...")
    
    if results:

        results_df = pd.DataFrame(results)
        detailed_csv_path = os.path.join(results_dir, "PO_detailed_results.csv")
        results_df.to_csv(detailed_csv_path, index=False)
        print(f"详细结果已保存到: {detailed_csv_path}")


        summary = results_df.groupby('scenario')['total_profit'].agg(['mean', 'std'])
        

        mean_df = summary[['mean']]
        mean_df.columns = ['PO']
        mean_df.index = [f"Scenario_{s}" for s in summary.index]
        mean_csv_path = os.path.join(results_dir, "PO_results_mean.csv")
        mean_df.to_csv(mean_csv_path)
        print(f"均值结果已保存到: {mean_csv_path}")
        

        std_df = summary[['std']]
        std_df.columns = ['PO']
        std_df.index = [f"Scenario_{s}" for s in summary.index]
        std_csv_path = os.path.join(results_dir, "PO_results_std.csv")
        std_df.to_csv(std_csv_path)
        print(f"标准差结果已保存到: {std_csv_path}")
        

        print(f"\n{'='*60}")
        print("PO Online 实验结果总结:")
        print("\n均值结果:")
        print(mean_df.round(2))
        print("\n标准差结果:")
        print(std_df.round(2))
        
    else:
        print("没有获得有效的实验结果!")
    
    end_time = datetime.now()
    print(f"\n实验完成! 总耗时: {end_time - start_time}")
    print(f"结果文件保存在 {results_dir} 目录中")

if __name__ == "__main__":
    main()

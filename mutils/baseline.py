from .env import LeasingEnv
import numpy as np
import math
import time
from decimal import Decimal, getcontext

class Clairvoyant:
    def __init__(self, values):
        self.values = values
        self.B_dict = {}
        self.env = LeasingEnv(*values)
        self.optimal_price, self.optimal_inventory = self.get_optimal_price_inventory()
        

    def get_optimal_price_inventory(self):
        """
        计算最优价格和库存水平
        """
        p_list = self.env.p_list

        max_lambda = max(self.env.lambda_dict.values())
        I_max = min(150, int(max_lambda * 20))
        I_list = range(max(self.env.I0, 0), max(I_max + 1, self.env.T * self.env.a_max + self.env.I0) + 1)
        # print(self.env.I0, I_max, max_lambda)

        r_list = {}

        for p in p_list:
            for I in I_list:
                c_a = self.env.c_a
                c_h = self.env.c_h
                T = self.env.T
                I0 = self.env.I0



                # start_time = time.time()
                expected_remaining = self.calc_expected_inventory(p, I)

                expected_sales = I - expected_remaining
                expected_profit = p * expected_sales - c_h * expected_remaining - c_a * max(0, I - I0)/T
                r_list[(float(p), int(I))] = expected_profit
        
        if not r_list:

            return p_list[0], max(1, self.env.I0)
            

        optimal_price, optimal_inventory = max(r_list, key=r_list.get) # type: ignore
        max_profit = r_list[(optimal_price, optimal_inventory)]
        

        return optimal_price, optimal_inventory

    def Erlang_B(self, I, rho):
        if I == 0:
            return 1
        else:
            if (I - 1, rho) not in self.B_dict:
                self.B_dict[(I - 1, rho)] = self.Erlang_B(I - 1, rho)
            if (I, rho) not in self.B_dict:
                self.B_dict[(I, rho)] = rho * self.B_dict[(I - 1, rho)] / (I + rho * self.B_dict[(I - 1, rho)])
            return self.B_dict[(I, rho)]

    def calc_expected_inventory(self, p, I):
        """
        计算在价格p和库存水平I下的期望库存
        使用M/M/I/I排队系统的稳态概率公式
        """
        mu = self.env.mu_dict[p]
        lam = self.env.lambda_dict[p]
        if mu == 0:
            return I
        if lam == 0:
            return 0
        rho = lam / mu
        
        if I == 0:
            return 0
        




        try:
            I = Decimal(I)
            rho = Decimal(rho)
            for i in range(1, int(I)+1, 800):
                self.Erlang_B(i, rho)

            erlang_B = self.Erlang_B(I, rho)
            expected_inventory = I - rho * (1-erlang_B)

        except Exception as e:
            print(f"计算期望库存时出错: {e}")
            expected_inventory = I - rho + rho**(I+1)//math.factorial(I)//np.sum([rho**i//math.factorial(i) for i in range(I+1)])
            # print(rho, I, expected_inventory, e)
        

        expected_inventory = float(expected_inventory)
        return max(expected_inventory, 0)


    def get_acquisition(self, p, PI):
        best_I = np.nan
        best_profit = -np.inf
        for I in range(PI, PI+self.env.a_max + 1):
            expected_remaining = self.calc_expected_inventory(p, I)
            expected_profit = p * (I - expected_remaining) - self.env.c_h * expected_remaining - self.env.c_a * max(0, I - PI)/self.env.T
            if expected_profit > best_profit:
                best_profit = expected_profit
                best_I = I  
        if best_I is None:
            return 0
        else:
            return min(self.env.a_max,best_I - PI)

    def solve(self, PI):
        """
        求解最优策略
        """
        return self.optimal_price, np.min([self.env.a_max,np.max([self.optimal_inventory - PI, 0])])


if __name__ == "__main__":

    lambda_dict = {10: 20, 20: 19, 30: 18}
    mu_dict = {10: 1, 20: 1.5, 30: 2}
    p_list = [10, 20, 30]
    a_max = 10
    T = 100
    I0 = 0
    c_a = 0.5
    c_h = 1e2

    values = (lambda_dict, mu_dict, p_list, a_max, T, I0, c_a, c_h)
    clairvoyant = Clairvoyant(values)
    optimal_price, optimal_inventory = clairvoyant.get_optimal_price_inventory()
    print(f"Optimal Price: {optimal_price}, Optimal Inventory: {optimal_inventory}")

        
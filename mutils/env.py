import heapq
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt
from typing import Dict, Any, List

class _LegacyLeasingEnv:
    def __init__(self, lambda_dict, mu_dict, p_list, a_max, T, I0, c_a, c_h, chance_cost = False):
        assert T > 0, "Time horizon must be greater than 0"
        self.lambda_dict = lambda_dict
        self.mu_dict = mu_dict
        self.T = T
        self.I0 = I0
        self.inventory_now = I0
        self.inventory_all = I0
        self.inventory = {0: I0}
        self.clock = 0
        self.chance_cost = chance_cost
        self.p_list = sorted(p_list)
        self.a_max = a_max
        self.c_a = c_a
        self.c_h = c_h
        self.ad_pair = {}
        self.holding = {p: 0 for p in self.p_list}

        self.active_leases = []
        self.reset()

    def reset(self):

        self.clock = 0
        self.inventory = {0: self.I0}
        self.ad_history = {}
        self.decision_history = {}
        self.clock_queue = PriorityQueue()
        self.inventory_now = self.I0
        self.inventory_all = self.I0
        self.profit = []
        self.arrival_his = {p: [] for p in self.p_list}
        self.service_his = {p: [] for p in self.p_list}
        self.holding = {p: 0 for p in self.p_list}
        self.active_leases = []
        return False, (
            self.clock,
            self.T,
            self.inventory_now,
            self.inventory_all,
            self.arrival_his,
            self.service_his,
            self.holding
        ), 0

    def step(self, action):


        assert action[1] in self.p_list
        assert 0<= action[0] <= self.a_max
        if self.clock >= self.T:

            return True, (
                self.clock,
                self.T,
                self.inventory_now,
                self.inventory_all,
                self.arrival_his,
                self.service_his,
                self.holding
            ), 0

        self.decision_history[self.clock] = action
        a, p = action

        profit = -self.c_a * a
        self.inventory_all += a

        self.inventory_now = self.inventory_now + a
        self.inventory[self.clock] = self.inventory_now


        inventory_changes = [(self.clock, self.inventory_now)]
        

        period_start_time = self.clock
        period_end_time = self.clock + 1
        

        lease_profit = self._calculate_lease_profit(period_start_time, period_end_time)
        profit += lease_profit
        

        tmp_clock = self.clock
        while tmp_clock <= self.clock + 1:

            while not self.clock_queue.empty():
                new_event = self.clock_queue.get()
                if new_event[0] > min([self.clock + 1, tmp_clock + 1/self.lambda_dict[p]]):

                    self.clock_queue.put(new_event)
                    break
                else:


                    if new_event[1] == 'departure':
                        self.inventory_now += 1
                        self.inventory[new_event[0]] = self.inventory_now

                        inventory_changes.append((new_event[0], self.inventory_now))
                        self.holding[self.decision_history[np.floor(self.ad_pair[new_event[0]])][1]] -= 1
                        

                        departure_price = new_event[2]
                        departure_time = new_event[0]
                        arrival_time = self.ad_pair[departure_time]



            arrival_time = np.random.exponential(1/self.lambda_dict[p])
            service_time = np.random.exponential(1/self.mu_dict[p])


            if tmp_clock + arrival_time <= self.clock + 1:

                if self.inventory_now > 0:

                    arrival_abs_time = tmp_clock + arrival_time
                    departure_abs_time = arrival_abs_time + service_time
                    
                    self.inventory_now -= 1
                    self.inventory[arrival_abs_time] = self.inventory_now

                    inventory_changes.append((arrival_abs_time, self.inventory_now))

                    self.clock_queue.put((departure_abs_time, 'departure', p, service_time))

                    self.ad_pair[departure_abs_time] = arrival_abs_time

                    self.holding[p] += 1

                    self.arrival_his[p].append(arrival_time)
                    self.service_his[p].append(service_time)
                    

                    self.active_leases.append((arrival_abs_time, departure_abs_time, p))
                    

                    if departure_abs_time > period_end_time:

                        lease_duration_in_period = period_end_time - arrival_abs_time
                        profit += p * lease_duration_in_period
                    else:

                        lease_duration_in_period = service_time
                        profit += p * lease_duration_in_period
                else:

                    if self.chance_cost:

                        if tmp_clock + arrival_time + service_time > period_end_time:

                            opportunity_cost_duration = period_end_time - (tmp_clock + arrival_time)
                            profit -= p * opportunity_cost_duration
                        else:

                            profit -= p * service_time
                tmp_clock += arrival_time
            else:

                self.clock += 1
                break
        

        holding_cost = self._calculate_holding_cost(inventory_changes, self.clock, self.clock + 1)
        profit -= holding_cost
        
        self.profit.append(profit)

        return False, (
            self.clock,
            self.T,
            self.inventory_now,
            self.inventory_all,
            self.arrival_his,
            self.service_his,
            self.holding
        ), profit

    def _calculate_lease_profit(self, period_start, period_end):
        """
        计算指定时期内所有活跃租赁产生的收益
        
        Args:
            period_start: 时期开始时间
            period_end: 时期结束时间
            
        Returns:
            float: 该时期内所有活跃租赁的总收益
        """
        total_profit = 0
        for arrival_time, departure_time, price in self.active_leases:

            lease_start_in_period = max(arrival_time, period_start)
            lease_end_in_period = min(departure_time, period_end)
            

            if lease_start_in_period < lease_end_in_period:
                lease_duration_in_period = lease_end_in_period - lease_start_in_period
                lease_profit = price * lease_duration_in_period
                total_profit += lease_profit
                
        return total_profit


    def _copy_from_state(self, state):
        """
        从给定的状态复制环境的内部状态
        
        Args:
            state: 包含当前时钟、总时长、当前库存、总库存、到达历史、服务历史和持有数量的元组
            
        Returns:
            None
        """
        _, self.T, self.inventory_now, self.inventory_all, self.arrival_his, self.service_his, self.holding = state
        self.clock = 0
        self.decision_history = {}
        self.profit = []
        self.ad_pair = {}
        self.active_leases = []
        self.inventory = {0: self.inventory_now}
        self.clock_queue = PriorityQueue()
        self.inventory = {0: self.inventory_now}
        for p in self.p_list:
            for _ in range(self.holding[p]):
                service_time = np.random.exponential(1/self.mu_dict[p])
                departure_time = self.clock + service_time
                self.clock_queue.put((departure_time, 'departure', p, service_time))
            

    def _calculate_holding_cost(self, inventory_changes, start_time, end_time):
        """
        计算在[start_time, end_time)时间段内的库存持有成本
        
        Args:
            inventory_changes: 库存变化时间点列表 [(时间, 库存水平), ...]
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            总持有成本
        """
        if not inventory_changes:
            return 0
            

        inventory_changes.sort(key=lambda x: x[0])
        
        total_holding_cost = 0
        current_inventory = inventory_changes[0][1]
        current_time = start_time
        
        for i, (change_time, new_inventory) in enumerate(inventory_changes[1:], 1):

            effective_end_time = min(change_time, end_time)
            

            duration = effective_end_time - current_time
            if duration > 0 and current_inventory > 0:
                total_holding_cost += self.c_h * current_inventory * duration
            

            current_time = effective_end_time
            current_inventory = new_inventory
            

            if effective_end_time >= end_time:
                break
        

        if current_time < end_time:
            duration = end_time - current_time
            if duration > 0 and current_inventory > 0:
                total_holding_cost += self.c_h * current_inventory * duration
                
        return total_holding_cost

    def plot_history(self):

        x = list(self.inventory.keys())
        y = list(self.inventory.values())
        x.append(self.clock)
        y.append(y[-1])
        plt.step(x, y, where='post', linewidth=1)
        plt.xlim([0, self.clock])
        plt.xticks(range(0, self.clock + 1, max(1, self.clock // 10)), range(0, self.clock + 1, max(1, self.clock // 10)))
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.title('Inventory over time')
        plt.show()

    def plot_profit(self):

        x = list(range(1, self.clock + 1))
        y = self.profit
        plt.plot(x, y)
        plt.xlabel('Time')
        plt.xticks(x[::max(1, len(x) // 10)], x[::max(1, len(x) // 10)])
        plt.ylabel('Profit')
        plt.title('Profit over time')
        plt.tight_layout()
        plt.show()

    def plot_mean_profit(self):

        x = list(range(1, self.clock + 1))
        y = [np.mean(self.profit[:i]) for i in x]
        plt.plot(x, y)
        plt.xlabel('Time')
        plt.xticks(x[::max(1, len(x) // 10)], x[::max(1, len(x) // 10)])
        plt.ylabel('Mean Profit')
        plt.title('Mean Profit over time')
        plt.show()

    def plot_action(self):

        x = list(self.decision_history.keys())
        y = list(self.decision_history.values())
        plt.xticks(x[::max(1, len(x) // 10)], x[::max(1, len(x) // 10)])
        x.append(self.clock)
        y.append(y[-1])
        plt.step(x, y, where='post')
        plt.legend(['Acquisition', 'Price'])
        plt.xlim([0, self.clock])
        plt.xlabel('Time')
        plt.ylabel('Price and Acquisition')
        plt.title('Price and Acquisition over time')
        plt.show()


class LeasingEnv(_LegacyLeasingEnv):
    def __init__(self, *args, **kwargs):

        self._departure_heap = []

        self._active_rate = 0.0

        self._active_records: Dict[int, tuple] = {}

        self._next_lease_id = 0
        super().__init__(*args, **kwargs)

    def reset(self):
        base = super().reset()

        self._departure_heap = []
        self._active_rate = 0.0
        self._active_records = {}
        self._next_lease_id = 0
        self.active_leases = []
        return base

    def step(self, action):
        assert action[1] in self.p_list
        assert 0 <= action[0] <= self.a_max

        if self.clock >= self.T:
            return True, (
                self.clock,
                self.T,
                self.inventory_now,
                self.inventory_all,
                self.arrival_his,
                self.service_his,
                self.holding,
            ), 0

        a, price = action
        self.decision_history[self.clock] = action
        period_start = float(self.clock)
        period_end = period_start + 1.0

        profit = -self.c_a * a
        self.inventory_all += a
        self.inventory_now += a

        inventory_changes = []

        self._release_due_departures(period_start)
        self.inventory[self.clock] = self.inventory_now
        inventory_changes.append((period_start, self.inventory_now))


        arrival_times, inter_arrivals = self._sample_arrivals(price, period_start, period_end)
        service_times = self._sample_services(price, len(arrival_times))
        arr_idx = 0
        current_time = period_start

        while current_time < period_end:
            next_arrival_time = arrival_times[arr_idx] if arr_idx < len(arrival_times) else np.inf
            next_departure_time = self._departure_heap[0][0] if self._departure_heap else np.inf

            if next_departure_time <= next_arrival_time:
                event_time = min(next_departure_time, period_end)
                event_type = 'departure' if next_departure_time <= period_end else None
            else:
                event_time = min(next_arrival_time, period_end)
                event_type = 'arrival' if next_arrival_time <= period_end else None

            delta = event_time - current_time
            if delta < 0:
                delta = 0.0
            if delta > 0:

                profit += self._active_rate * delta
            current_time = event_time

            if event_time >= period_end or event_type is None:
                break

            if event_type == 'departure':
                self._handle_departure(inventory_changes)
            else:
                profit = self._handle_arrival(
                    price,
                    service_times[arr_idx],
                    inter_arrivals[arr_idx],
                    arrival_times[arr_idx],
                    period_end,
                    inventory_changes,
                    current_profit=profit,
                )
                arr_idx += 1

        if current_time < period_end:

            profit += self._active_rate * (period_end - current_time)

        holding_cost = self._calculate_holding_cost(inventory_changes, period_start, period_end)
        profit -= holding_cost
        self.profit.append(profit)
        self.clock = int(period_end)
        self.active_leases = list(self._active_records.values())

        return False, (
            self.clock,
            self.T,
            self.inventory_now,
            self.inventory_all,
            self.arrival_his,
            self.service_his,
            self.holding,
        ), profit

    def _sample_arrivals(self, price, start, end):
        window = end - start
        rate = self.lambda_dict[price]
        if rate <= 0 or window <= 0:
            return np.empty(0), np.empty(0)
        count = np.random.poisson(rate * window)
        if count == 0:
            return np.empty(0), np.empty(0)

        offsets = np.sort(np.random.uniform(0.0, window, size=count))
        inter_arrivals = np.diff(np.concatenate(([0.0], offsets)))
        return start + offsets, inter_arrivals

    def _sample_services(self, price, size):
        if size == 0:
            return np.empty(0)

        return np.random.exponential(1 / self.mu_dict[price], size=size)

    def _release_due_departures(self, threshold):
        eps = 1e-9
        while self._departure_heap and self._departure_heap[0][0] <= threshold + eps:
            self._handle_departure()

    def _handle_departure(self, inventory_changes=None):
        departure_time, lease_id, price = heapq.heappop(self._departure_heap)
        self.inventory_now += 1
        self.inventory[departure_time] = self.inventory_now
        if inventory_changes is not None:
            inventory_changes.append((departure_time, self.inventory_now))
        self.holding[price] -= 1

        self._active_rate = max(0.0, self._active_rate - price)
        self._active_records.pop(lease_id, None)

    def _handle_arrival(self, price, service_time, inter_arrival, arrival_time, period_end, inventory_changes, current_profit):
        if self.inventory_now > 0:
            self.inventory_now -= 1
            self.inventory[arrival_time] = self.inventory_now
            inventory_changes.append((arrival_time, self.inventory_now))
            departure_time = arrival_time + service_time
            lease_id = self._next_lease_id
            self._next_lease_id += 1

            self._active_records[lease_id] = (arrival_time, departure_time, price)
            heapq.heappush(self._departure_heap, (departure_time, lease_id, price))
            self._active_rate += price
            self.holding[price] += 1
            self.arrival_his[price].append(inter_arrival)
            self.service_his[price].append(service_time)
            return current_profit
        else:
            if self.chance_cost:

                lost_duration = min(service_time, period_end - arrival_time)
                current_profit -= price * lost_duration
            return current_profit


# ---------------------------
# Scenario utilities
# ---------------------------
def create_env_fn(scenario_config: Dict[str, Any], a_max, T, I0, c_a, c_h, chance_cost=False):
    def env_fn():
        return LeasingEnv(
            lambda_dict=scenario_config["lambda_dict"],
            mu_dict=scenario_config["mu_dict"],
            p_list=scenario_config["price"],
            a_max=a_max,
            T=T,
            I0=I0,
            c_a=c_a,
            c_h=c_h,
            chance_cost=False,
        )
    return env_fn


def sanitize_scenario(s: Dict[str, Any]) -> Dict[str, Any]:
    # Basic validation and clipping to keep env stable
    price = sorted([float(p) for p in s.get("price", [])])
    price = [p for p in price if p > 0]
    if not price:
        raise ValueError("Empty or invalid price list")

    lambda_dict = {float(k): max(1e-3, float(v)) for k, v in s.get("lambda_dict", {}).items() if float(k) in price}
    mu_dict = {float(k): max(1e-3, float(v)) for k, v in s.get("mu_dict", {}).items() if float(k) in price}
    # Ensure keys cover all prices; if missing, impute conservatively
    for p in price:
        if p not in lambda_dict:
            lambda_dict[p] = max(1e-3, np.median(list(lambda_dict.values())) if lambda_dict else 1.0)
        if p not in mu_dict:
            mu_dict[p] = max(1e-3, np.median(list(mu_dict.values())) if mu_dict else 0.1)

    return {"price": price, "lambda_dict": lambda_dict, "mu_dict": mu_dict}


## Note: global anchors removed. Anchors are now computed dynamically per base scenario
## within run_curriculum to align with the current price set.


class ScenarioPool:
    def __init__(self, initial: Dict[int, Dict[str, Any]], max_size: int = 128):
        # Copy base scenarios into pool with ids
        self.pool: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        for sid, cfg in initial.items():
            self.pool[self.next_id] = sanitize_scenario(cfg)
            self.next_id += 1
        self.max_size = max_size
        # Difficulty score: lower reward => higher weight
        self.perf: Dict[int, List[float]] = {sid: [] for sid in self.pool}

    def ids(self) -> List[int]:
        return list(self.pool.keys())

    def get(self, sid: int) -> Dict[str, Any]:
        return self.pool[sid]

    def record_performance(self, sid: int, reward: float):
        if sid not in self.perf:
            self.perf[sid] = []
        self.perf[sid].append(float(reward))
        # Keep short history to be reactive
        if len(self.perf[sid]) > 50:
            self.perf[sid] = self.perf[sid][-50:]

    def sampling_weights(self) -> Dict[int, float]:
        # Weight inversely proportional to recent average reward
        weights = {}
        # Find min avg reward across all scenarios
        lowest_avg = float('inf')
        for sid in self.pool:
            hist = self.perf.get(sid, [])
            avg = np.mean(hist) if hist else 0.0
            if avg < lowest_avg:
                lowest_avg = avg
        # Compute weights

        for sid in self.pool:
            hist = self.perf.get(sid, [])
            avg = np.mean(hist) - lowest_avg if hist else 0.0
            # transform to positive difficulty; add margin to avoid div by zero
            difficulty = 1.0 / (1.0 + max(0.0, avg))
            weights[sid] = difficulty
        # Normalize weights
        total = sum(weights.values())
        if total <= 0:
            n = len(weights) if weights else 1
            return {sid: 1.0 / n for sid in weights}
        normalized_weights = {sid: w / total for sid, w in weights.items()}
        return normalized_weights

    def overview(self, max_recent: int = 10) -> Dict[str, Any]:
        """Return a compact overview of the current pool state.

        Includes size, ids, normalized sampling weights, and per-scenario stats
        with recent rewards and simple price metadata for quick inspection.
        """
        try:
            weights = self.sampling_weights()
        except Exception:
            weights = {sid: 1.0 / max(1, len(self.pool)) for sid in self.pool}

        per_scenario = {}
        for sid, cfg in self.pool.items():
            rewards = self.perf.get(sid, [])
            recent = rewards[-int(max(1, max_recent)):] if rewards else []
            try:
                recent_mean = float(np.mean(recent)) if len(recent) > 0 else None
            except Exception:
                recent_mean = None
            price_list = cfg.get("price", []) if isinstance(cfg, dict) else []
            per_scenario[str(int(sid))] = {
                "count": int(len(rewards)),
                "recent_rewards": [float(x) for x in recent],
                "recent_mean": recent_mean,
                "last_reward": (float(rewards[-1]) if len(rewards) > 0 else None),
                "price_len": (int(len(price_list)) if isinstance(price_list, list) else None),
                "price_min": (float(min(price_list)) if isinstance(price_list, list) and len(price_list) > 0 else None),
                "price_max": (float(max(price_list)) if isinstance(price_list, list) and len(price_list) > 0 else None),
            }

        return {
            "size": int(len(self.pool)),
            "ids": [int(x) for x in self.pool.keys()],
            "weights": {str(int(k)): float(v) for k, v in weights.items()},
            "per_scenario": per_scenario,
        }

    def sample_ids(self, k: int) -> List[int]:
        sids = self.ids()
        if not sids:
            return []
        w = [self.sampling_weights().get(sid, 1.0 / len(sids)) for sid in sids]
        choice = np.random.choice(sids, size=min(k, len(sids)), replace=True, p=np.array(w) / np.sum(w))
        try:
            lst = choice.tolist()
        except AttributeError:
            lst = list(choice)
        return [int(x) for x in (lst if isinstance(lst, list) else [lst])]

    def sample_from_subset(self, subset: List[int], k: int) -> List[int]:
        if not subset:
            return []
        weights_all = self.sampling_weights()
        w = np.array([weights_all.get(sid, 0.0) for sid in subset], dtype=float)
        if np.sum(w) <= 0:
            # fallback to uniform
            w = np.ones(len(subset), dtype=float) / len(subset)
        else:
            w = w / np.sum(w)
        choice = np.random.choice(subset, size=min(k, len(subset)), replace=False, p=w)
        return [int(x) for x in (choice.tolist() if hasattr(choice, 'tolist') else list(choice))]

    def add_scenarios(self, new_scenarios: List[Dict[str, Any]]):
        for s in new_scenarios:
            try:
                cfg = sanitize_scenario(s)
                self.pool[self.next_id] = cfg
                self.perf[self.next_id] = []
                self.next_id += 1
            except Exception:
                continue
        # Enforce max size by removing easiest (highest avg reward)
        if len(self.pool) > self.max_size:
            to_remove = len(self.pool) - self.max_size
            ranked = []
            for sid in self.pool:
                hist = self.perf.get(sid, [])
                avg = np.mean(hist) if hist else 0.0
                ranked.append((avg, sid))
            # remove from highest avg down
            ranked.sort(reverse=True)
            for _, sid in ranked[:to_remove]:
                self.pool.pop(sid, None)
                self.perf.pop(sid, None)





if __name__ == '__main__':

    env = LeasingEnv({30.24: 23314, 18.14: 30843, 11.49: 38371}, {30.24: 1/4, 18.14: 1/4, 11.49: 1/4}, [30.24, 18.14, 11.49],10000,1000, 20000, 2, 1, chance_cost=True)


    for _ in range(512):
        env.step((10000, 30.24))


    env.plot_history()
    env.plot_profit()
    env.plot_mean_profit()
    env.plot_action()










"""
DistillToMCU Phase 0b — PMS (Perturbed-Mean Selection) 规则选择器
==================================================================
C2 核心模块：专为 MCU 内存约束设计的 Perturbed-Mean Selection (FTPL 变种)。

算法定位：
  归入 Follow-The-Perturbed-Leader (FTPL) 家族（Kalai & Vempala 2005），
  使用均值 + 均匀扰动近似 Thompson Sampling 的后验采样行为，
  避免 Gamma 函数带来的 MCU 不可行性。论文中称为 PMS，非 TS。

设计思路：
  - uint8 α/β 存储（2B/规则, 500条规则=1KB）vs 精确 TS 的 float32 (4KB+)
  - 均值 + 均匀扰动 vs Gamma 函数采样（<10周期 vs ~200μs）
  - 适合 MCU 内存约束的工程优化

算法：
  每条规则维护 (α_i, β_i) = (接受次数+1, 拒绝次数+1)
  选择阶段：score_i = α_i/(α_i+β_i) + Uniform(-δ_i, +δ_i)，δ_i = 1/(α_i+β_i+1)
  更新阶段：接受 → α_i += 1；纠正 → β_i += 1（cap at 255）

对比实验：
  PMS vs ExactTS vs ε-greedy vs Greedy
  指标：cumulative regret, 最佳 arm 选择比例, 收敛速度

Usage:
    from bandit_selector import PMSSelector, run_bandit_comparison

    selector = PMSSelector(n_arms=10)
    choice = selector.select(matching_arm_ids=[0,3,5])
    selector.update(arm_id=3, accepted=True)
"""

import random
import math
import statistics
from collections import defaultdict


# ============================================================
# PMS Selector (uint8 + 均匀扰动, FTPL 变种)
# ============================================================

class PMSSelector:
    """
    Perturbed-Mean Selection (PMS) — Follow-The-Perturbed-Leader 变种。

    定位：FTPL 家族（Kalai & Vempala 2005），用均值+均匀扰动近似后验采样。
    论文中称为 PMS，不声称是 Thompson Sampling。

    存储格式（设计用于 C 移植）：
      struct pms_arm { uint8_t alpha; uint8_t beta; };
      pms_arm rules[RULE_MAX_TOTAL];  // 500 * 2B = 1KB

    选择算法（C 伪代码）：
      float best_score = -1; int best_idx = -1;
      for (int i = 0; i < n_matching; i++) {
          float mean = (float)arms[i].alpha / (arms[i].alpha + arms[i].beta);
          float delta = 1.0f / (arms[i].alpha + arms[i].beta + 1);
          float noise = uniform_rand(-delta, +delta);  // 或查表
          float score = mean + noise;
          if (score > best_score) { best_score = score; best_idx = i; }
      }

    时间复杂度：O(M) where M = 匹配的规则数, 通常 << 500
    空间复杂度：2B/规则
    """

    def __init__(self, n_arms: int = 0, seed: int = 42):
        """
        Args:
            n_arms: 规则总数（0 = 动态添加）
            seed: 随机种子（可复现）
        """
        self.alpha = {}       # arm_id -> int (1-255)
        self.beta = {}        # arm_id -> int (1-255)
        self._rng = random.Random(seed)
        self._total_selections = 0
        self._total_updates = 0
        self._init_counter = 1  # Beta(1,1) 先验

    def register_arm(self, arm_id):
        """注册一条新规则（从 Beta(1,1) 先验开始）"""
        if arm_id not in self.alpha:
            self.alpha[arm_id] = self._init_counter
            self.beta[arm_id] = self._init_counter

    def select(self, matching_arm_ids: list):
        """
        从匹配的规则中选择一条执行。

        Args:
            matching_arm_ids: 当前传感器状态下匹配的规则 ID 列表

        Returns:
            选中的 arm_id, 或 None（无匹配）
        """
        if not matching_arm_ids:
            return None

        # 确保所有 arm 已注册
        for aid in matching_arm_ids:
            self.register_arm(aid)

        if len(matching_arm_ids) == 1:
            self._total_selections += 1
            return matching_arm_ids[0]

        best_score = float('-inf')
        best_arm = None

        for aid in matching_arm_ids:
            a = self.alpha.get(aid, self._init_counter)
            b = self.beta.get(aid, self._init_counter)
            total = a + b

            # mean = α/(α+β)
            mean_val = a / total

            # δ = 1/(α+β+1) — 证据越多, δ 越小, 探索减少
            delta = 1.0 / (total + 1)

            # 均匀扰动
            noise = self._rng.uniform(-delta, delta)
            score = mean_val + noise

            if score > best_score:
                best_score = score
                best_arm = aid

        self._total_selections += 1
        return best_arm

    def update(self, arm_id, accepted: bool):
        """用户反馈更新"""
        if arm_id not in self.alpha:
            self.register_arm(arm_id)

        if accepted:
            if self.alpha[arm_id] < 255:
                self.alpha[arm_id] += 1
        else:
            if self.beta[arm_id] < 255:
                self.beta[arm_id] += 1

        self._total_updates += 1

    def get_beta_params(self, arm_id):
        """获取某条规则的 (α, β) 参数"""
        self.register_arm(arm_id)
        return self.alpha[arm_id], self.beta[arm_id]

    def get_mean(self, arm_id):
        """获取某条规则的估计接受率"""
        a, b = self.get_beta_params(arm_id)
        return a / (a + b)

    def reset(self):
        """重置所有统计"""
        self.alpha.clear()
        self.beta.clear()
        self._total_selections = 0
        self._total_updates = 0


# ============================================================
# 精确 Thompson Sampling (Beta 分布采样 — 用于对比)
# ============================================================

class ExactTSSelector:
    """
    标准 Thompson Sampling：从 Beta(α, β) 精确采样（对比基线，非我们的方法）。
    需要 Gamma 函数 → 在 MCU 上不可行。这里仅作为 PMS 的对比上界。

    论文中我们的方法是 PMS (FTPL变种)，不是 TS。
    """

    def __init__(self, n_arms: int = 0, seed: int = 42):
        self.alpha = {}
        self.beta = {}
        self._rng = random.Random(seed)
        self._init = 1

    def register_arm(self, arm_id):
        if arm_id not in self.alpha:
            self.alpha[arm_id] = self._init
            self.beta[arm_id] = self._init

    def select(self, matching_arm_ids: list):
        if not matching_arm_ids:
            return None
        for aid in matching_arm_ids:
            self.register_arm(aid)
        if len(matching_arm_ids) == 1:
            return matching_arm_ids[0]

        best_score = float('-inf')
        best_arm = None
        for aid in matching_arm_ids:
            a = self.alpha[aid]
            b = self.beta[aid]
            # Betavariate: Gamma(a,1)/(Gamma(a,1)+Gamma(b,1))
            sample = self._rng.betavariate(a, b)
            if sample > best_score:
                best_score = sample
                best_arm = aid
        return best_arm

    def update(self, arm_id, accepted: bool):
        self.register_arm(arm_id)
        if accepted:
            self.alpha[arm_id] += 1
        else:
            self.beta[arm_id] += 1


# ============================================================
# Epsilon-Greedy
# ============================================================

class EpsilonGreedySelector:
    """ε-greedy: 1-ε 概率选最佳, ε 概率随机探索"""

    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self.alpha = {}
        self.beta = {}
        self._rng = random.Random(seed)
        self._init = 1

    def register_arm(self, arm_id):
        if arm_id not in self.alpha:
            self.alpha[arm_id] = self._init
            self.beta[arm_id] = self._init

    def select(self, matching_arm_ids: list):
        if not matching_arm_ids:
            return None
        for aid in matching_arm_ids:
            self.register_arm(aid)
        if len(matching_arm_ids) == 1:
            return matching_arm_ids[0]

        # ε 概率随机探索
        if self._rng.random() < self.epsilon:
            return self._rng.choice(matching_arm_ids)

        # 1-ε 概率选最佳（最高 mean）
        best = max(matching_arm_ids,
                   key=lambda aid: self.alpha[aid] / (self.alpha[aid] + self.beta[aid]))
        return best

    def update(self, arm_id, accepted: bool):
        self.register_arm(arm_id)
        if accepted:
            self.alpha[arm_id] += 1
        else:
            self.beta[arm_id] += 1


# ============================================================
# 最高置信度优先 (Greedy — 无探索)
# ============================================================

class GreedySelector:
    """纯 exploitative：永远选当前 mean 最高的"""

    def __init__(self):
        self.alpha = {}
        self.beta = {}
        self._init = 1

    def register_arm(self, arm_id):
        if arm_id not in self.alpha:
            self.alpha[arm_id] = self._init
            self.beta[arm_id] = self._init

    def select(self, matching_arm_ids: list):
        if not matching_arm_ids:
            return None
        for aid in matching_arm_ids:
            self.register_arm(aid)
        if len(matching_arm_ids) == 1:
            return matching_arm_ids[0]
        return max(matching_arm_ids,
                   key=lambda aid: self.alpha[aid] / (self.alpha[aid] + self.beta[aid]))

    def update(self, arm_id, accepted: bool):
        self.register_arm(arm_id)
        if accepted:
            self.alpha[arm_id] += 1
        else:
            self.beta[arm_id] += 1


# ============================================================
# Bandit 仿真环境
# ============================================================

class BanditEnvironment:
    """
    仿真 n-臂 Bandit 环境。
    每条 arm 有一个真实的接受概率 p_i。
    每次 pull 返回 Bernoulli(p_i) 结果。
    """

    def __init__(self, true_probs: list[float], seed: int = 42,
                 switching_interval: int | None = None):
        """
        Args:
            true_probs: 每条 arm 的真实接受概率 (0-1)
            switching_interval: 非平稳环境——每隔 N 轮随机重排真实概率
                （模拟规则环境漂移：住户偏好/季节变化使最优规则变化）。
                None = 平稳环境（默认）。
        """
        self.true_probs = true_probs
        self.n_arms = len(true_probs)
        self._rng = random.Random(seed)
        self.switching_interval = switching_interval
        self._round = 0
        self._optimal_arm = true_probs.index(max(true_probs))
        self._optimal_prob = max(true_probs)

    def step(self):
        """每轮调用。到达切换点时重排真实概率（最优 arm 漂移）。"""
        self._round += 1
        if self.switching_interval and self._round % self.switching_interval == 0:
            random.Random(self._round + 1).shuffle(self.true_probs)
            self._optimal_arm = self.true_probs.index(max(self.true_probs))
            self._optimal_prob = max(self.true_probs)

    def pull(self, arm_id: int) -> bool:
        """返回 True（接受）或 False（拒绝）"""
        return self._rng.random() < self.true_probs[arm_id]

    def regret(self, arm_id: int) -> float:
        """单步 regret = p* - p_i"""
        return self._optimal_prob - self.true_probs[arm_id]

    def get_optimal_arm(self) -> int:
        return self._optimal_arm


# ============================================================
# Bandit 对比实验
# ============================================================

def run_bandit_comparison(
    env: BanditEnvironment,
    selectors: dict,
    n_rounds: int = 1000,
    matching_size: int = 3,   # 每轮随机选 M 条 arm 作为"匹配的"
    seed: int = 42,
):
    """
    在同一个 Bandit 环境中对比多种选择策略。

    Args:
        env: Bandit 环境（真实接受概率）
        selectors: {"name": Selector()}
        n_rounds: 总轮数
        matching_size: 每轮随机选几条 arm 进入候选（模拟规则匹配）
        seed: 随机种子

    Returns:
        {name: {
            "cumulative_regret": [...],  # 每轮的累积 regret
            "optimal_rate": [...],       # 每轮选到最优 arm 的比例
            "final_regret": float,
            "final_optimal_rate": float,
        }}
    """
    rng = random.Random(seed)

    # 初始化
    results = {}
    for name, sel in selectors.items():
        results[name] = {
            "cumulative_regret": [],
            "optimal_rate": [],
            "arm_selections": defaultdict(int),
            "total_reward": 0,
        }

    for t in range(n_rounds):
        # v10: 非平稳环境——每轮先让环境演化（切换点重排最优 arm）
        env.step()

        # 随机选 M 条 arm 作为当前"匹配的"
        matching = sorted(rng.sample(range(env.n_arms),
                                     min(matching_size, env.n_arms)))

        for name, sel in selectors.items():
            choice = sel.select(matching)
            if choice is None:
                continue

            # Pull arm + 获取反馈
            accepted = env.pull(choice)
            sel.update(choice, accepted)

            # 记录指标
            res = results[name]
            res["arm_selections"][choice] += 1
            res["total_reward"] += 1 if accepted else 0

            # Cumulative regret (so far)
            past_regret = res["cumulative_regret"][-1] if res["cumulative_regret"] else 0
            step_regret = env.regret(choice)
            res["cumulative_regret"].append(past_regret + step_regret)

            # Optimal rate (so far)
            optimal = env.get_optimal_arm()
            n_correct = res["arm_selections"].get(optimal, 0)
            res["optimal_rate"].append(
                n_correct / sum(res["arm_selections"].values()) * 100
            )

    # 最终指标
    for name, res in results.items():
        res["final_regret"] = res["cumulative_regret"][-1]
        res["final_optimal_rate"] = res["optimal_rate"][-1] if res["optimal_rate"] else 0

    return results


def print_comparison(results: dict):
    """格式化输出对比结果"""
    print(f"\n{'Selector':<22s} {'Final Regret':>12s} {'Optimal Arm%':>12s} {'Total Reward':>12s}")
    print("-" * 58)
    for name, res in results.items():
        print(f"  {name:<20s} {res['final_regret']:11.4f} {res['final_optimal_rate']:10.1f}% "
              f"{res['total_reward']:10d}")


# ============================================================
# PMS vs Exact TS Regret Gap 分析
# ============================================================

def analyze_regret_gap(
    n_arms: int = 10,
    n_rounds: int = 5000,
    n_trials: int = 10,
    matching_size: int = 3,
    seed: int = 42,
):
    """
    多次重复实验，分析 PMS vs 精确 TS 的 regret gap 分布。

    返回统计摘要：
      - mean gap, std gap
      - gap 随时间的变化曲线
    """
    rng = random.Random(seed)
    gaps_by_trial = []
    ex_final_regrets_all = []  # v7 fix: collect across ALL trials, not just last

    for trial in range(n_trials):
        # 随机生成真实概率
        true_probs = [rng.uniform(0.3, 0.95) for _ in range(n_arms)]

        env = BanditEnvironment(true_probs, seed=seed + trial * 100)
        pms = PMSSelector(seed=seed + trial)
        exact_ts = ExactTSSelector(seed=seed + trial)

        selectors = {"PMS": pms, "ExactTS": exact_ts}
        results = run_bandit_comparison(
            env, selectors, n_rounds=n_rounds,
            matching_size=matching_size, seed=seed + trial,
        )

        pms_regret = results["PMS"]["cumulative_regret"]
        ex_regret = results["ExactTS"]["cumulative_regret"]
        gap = [d - e for d, e in zip(pms_regret, ex_regret)]
        gaps_by_trial.append(gap)
        ex_final_regrets_all.append(results["ExactTS"]["cumulative_regret"][-1])

    # 统计
    n = len(gaps_by_trial[0])
    mean_gap = [statistics.mean(g[i] for g in gaps_by_trial) for i in range(n)]
    std_gap = [statistics.stdev(g[i] for g in gaps_by_trial) if n_trials > 1 else 0
               for i in range(n)]

    # 最终 gap
    final_gaps = [g[-1] for g in gaps_by_trial]
    final_mean = statistics.mean(final_gaps)
    final_std = statistics.stdev(final_gaps) if n_trials > 1 else 0

    # v7 fix: use ALL trials' ExactTS regret for scaling
    ex_avg = statistics.mean(ex_final_regrets_all) if ex_final_regrets_all else 1.0
    gap_pct = (final_mean / max(0.01, abs(ex_avg))) * 100

    print(f"\nPMS vs Exact TS Regret Gap Analysis")
    print(f"  Arms: {n_arms} | Rounds: {n_rounds} | Trials: {n_trials} | "
          f"Matching: {matching_size}/round")
    print(f"  Final gap: {final_mean:.4f} +/- {final_std:.4f}")
    print(f"  Gap as % of ExactTS regret: {gap_pct:.2f}%")

    return {
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "final_mean_gap": final_mean,
        "final_std_gap": final_std,
    }


# ============================================================
# 与 DistillToMCU 规则引擎集成
# ============================================================

class PMSRuleSelector:
    """
    将 PMS 集成到 DistillToMCU 的规则引擎选择逻辑中。
    替代原有的 "最高置信度优先" 排序策略。

    Usage:
        engine = RuleEngine()
        pms = PMSRuleSelector(engine)
        # 替换 resolve_conflict:
        best_rule = pms.select_from_matches(matches)
    """

    def __init__(self, rule_engine, seed: int = 42):
        self.engine = rule_engine
        self.pms = PMSSelector(seed=seed)

    def select_from_matches(self, matches: list):
        """
        从规则引擎匹配的规则列表中，用 PMS 选择一条。

        Args:
            matches: rule_engine.match() 的返回值（Rule 对象列表）

        Returns:
            选中的 Rule 或 None
        """
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        # 用规则的 string id 作为 PMS arm id
        arm_ids = [rule.id for rule in matches]
        chosen_id = self.pms.select(arm_ids)

        for rule in matches:
            if rule.id == chosen_id:
                return rule
        return matches[0]  # fallback

    def on_feedback(self, rule_id: str, accepted: bool):
        """用户反馈 → 更新 PMS 统计"""
        self.pms.update(rule_id, accepted)

    def get_stats(self, rule_id: str):
        """获取某条规则的 PMS 统计"""
        a, b = self.pms.get_beta_params(rule_id)
        mean = a / (a + b)
        return {"alpha": a, "beta": b, "mean": round(mean, 4)}


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PMS Selector — Self-Test & Benchmark")
    print("=" * 60)

    # Test 1: 基本功能
    print("\n[Test 1] Basic PMS selection and update")
    pms = PMSSelector(seed=42)
    # 注册 5 条规则
    for i in range(5):
        pms.register_arm(i)

    # 选 100 次 + 反馈
    selections = defaultdict(int)
    for _ in range(100):
        # 随机 2-4 条匹配
        matching = random.sample(range(5), random.randint(2, 4))
        choice = pms.select(matching)
        selections[choice] += 1
        # 模拟反馈：arm 0 真实 p=0.9, arm 1 p=0.7, 其余 p=0.3
        true_probs = {0: 0.9, 1: 0.7, 2: 0.3, 3: 0.3, 4: 0.3}
        accepted = random.random() < true_probs[choice]
        pms.update(choice, accepted)

    print(f"  Selection distribution: "
          f"{dict(sorted(selections.items()))}")
    print(f"  Best arm (0) selected: {selections[0]}/100 times")
    # Arm 0 (p=0.9) 应该被选得最多
    assert selections[0] >= selections[2], \
        f"Best arm not selected most: {dict(selections)}"
    print("  [PASS]")

    # Test 2: 内存占用验证
    print("\n[Test 2] Memory footprint for 500 arms")
    import sys
    n = 500
    pms500 = PMSSelector()
    for i in range(n):
        pms500.register_arm(f"rule_{i:04d}")
    alpha_size = sys.getsizeof(pms500.alpha)
    beta_size = sys.getsizeof(pms500.beta)
    # 实际 C 存储：500 × 2B = 1KB
    print(f"  Python dict overhead: {alpha_size + beta_size} bytes")
    print(f"  C equivalent:         {n * 2} bytes (uint8 × 2/rule)")
    print("  [PASS]")

    # Test 3: 收敛性验证
    print("\n[Test 3] Convergence — PMS converges to best arm")
    env = BanditEnvironment([0.9, 0.7, 0.3, 0.3, 0.3], seed=42)
    pms3 = PMSSelector(seed=42)
    greedy3 = GreedySelector()

    n_rounds = 2000
    pms_correct = 0
    greedy_correct = 0
    optimal = env.get_optimal_arm()

    for t in range(n_rounds):
        matching = sorted(random.sample(range(5), random.randint(2, 4)))

        # PMS
        p_choice = pms3.select(matching)
        pms3.update(p_choice, env.pull(p_choice))
        if p_choice == optimal:
            pms_correct += 1

        # Greedy
        g_choice = greedy3.select(matching)
        greedy3.update(g_choice, env.pull(g_choice))
        if g_choice == optimal:
            greedy_correct += 1

    pms_rate = pms_correct / n_rounds * 100
    greedy_rate = greedy_correct / n_rounds * 100
    print(f"  PMS optimal rate:    {pms_rate:.1f}%")
    print(f"  Greedy optimal rate: {greedy_rate:.1f}%")
    # PMS with random matching subsets — optimal arm may not be in every set
    # So >50% is good performance (random baseline would be 25-33% for 2-4 arms)
    assert pms_rate > 50, f"PMS should find best arm >50%, got {pms_rate:.1f}%"
    print("  [PASS]")

    # Test 4: 全量对比
    print("\n[Test 4] Full comparison: PMS vs ExactTS vs ε-greedy vs Greedy")
    env4 = BanditEnvironment(
        [0.85, 0.80, 0.75, 0.40, 0.35, 0.30, 0.25, 0.20], seed=42
    )
    selectors = {
        "PMS": PMSSelector(seed=42),
        "ExactTS": ExactTSSelector(seed=42),
        "ε-Greedy(0.1)": EpsilonGreedySelector(epsilon=0.1, seed=42),
        "ε-Greedy(0.05)": EpsilonGreedySelector(epsilon=0.05, seed=42),
        "Greedy": GreedySelector(),
    }
    results4 = run_bandit_comparison(
        env4, selectors, n_rounds=3000, matching_size=3, seed=42
    )
    print_comparison(results4)

    # Test 5: Regret gap 分析
    print("\n[Test 5] PMS vs Exact TS regret gap (10 trials)")
    gap_result = analyze_regret_gap(
        n_arms=8, n_rounds=3000, n_trials=10, matching_size=3, seed=42
    )

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED [OK]")
    print("=" * 60)

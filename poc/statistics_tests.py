"""
DistillToMCU — 统计检验工具（纯 Python，无 scipy 依赖）
========================================================
论文要求的统计检验：
  1. Friedman test（多基线多数据集的主效应检验）
  2. Nemenyi post-hoc（成对显著性，Critical Difference）
  3. Bootstrap 95% CI（Autonomy Rate / Cloud Call Reduction）

AGENTS.md 要求 Q1/Q2 期刊提交 Friedman + Nemenyi + Bootstrap CI。
scipy 不可用时本模块自包含实现（不完全 Gamma 函数来自 Numerical Recipes，
仅用于 p 值近似，结果与 scipy 差异在 1e-4 量级）。
"""

import math
import random
import statistics


# ============ 不完全 Gamma（Numerical Recipes gammp/gammq） ============

def _gser(a, x):
    ITMAX = 200
    EPS = 3e-12
    if x <= 0.0:
        return 0.0
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(ITMAX):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * EPS:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x):
    ITMAX = 200
    EPS = 3e-12
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammq(a, x):
    """上尾不完全 Gamma Q(a,x) = P(X > x)，X ~ Gamma(a)。"""
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x2, df):
    """χ² 分布的生存函数（右尾概率）。"""
    if x2 <= 0:
        return 1.0
    return gammq(df / 2.0, x2 / 2.0)


# ============ Friedman Test ============

def friedman_test(data, higher_is_better=True):
    """
    Friedman 检验。
    data: list of blocks（每个 block 是 k 个处理的得分列表），shape (n_blocks, k)。
    higher_is_better: 得分是否越大越好（AR/AGREE 等）。
        默认 True：rank 1 = 最高分（最佳）；False：rank 1 = 最低分。
        v10.5d 修复：此前按升序排名，导致 0% 的基线（B1/B2）在
        AR/AGREE 检验中被错误地排为"最佳"（rank 1）。
    返回 {chi2, df, p_value, mean_ranks}。
    """
    n = len(data)
    k = len(data[0])
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 blocks and 2 treatments")

    # 每个 block 内独立排名（并列取平均）
    ranks = []
    for block in data:
        order = sorted(range(k), key=lambda j: block[j],
                       reverse=higher_is_better)
        r = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j + 1 < k and block[order[j + 1]] == block[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based
            for t in range(i, j + 1):
                r[order[t]] = avg_rank
            i = j + 1
        ranks.append(r)

    sum_ranks = [sum(r[j] for r in ranks) for j in range(k)]
    mean_ranks = [s / n for s in sum_ranks]

    # Friedman χ² 统计量（Iman-Davenport 修正前）
    ss = sum(s * s for s in sum_ranks)
    chi2 = (12.0 / (n * k * (k + 1))) * ss - 3.0 * n * (k + 1)

    chi2_p = chi2_sf(chi2, k - 1)

    return {
        "n_blocks": n,
        "k_treatments": k,
        "chi2": round(chi2, 4),
        "df": k - 1,
        "p_value": round(chi2_p, 6),
        "mean_ranks": [round(x, 4) for x in mean_ranks],
        "significant": chi2_p < 0.05,
    }


# ============ Nemenyi Post-hoc ============

# Studentized range 临界值 q_alpha(k)，自由度 ∞（Nemenyi 标准表）
NEMENYI_Q_ALPHA_005 = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def nemenyi_critical_difference(k, n_blocks, alpha=0.05):
    """Nemenyi 临界差 CD = q_α(k) * sqrt(k(k+1)/(12n))。"""
    q = NEMENYI_Q_ALPHA_005.get(k, 3.164)
    return q * math.sqrt(k * (k + 1) / (12.0 * n_blocks))


def nemenyi_posthoc(mean_ranks, n_blocks, treatment_names):
    """
    返回成对平均排名差和是否超过 CD。
    mean_ranks: 1-based 平均排名列表；treatment_names: 处理名列表。
    """
    k = len(mean_ranks)
    cd = nemenyi_critical_difference(k, n_blocks)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            pairs.append({
                "pair": f"{treatment_names[i]} vs {treatment_names[j]}",
                "rank_diff": round(diff, 4),
                "significant": diff > cd,
            })
    return {"critical_difference": round(cd, 4), "pairs": pairs}


# ============ Bootstrap CI ============

def bootstrap_ci(data, n_bootstrap=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI（均值）。"""
    rng = random.Random(seed)
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(data) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(n_bootstrap * alpha / 2)]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    return lo, hi


if __name__ == "__main__":
    # 自测：已知数据 (F 检验教材例)
    demo = [
        [9, 8, 7, 6],
        [8, 7, 6, 5],
        [7, 6, 5, 4],
        [6, 5, 4, 3],
        [5, 4, 3, 2],
    ]
    ft = friedman_test(demo)
    print("Friedman:", ft)
    nm = nemenyi_posthoc(ft["mean_ranks"], ft["n_blocks"], ["A", "B", "C", "D"])
    print("Nemenyi CD:", nm["critical_difference"])
    print("Pairwise significant:", [p for p in nm["pairs"] if p["significant"]])

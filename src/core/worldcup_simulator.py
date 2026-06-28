"""
蒙特卡洛模拟引擎 + 权重优化框架
==============================
基于 worldcup_model.py 的核心预测器，进行完整的赛事模拟和权重优化。

模块结构：
  - TournamentSimulator: 10,000次完整赛事模拟
  - WeightOptimizer: 基于历史数据的权重优化
"""

import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

# 导入核心模型
from worldcup_model import (
    WorldCupPredictor, ModelWeights, FeatureExtractor, TeamData, load_2026_teams
)


# ============================================================
# 一、48队分组结构 (2026实际分组)
# ============================================================

GROUPS_2026 = {
    'A': ['MEX', 'RSA', 'KOR', 'CZE'],
    'B': ['CAN', 'BIH', 'QAT', 'SUI'],
    'C': ['BRA', 'MAR', 'HAI', 'SCO'],
    'D': ['USA', 'PAR', 'AUS', 'TUR'],
    'E': ['GER', 'CUW', 'CIV', 'ECU'],
    'F': ['NED', 'JPN', 'TUN', 'SWE'],
    'G': ['BEL', 'EGY', 'IRN', 'NZL'],
    'H': ['ESP', 'CPV', 'KSA', 'URU'],
    'I': ['FRA', 'SEN', 'IRQ', 'NOR'],
    'J': ['ARG', 'ALG', 'AUT', 'JOR'],
    'K': ['POR', 'COD', 'UZB', 'COL'],
    'L': ['ENG', 'CRO', 'GHA', 'PAN'],
}


# ═══════════════════════════════════════════════════════════
# 小组赛 → 场馆映射 (基于实际赛程)
# ═══════════════════════════════════════════════════════════
GROUP_MATCH_VENUES = {
    ('A', 0): 'Mexico City', ('A', 1): 'Guadalajara',
    ('B', 0): 'San Francisco Bay Area', ('B', 1): 'Toronto',
    ('C', 0): 'New York / New Jersey', ('C', 1): 'Boston',
    ('D', 0): 'Los Angeles', ('D', 1): 'Vancouver',
    ('E', 0): 'Houston', ('E', 1): 'Philadelphia',
    ('F', 0): 'Dallas', ('F', 1): 'Kansas City',
    ('G', 0): 'Seattle', ('G', 1): 'Los Angeles',
    ('H', 0): 'Atlanta', ('H', 1): 'Miami',
    ('I', 0): 'New York / New Jersey', ('I', 1): 'Boston',
    ('J', 0): 'Kansas City', ('J', 1): 'San Francisco Bay Area',
    ('K', 0): 'Dallas', ('K', 1): 'Mexico City',
    ('L', 0): 'Dallas', ('L', 1): 'Toronto',
}
DEFAULT_VENUES = {
    'A': 'Guadalajara', 'B': 'Vancouver', 'C': 'Miami',
    'D': 'Los Angeles', 'E': 'Houston', 'F': 'Dallas',
    'G': 'Seattle', 'H': 'Atlanta', 'I': 'Boston',
    'J': 'Kansas City', 'K': 'Mexico City', 'L': 'Toronto',
}

def get_match_venue(group: str, match_index: int) -> str:
    return GROUP_MATCH_VENUES.get((group, match_index), DEFAULT_VENUES.get(group, 'Dallas'))

# ============================================================
# 二、小组赛模拟器
# ============================================================

@dataclass
class GroupStanding:
    """小组积分榜"""
    team: str
    pts: int = 0
    gf: int = 0
    ga: int = 0
    gd: int = 0


class GroupSimulator:
    """模拟一个小组的完整赛程（场馆感知版）"""

    def __init__(self, predictor: WorldCupPredictor, teams: Dict[str, TeamData]):
        self.predictor = predictor
        self.teams_dict = teams

    def simulate_group(self, group_teams: List[str], group_name: str = "") -> List[GroupStanding]:
        """模拟单组3轮比赛，返回积分榜"""
        standings = {code: GroupStanding(team=code) for code in group_teams}
        team_codes = group_teams

        # 6场对阵: (0,1) round1, (2,3) round1, (0,2) round2, (1,3) round2, (0,3) round3, (1,2) round3
        matchups = [(0, 1, 0), (2, 3, 1), (0, 2, 2), (1, 3, 3), (0, 3, 4), (1, 2, 5)]

        for i, j, match_idx in matchups:
            home_code = team_codes[i]
            away_code = team_codes[j]
            home = self.teams_dict[home_code]
            away = self.teams_dict[away_code]

            # 场馆感知预测
            venue_key = get_match_venue(group_name, match_idx)
            eg_h, eg_a = self.predictor.compute_expected_goals(
                home, away, home_code=home_code, away_code=away_code,
                venue_data=__import__('worldcup_venues').VENUES.get(venue_key)
            )
            goals_h = int(np.random.poisson(eg_h))
            goals_a = int(np.random.poisson(eg_a))

            standings[home_code].gf += goals_h
            standings[home_code].ga += goals_a
            standings[away_code].gf += goals_a
            standings[away_code].ga += goals_h

            if goals_h > goals_a:
                standings[home_code].pts += 3
            elif goals_h < goals_a:
                standings[away_code].pts += 3
            else:
                standings[home_code].pts += 1
                standings[away_code].pts += 1

        for st in standings.values():
            st.gd = st.gf - st.ga

        sorted_standings = sorted(standings.values(),
                                  key=lambda s: (s.pts, s.gd, s.gf), reverse=True)
        return sorted_standings


# ============================================================
# 三、淘汰赛模拟器
# ============================================================

class KnockoutSimulator:
    """模拟淘汰赛阶段"""

    def __init__(self, predictor: WorldCupPredictor, teams: Dict[str, TeamData]):
        self.predictor = predictor
        self.teams = teams

    def simulate_match(self, team_a: str, team_b: str) -> str:
        """模拟一场淘汰赛，返回胜者（含加时+点球逻辑）"""
        t_a = self.teams[team_a]
        t_b = self.teams[team_b]

        # 直接使用期望进球采样（跳过完整predict_match以提升性能）
        eg_a, eg_b = self.predictor.compute_expected_goals(t_a, t_b)

        goals_a = np.random.poisson(eg_a)
        goals_b = np.random.poisson(eg_b)

        if goals_a > goals_b:
            return team_a
        elif goals_b > goals_a:
            return team_b
        else:
            # 平局 → 点球大战
            p_a = 0.50 + 0.03 * (t_a.opr_gk - t_b.opr_gk) / 100
            p_a = max(0.35, min(0.65, p_a))
            return team_a if random.random() < p_a else team_b

    def simulate_round(self, teams: List[str]) -> List[str]:
        """模拟一轮淘汰赛 (16 → 8, 8 → 4, 4 → 2, 2 → 1)"""
        winners = []
        for i in range(0, len(teams), 2):
            winner = self.simulate_match(teams[i], teams[i + 1])
            winners.append(winner)
        return winners


# ============================================================
# 四、完整赛事蒙特卡洛引擎
# ============================================================

class TournamentSimulator:
    """
    完整赛事蒙特卡洛模拟器

    用法：
        sim = TournamentSimulator()
        results = sim.run_simulations(n=10000)
        # results 包含夺冠概率、出线概率等
    """

    # 2026世界杯淘汰赛对阵模板
    # (A1, C2, E1, G2, ...) vs (B2, D1, F2, H1, ...)  ← 传统对称结构
    # 实际上2026有32队出线（24队小组前二+8个最佳第三），对阵更复杂
    # 这里简化处理：12组前二(24队) + 8个最佳第三(8队) → 32强

    R32_MATCHUPS = [
        # (A1, B2), (C1, D2), (E1, F2), (G1, H2), (I1, J2), (K1, L2)
        # (B1, A2), (D1, C2), (F1, E2), (H1, G2), (J1, I2), (L1, K2)
        # 最佳第三填充
    ]

    def __init__(self, predictor: WorldCupPredictor = None):
        self.predictor = predictor or WorldCupPredictor()
        self.teams = load_2026_teams()
        self.group_sim = GroupSimulator(self.predictor, self.teams)
        self.knockout_sim = KnockoutSimulator(self.predictor, self.teams)

    def run_simulations(self, n: int = 10000, verbose: bool = True) -> Dict:
        """
        运行 n 次完整赛事模拟

        返回：
        {
            'champion_probs': {team: prob},
            'final_probs': {team: prob},
            'semifinal_probs': {team: prob},
            'knockout_probs': {team: prob},
            'group_3rd_qualify': {team: prob},  # 最佳第三出线概率
        }
        """
        champion_count = defaultdict(int)
        final_count = defaultdict(int)
        semifinal_count = defaultdict(int)
        knockout_count = defaultdict(int)
        group_exit_count = defaultdict(int)

        for sim_id in range(n):
            if verbose and sim_id % 2000 == 0:
                print(f"  模拟进度: {sim_id}/{n} ({sim_id/n*100:.0f}%)")

            # 1) 小组赛
            group_results = {}
            third_place_teams = []
            for g_name, g_teams in GROUPS_2026.items():
                standings = self.group_sim.simulate_group(g_teams, g_name)
                group_results[g_name] = standings

                # 前两名晋级
                for st in standings[:2]:
                    knockout_count[st.team] += 1
                # 第三名进入候选池
                third_place_teams.append((standings[2], g_name))

            # 2) 选出8个最佳第三名
            third_sorted = sorted(third_place_teams,
                                  key=lambda x: (x[0].pts, x[0].gd, x[0].gf), reverse=True)
            best_thirds = [tp[0].team for tp in third_sorted[:8]]
            # 没有晋级的第三名和第四名算小组出局
            eliminated_thirds = [tp[0].team for tp in third_sorted[8:]]
            for g_name, g_teams in GROUPS_2026.items():
                for st in group_results[g_name]:
                    if st.team in eliminated_thirds:
                        group_exit_count[st.team] += 1
                    elif st == group_results[g_name][3]:
                        group_exit_count[st.team] += 1

            # 3) 构建32强并模拟淘汰赛
            # 简化对阵：按组序排列，A1→B2等 (真实对阵表此处可细化)
            # 由于2026实际对阵规则复杂（48队特有），这里用随机配对模拟
            r32_teams = self._build_r32(group_results, best_thirds)

            # 逐轮淘汰
            r16_teams = self.knockout_sim.simulate_round(r32_teams)
            qf_teams = self.knockout_sim.simulate_round(r16_teams)
            sf_teams = self.knockout_sim.simulate_round(qf_teams)

            for t in sf_teams:
                semifinal_count[t] += 1

            final_teams = self.knockout_sim.simulate_round(sf_teams)
            for t in final_teams:
                final_count[t] += 1

            champion = self.knockout_sim.simulate_round(final_teams)[0]
            champion_count[champion] += 1

        if verbose:
            print(f"  模拟完成! 共 {n} 次模拟")

        return {
            'champion_probs': {t: c / n for t, c in champion_count.items()},
            'final_probs': {t: c / n for t, c in final_count.items()},
            'semifinal_probs': {t: c / n for t, c in semifinal_count.items()},
            'knockout_probs': {t: c / n for t, c in knockout_count.items()},
            'group_exit_probs': {t: c / n for t, c in group_exit_count.items()},
        }

    def _build_r32(self, group_results: Dict, best_thirds: List[str]) -> List[str]:
        """
        构建32强对阵表 (2026规则简化版)
        """
        # 简化：按组别顺序排列前两名
        r32 = []
        group_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

        # 取每个组前两名
        winners = []
        runners_up = []
        for g in group_order:
            standings = group_results[g]
            winners.append(standings[0].team)
            runners_up.append(standings[1].team)

        # 构建对阵: 上半区
        for i in range(0, 12, 2):
            r32.append(winners[i])       # A1, C1, E1, ...
            r32.append(runners_up[i+1])  # B2, D2, F2, ...
            r32.append(winners[i+1])     # B1, D1, F1, ...
            r32.append(runners_up[i])    # A2, C2, E2, ...

        # 填充8个最佳第三
        r32.extend(best_thirds)

        # shuffle以保证公平性
        random.shuffle(r32)

        return r32


# ============================================================
# 五、权重优化框架
# ============================================================

@dataclass
class MatchRecord:
    """历史比赛记录"""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    result: str  # 'H', 'D', 'A'


class WeightOptimizer:
    """
    基于历史数据的权重优化器

    使用随机搜索（可扩展为贝叶斯优化）在当前参数邻域内寻找更优解。
    损失函数：对数似然损失

    优化流程：
    1. 收集历史世界杯比赛记录
    2. 在当前权重附近采样候选权重
    3. 计算每个候选的损失
    4. 选择损失最小的作为下一轮权重
    5. 重复直到收敛
    """

    def __init__(self, predictor: WorldCupPredictor, teams: Dict[str, TeamData]):
        self.predictor = predictor
        self.teams = teams

    def compute_loss(self, matches: List[MatchRecord]) -> float:
        """计算对数似然损失"""
        total_loss = 0.0
        n = len(matches)

        for m in matches:
            if m.home_team not in self.teams or m.away_team not in self.teams:
                continue
            home = self.teams[m.home_team]
            away = self.teams[m.away_team]
            pred = self.predictor.predict_match(home, away)

            # soft clipping 防止 log(0)
            eps = 1e-8
            if m.result == 'H':
                prob = max(pred['p_home_win'] / 100, eps)
            elif m.result == 'A':
                prob = max(pred['p_away_win'] / 100, eps)
            else:
                prob = max(pred['p_draw'] / 100, eps)

            total_loss += math.log(prob)

        return -total_loss / n  # 负对数似然均值

    def random_search_step(self, matches: List[MatchRecord],
                           sigma: float = 0.01, n_trials: int = 50) -> ModelWeights:
        """一次随机搜索优化步骤"""
        current_loss = self.compute_loss(matches)
        best_weights = self.predictor.weights
        best_loss = current_loss

        for _ in range(n_trials):
            # 在当前权重附近随机扰动
            new_weights = self._perturb_weights(self.predictor.weights, sigma)
            self.predictor.weights = new_weights
            loss = self.compute_loss(matches)

            if loss < best_loss:
                best_loss = loss
                best_weights = new_weights

        self.predictor.weights = best_weights
        return best_weights

    def _perturb_weights(self, weights: ModelWeights, sigma: float) -> ModelWeights:
        """在权重附近添加高斯噪声"""
        new_w = ModelWeights()
        for field_name in vars(weights):
            val = getattr(weights, field_name)
            # 只扰动 float 字段
            if isinstance(val, float):
                new_val = val + np.random.normal(0, sigma)
                setattr(new_w, field_name, new_val)
            else:
                setattr(new_w, field_name, val)
        return new_w

    def optimize(self, matches: List[MatchRecord],
                 n_iterations: int = 20, sigma: float = 0.01,
                 verbose: bool = True) -> List[float]:
        """多轮优化"""
        losses = []
        for i in range(n_iterations):
            self.random_search_step(matches, sigma=sigma, n_trials=50)
            loss = self.compute_loss(matches)
            losses.append(loss)
            if verbose and i % 5 == 0:
                print(f"  优化轮次 {i+1}/{n_iterations}: 损失 = {loss:.4f}")
        return losses


# ============================================================
# 六、灵敏度分析
# ============================================================

def sensitivity_analysis(predictor: WorldCupPredictor, teams: Dict[str, TeamData],
                         test_match: Tuple[str, str] = ('ENG', 'CRO')):
    """分析各因子层对预测结果的贡献度"""
    home_code, away_code = test_match
    home = teams[home_code]
    away = teams[away_code]

    baseline = predictor.predict_match(home, away)
    pw = baseline['p_home_win']

    print(f"\n{'='*60}")
    print(f"灵敏度分析: {home.name} vs {away.name}")
    print(f"基准胜率: {pw:.1f}%")
    print(f"{'='*60}")

    layer_names = ['个人能力', '团队化学', '战术因子', '环境因子', '大赛基因', '软性因子']

    layers_impact = {}
    for layer_idx in range(6):
        layer_range = [(0,4), (4,8), (8,12), (12,16), (16,20), (20,24)][layer_idx]
        start, end = layer_range

        # 暂时移除该层的贡献
        original_attack_w = predictor.weights.get_attack_weights().copy()
        original_defense_w = predictor.weights.get_defense_weights().copy()

        new_att_w = original_attack_w.copy()
        new_def_w = original_defense_w.copy()
        new_att_w[start:end] = 0
        new_def_w[start:end] = 0

        # 简单替换并计算（这里需要修改weights结构，简化处理）
        # 实际上我们直接计算该层贡献的lambda
        lb = baseline['lambda_breakdown_home']
        total_lambda = sum(v for k, v in lb.items())
        layer_key = f'{layer_names[layer_idx]}_lambda'
        layer_lambda = lb.get(layer_key, 0)
        contribution = (layer_lambda / max(total_lambda, 0.001)) * 100

        layers_impact[layer_names[layer_idx]] = contribution

    # 排序展示
    for name, impact in sorted(layers_impact.items(), key=lambda x: -x[1]):
        bar = '█' * int(impact / 2) + '░' * (50 - int(impact / 2))
        print(f"  {name:<12} {impact:>5.1f}%  {bar}")

    return layers_impact


# ============================================================
# 七、主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("2026世界杯 · 蒙特卡洛模拟引擎")
    print("=" * 70)

    predictor = WorldCupPredictor()
    simulator = TournamentSimulator(predictor)

    # 运行模拟
    print("\n正在运行 200 次完整赛事模拟...")
    print("-" * 50)
    results = simulator.run_simulations(n=200, verbose=True)

    # ── 夺冠概率 Top 12 ──
    print("\n" + "=" * 70)
    print("🏆 夺冠概率 Top 12")
    print("=" * 70)

    champ_sorted = sorted(results['champion_probs'].items(), key=lambda x: -x[1])
    for rank, (team_code, prob) in enumerate(champ_sorted[:12], 1):
        team = simulator.teams.get(team_code)
        name = team.name if team else team_code
        bar = '▓' * int(prob * 200) + '░' * max(0, 40 - int(prob * 200))
        print(f"  {rank:>2}. {name:<16} {prob*100:>5.1f}%  {bar}")

    # ── 小组出线概率 (含最佳第三) ──
    print("\n" + "=" * 70)
    print("🚪 淘汰赛出线概率 (32强)")
    print("=" * 70)

    ko_sorted = sorted(results['knockout_probs'].items(), key=lambda x: -x[1])
    for team_code, prob in ko_sorted[:16]:
        team = simulator.teams.get(team_code)
        name = team.name if team else team_code
        bar = '▓' * int(prob * 100) + '░' * max(0, 20 - int(prob * 100))
        print(f"  {name:<16} {prob*100:>5.1f}%  {bar}")

    # ── 灵敏度分析 ──
    print("\n")
    sensitivity_analysis(predictor, simulator.teams, ('ENG', 'CRO'))
    sensitivity_analysis(predictor, simulator.teams, ('ARG', 'FRA'))

    print("\n" + "=" * 70)
    print("模拟结果说明：")
    print("  夺冠概率 = 该队在所有5000次模拟中夺冠的次数占比")
    print("  出线概率 = 该队成功进入32强淘汰赛的模拟次数占比")
    print("  灵敏度分析 = 每层因子对预测总λ的贡献百分比")
    print("=" * 70)

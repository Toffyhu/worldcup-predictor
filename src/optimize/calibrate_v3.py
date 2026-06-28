"""
校准策略 v3 — 高效版
预先加载球队数据和化学分，快速扫描参数组合
"""
import sys, math, copy, json
sys.path.insert(0, '/workspace')

import numpy as np
from scipy.stats import poisson
from worldcup_model import WorldCupPredictor, ModelWeights, load_2026_teams, FeatureExtractor
from worldcup_venues import VENUES

# ========== 预加载球队数据（含球员化学分） ==========
print("加载球队数据...")
teams = load_2026_teams()
# 从integrate_all_teams导入球员化学分
from integrate_all_teams import compute_and_update_model
db, raw_chem, updated_teams = compute_and_update_model()
# 使用更新后的球队
for code, td in updated_teams.items():
    teams[code] = td
print(f"  已加载 {len(teams)} 支球队")

# ========== 已完赛比赛 ==========
MATCHES = [
    ("MEX","RSA",(2,0),"Mexico City"),
    ("KOR","CZE",(2,1),"Guadalajara"),
    ("CAN","BIH",(1,1),"Toronto"),
    ("USA","PAR",(4,1),"Los Angeles"),
    ("QAT","SUI",(1,1),"Vancouver"),
    ("BRA","MAR",(1,1),"Dallas"),
    ("SCO","HAI",(1,0),"Boston"),
]

def evaluate_simple(alpha, atk_s, def_s):
    """快速评估一组参数"""
    w = ModelWeights()
    for attr in [a for a in dir(w) if a.endswith('_a') and a.startswith('w_')]:
        setattr(w, attr, getattr(w, attr) * atk_s)
    for attr in [a for a in dir(w) if a.endswith('_d') and a.startswith('w_')]:
        setattr(w, attr, getattr(w, attr) * def_s)
    
    p = WorldCupPredictor(weights=w)
    p.ALPHA = alpha
    p.HOME_ADVANTAGE = 0.25
    
    total_exp = 0
    total_act = 0
    correct_outcome = 0
    correct_score = 0
    brier_total = 0
    eg_spreads = []
    big_games = 0
    one_one_count = 0
    
    for hc, ac, (ha, aa), venue in MATCHES:
        r = p.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key=venue)
        eg_h, eg_a = r['expected_goals_home'], r['expected_goals_away']
        total_exp += eg_h + eg_a
        total_act += ha + aa
        eg_spreads.append(abs(eg_h - eg_a))
        if eg_h + eg_a >= 2.5:
            big_games += 1
        if r['most_likely_score'] == '1-1':
            one_one_count += 1
        
        pw, pd, pa = r['p_home_win']/100, r['p_draw']/100, r['p_away_win']/100
        actual_class = 0 if ha>aa else 2 if aa>ha else 1
        pred_class = 0 if pw>=max(pd,pa) else 1 if pd>=max(pw,pa) else 2
        if pred_class == actual_class:
            correct_outcome += 1
        if r['most_likely_score'] == f"{ha}-{aa}":
            correct_score += 1
        
        if actual_class == 0: brier_total += (1-pw)**2
        elif actual_class == 1: brier_total += (1-pd)**2
        else: brier_total += (1-pa)**2
    
    import statistics
    return {
        'total_exp': round(total_exp, 1),
        'total_act': total_act,
        'bias': round(total_exp - total_act, 1),
        'correct_outcome': correct_outcome,
        'correct_score': correct_score,
        'brier': round(brier_total/len(MATCHES), 4),
        'avg_spread': round(statistics.mean(eg_spreads), 3),
        'big_games': big_games,
        'one_one': one_one_count,
        'avg_eg': round(total_exp/len(MATCHES), 2),
    }

# ========== 基线 ==========
pred_base = WorldCupPredictor()
base = evaluate_simple(0.30, 1.0, 1.0)

print("=" * 70)
print("校准 v3 — 参数扫描")
print("=" * 70)

print(f"\n基线 (α=0.30):")
print(f"  E[G]={base['total_exp']} vs {base['total_act']}  偏差={base['bias']:+.1f}")
print(f"  胜负={base['correct_outcome']}/7  比分命中={base['correct_score']}/7")
print(f"  强弱差距={base['avg_spread']}  大球(>2.5)={base['big_games']}/7  1-1={base['one_one']}/7")

# ========== 扫描 ==========
print(f"\n{'α':>5} {'攻×':>4} {'守×':>4} {'E[G]':>6} {'偏差':>5} {'胜负':>4} {'比分':>4} {'差距':>6} {'大球':>4} {'1-1':>4} {'Brier':>7}")
print(f"  {'─'*57}")

results = []
for alpha in [0.25, 0.22, 0.28, 0.20]:
    for atk_s in [1.10, 1.15, 1.20, 1.25]:
        for def_s in [1.10, 1.15, 1.20]:
            r = evaluate_simple(alpha, atk_s, def_s)
            r['alpha'] = alpha
            r['atk_s'] = atk_s
            r['def_s'] = def_s
            results.append(r)
            print(f"  {alpha:.2f} {atk_s:.2f} {def_s:.2f}  "
                  f"{r['total_exp']:>5.1f} {r['bias']:>+4.1f}  "
                  f"{r['correct_outcome']}/7 {r['correct_score']}/7  "
                  f"{r['avg_spread']:.3f} {r['big_games']}/7  {r['one_one']}/7  {r['brier']:.4f}")

# ========== 选择最优 ==========
# 综合评分：偏差小 + 差距大 + Brier低 + 1-1少 + 大球适中
def score(r):
    bias_pen = abs(r['bias']) * 3
    spread = r['avg_spread'] * 10
    brier_pen = r['brier'] * 30
    one_one_pen = r['one_one'] * 0.5
    big_bonus = r['big_games'] * 0.8
    return spread + big_bonus - bias_pen - brier_pen - one_one_pen

best = max(results, key=score)

print(f"\n{'='*70}")
print(f"✅ 最优: α={best['alpha']:.2f}, 攻×{best['atk_s']:.2f}, 守×{best['def_s']:.2f}")
print(f"{'='*70}")
print(f"  基线: E[G]={base['total_exp']}  偏差={base['bias']:+.1f}  差距={base['avg_spread']}  大球={base['big_games']}")
print(f"  v3:   E[G]={best['total_exp']}  偏差={best['bias']:+.1f}  差距={best['avg_spread']}  大球={best['big_games']}/7  1-1={best['one_one']}/7  Brier={best['brier']}")

# ========== 列出top 3 ==========
ranked = sorted(results, key=score, reverse=True)
print(f"\nTop 3:")
for i, r in enumerate(ranked[:3], 1):
    print(f"  {i}. α={r['alpha']:.2f} 攻×{r['atk_s']:.2f} 守×{r['def_s']:.2f}  "
          f"E[G]={r['total_exp']:.1f}(偏{r['bias']:+.1f}) 差距={r['avg_spread']} 大球={r['big_games']}/7 1-1={r['one_one']}/7")

# ========== 逐场 ==========
w = ModelWeights()
for attr in [a for a in dir(w) if a.endswith('_a') and a.startswith('w_')]:
    setattr(w, attr, getattr(w, attr) * best['atk_s'])
for attr in [a for a in dir(w) if a.endswith('_d') and a.startswith('w_')]:
    setattr(w, attr, getattr(w, attr) * best['def_s'])
pred = WorldCupPredictor(weights=w)
pred.ALPHA = best['alpha']

print(f"\n逐场 (v3):")
for hc, ac, (ha, aa), venue in MATCHES:
    r = pred.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key=venue)
    eg = f"{r['expected_goals_home']:.2f}-{r['expected_goals_away']:.2f}"
    prob = f"{r['p_home_win']:.0f}/{r['p_draw']:.0f}/{r['p_away_win']:.0f}"
    tag = "✅" if r['most_likely_score'] == f"{ha}-{aa}" else ""
    print(f"  {teams[hc].name:12s} vs {teams[ac].name:12s}  {eg}  {prob}%  {r['most_likely_score']:>3s}(实{ha}-{aa}){tag}")

# ========== 15场强弱对决对比 ==========
test15 = [
    ('GER','CUW','德国-库拉索'),('BEL','NZL','比利时-新西兰'),
    ('ARG','JOR','阿根廷-约旦'),('ESP','CPV','西班牙-佛得角'),
    ('BRA','HAI','巴西-海地'),('MEX','KOR','墨西哥-韩国'),
    ('POR','UZB','葡萄牙-乌兹别克'),('CRO','PAN','克罗地亚-巴拿马'),
    ('MAR','HAI','摩洛哥-海地'),('URU','CPV','乌拉圭-佛得角'),
    ('FRA','SEN','法国-塞内加尔'),('ENG','GHA','英格兰-加纳'),
    ('NED','JPN','荷兰-日本'),('GER','CIV','德国-科特迪瓦'),
    ('CZE','MEX','捷克-墨西哥'),
]

print(f"\n{'='*70}")
print(f"强弱对决 — 原始 vs v3")
print(f"{'='*70}")
print(f"  {'对阵':<20} {'原始E[G]':>12} {'原比分':>6} {'v3 E[G]':>12} {'v3比分':>6}")
print(f"  {'─'*56}")

v3_one_one = 0
orig_one_one = 0
v3_spreads = []
orig_spreads = []

for hc, ac, label in test15:
    r_orig = pred_base.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key='Dallas')
    r_v3 = pred.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key='Dallas')
    
    eo = f"{r_orig['expected_goals_home']:.2f}-{r_orig['expected_goals_away']:.2f}"
    ev = f"{r_v3['expected_goals_home']:.2f}-{r_v3['expected_goals_away']:.2f}"
    
    if r_v3['most_likely_score'] == '1-1': v3_one_one += 1
    if r_orig['most_likely_score'] == '1-1': orig_one_one += 1
    v3_spreads.append(abs(r_v3['expected_goals_home'] - r_v3['expected_goals_away']))
    orig_spreads.append(abs(r_orig['expected_goals_home'] - r_orig['expected_goals_away']))
    
    print(f"  {label:<20} {eo:>12} {r_orig['most_likely_score']:>6} {ev:>12} {r_v3['most_likely_score']:>6}")

import statistics
print(f"\n汇总:")
print(f"  1-1占比:  原始 {orig_one_one}/15  v3 {v3_one_one}/15")
print(f"  强弱差距: 原始 {statistics.mean(orig_spreads):.3f}  v3 {statistics.mean(v3_spreads):.3f}")
print(f"  原始模型总E[G]/15场: {sum(r_orig['expected_goals_home']+r_orig['expected_goals_away'] for hc,ac,_ in test15 for r_orig in [pred_base.predict_match(teams[hc],teams[ac],home_code=hc,away_code=ac,venue_key='Dallas')] for _ in [1] if False):.1f}")
# simpler
orig_total = sum(
    pred_base.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key='Dallas')['expected_goals_home'] +
    pred_base.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key='Dallas')['expected_goals_away']
    for hc, ac, _ in test15
)
# just recalculate
print(f"  需重新计算...")

# 保存参数
params = {'alpha': best['alpha'], 'attack_scale': best['atk_s'], 'defense_scale': best['def_s'],
          'metrics': {k: best[k] for k in ['total_exp','bias','correct_outcome','correct_score','avg_spread','big_games','one_one','brier']}}
with open('/workspace/optimized_params_v3.json', 'w') as f:
    json.dump(params, f, indent=2)
print(f"\n参数已保存: /workspace/optimized_params_v3.json")

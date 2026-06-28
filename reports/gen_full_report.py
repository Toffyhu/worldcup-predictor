"""
完整104场预测报告生成器
========================
覆盖全部72场小组赛 + 32场淘汰赛
已赛比赛显示预测值 vs 实际结果
"""
import sys, json, copy, math, random
sys.path.insert(0, '/workspace')
from worldcup_model import WorldCupPredictor, ModelWeights, load_2026_teams
from worldcup_simulator import GROUPS_2026, GROUP_MATCH_VENUES, get_match_venue, DEFAULT_VENUES
from worldcup_venues import VENUES, is_home_stadium
from collections import defaultdict
from datetime import datetime

# ── 加载模型 ──────────────────────────────────────────────
print("加载模型数据...")
from integrate_all_teams import compute_and_update_model
db, rc, ut = compute_and_update_model()
teams = load_2026_teams()
for code, td in ut.items():
    teams[code] = td
print(f"已加载 {len(teams)} 支球队")

ALPHA, ATK_S, DEF_S = 0.20, 1.10, 1.20
w = ModelWeights()
for a in [x for x in dir(w) if x.endswith('_a') and x.startswith('w_')]:
    setattr(w, a, getattr(w, a) * ATK_S)
for a in [x for x in dir(w) if x.endswith('_d') and x.startswith('w_')]:
    setattr(w, a, getattr(w, a) * DEF_S)
pred = WorldCupPredictor(weights=w)
pred.ALPHA = ALPHA

# ── 中文名 ──────────────────────────────────────────────
CN = {
    'Mexico':'墨西哥','South Africa':'南非','South Korea':'韩国','Czech Republic':'捷克',
    'Canada':'加拿大','Bosnia and Herzegovina':'波黑','Qatar':'卡塔尔','Switzerland':'瑞士',
    'Brazil':'巴西','Morocco':'摩洛哥','Haiti':'海地','Scotland':'苏格兰','United States':'美国',
    'Paraguay':'巴拉圭','Australia':'澳大利亚','Turkey':'土耳其','Germany':'德国','Curacao':'库拉索',
    'Ivory Coast':'科特迪瓦','Ecuador':'厄瓜多尔','Netherlands':'荷兰','Japan':'日本','Tunisia':'突尼斯',
    'Sweden':'瑞典','Belgium':'比利时','Egypt':'埃及','Iran':'伊朗','New Zealand':'新西兰',
    'Spain':'西班牙','Cape Verde':'佛得角','Saudi Arabia':'沙特','Uruguay':'乌拉圭','France':'法国',
    'Senegal':'塞内加尔','Iraq':'伊拉克','Norway':'挪威','Argentina':'阿根廷','Algeria':'阿尔及利亚',
    'Austria':'奥地利','Jordan':'约旦','Portugal':'葡萄牙','DR Congo':'刚果金','Uzbekistan':'乌兹别克',
    'Colombia':'哥伦比亚','England':'英格兰','Croatia':'克罗地亚','Ghana':'加纳','Panama':'巴拿马'
}
def cn(c): return CN.get(teams[c].name, teams[c].name)

# ══════════════════════════════════════════════════════════════
# 一、已知实际结果（已赛比赛）
# ══════════════════════════════════════════════════════════════
# 格式: (主队, 客队) -> "主-客"
ACTUAL_RESULTS = {
    ('MEX','RSA'): '2-0', ('KOR','CZE'): '2-1',
    ('CAN','BIH'): '1-1', ('USA','PAR'): '4-1',
    ('QAT','SUI'): '1-1', ('BRA','MAR'): '1-1',
    ('SCO','HAI'): '1-0', ('AUS','TUR'): '2-0',
    ('GER','CUW'): '7-1', ('NED','JPN'): '2-2',
    ('CIV','ECU'): '1-0', ('SWE','TUN'): '5-1',
    ('ESP','CPV'): '0-0', ('BEL','EGY'): '1-1',
    ('KSA','URU'): '1-1', ('IRN','NZL'): '2-2',
    ('FRA','SEN'): '3-1', ('IRQ','NOR'): '1-4',
    ('ARG','ALG'): '3-0', ('AUT','JOR'): '3-1',
    ('POR','COD'): '1-1', ('ENG','CRO'): '4-2',
    ('GHA','PAN'): '1-0', ('UZB','COL'): '1-3',
    ('CZE','RSA'): '1-1', ('SUI','BIH'): '4-1',
    ('CAN','QAT'): '6-0', ('MEX','KOR'): '1-0',
}
# 当前日期（用于判断是否已赛）
CURRENT_DATE = datetime(2026, 6, 15)  # June 15, 2026

# ══════════════════════════════════════════════════════════════
# 二、生成完整104赛场次
# ══════════════════════════════════════════════════════════════

# 小组赛对阵方案：每组4队按GROUPS_2026顺序，6场对阵
# 轮次: (主队idx, 客队idx, match_idx)
GROUP_MATCHUPS = {
    'A': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'B': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'C': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'D': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'E': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'F': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'G': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'H': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'I': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'J': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'K': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
    'L': [(0,1,0),(2,3,1),(0,2,2),(1,3,3),(0,3,4),(1,2,5)],
}

# 比赛日期映射 (基于实际赛程，部分根据已知信息推算)
# 格式: (group, match_idx) -> (月, 日)
MATCH_DATES = {
    # Round 1 (match_idx 0, 1) - 6月11日~14日
    ('A',0): (6,11), ('A',1): (6,12),
    ('B',0): (6,13), ('B',1): (6,13),
    ('C',0): (6,13), ('C',1): (6,14),
    ('D',0): (6,13), ('D',1): (6,14),
    ('E',0): (6,14), ('E',1): (6,14),
    ('F',0): (6,14), ('F',1): (6,15),
    ('G',0): (6,15), ('G',1): (6,15),
    ('H',0): (6,15), ('H',1): (6,15),
    ('I',0): (6,16), ('I',1): (6,16),
    ('J',0): (6,17), ('J',1): (6,17),
    ('K',0): (6,18), ('K',1): (6,18),
    ('L',0): (6,17), ('L',1): (6,17),
    # Round 2 (match_idx 2, 3) - 6月18日~23日
    ('A',2): (6,19), ('A',3): (6,25),
    ('B',2): (6,18), ('B',3): (6,19),
    ('C',2): (6,20), ('C',3): (6,19),
    ('D',2): (6,19), ('D',3): (6,20),
    ('E',2): (6,20), ('E',3): (6,21),
    ('F',2): (6,21), ('F',3): (6,21),
    ('G',2): (6,21), ('G',3): (6,22),
    ('H',2): (6,21), ('H',3): (6,21),
    ('I',2): (6,22), ('I',3): (6,23),
    ('J',2): (6,22), ('J',3): (6,23),
    ('K',2): (6,22), ('K',3): (6,23),
    ('L',2): (6,23), ('L',3): (6,23),
    # Round 3 (match_idx 4, 5) - 6月24日~28日
    ('A',4): (6,24), ('A',5): (6,25),
    ('B',4): (6,24), ('B',5): (6,18),  # BIH-QAT moved to early
    ('C',4): (6,24), ('C',5): (6,24),
    ('D',4): (6,20), ('D',5): (6,26),
    ('E',4): (6,25), ('E',5): (6,25),
    ('F',4): (6,25), ('F',5): (6,21),
    ('G',4): (6,27), ('G',5): (6,27),
    ('H',4): (6,27), ('H',5): (6,27),
    ('I',4): (6,26), ('I',5): (6,23),
    ('J',4): (6,28), ('J',5): (6,28),
    ('K',4): (6,27), ('K',5): (6,27),
    ('L',4): (6,27), ('L',5): (6,27),
}

def is_match_played(month, day):
    """根据当前日期判断比赛是否已完成"""
    return datetime(2026, month, day) <= CURRENT_DATE

# 构建所有比赛列表
all_matches = []  # [(month, day, group, home_code, away_code, venue_key)]

for g, teams_list in GROUPS_2026.items():
    for i, j, midx in GROUP_MATCHUPS[g]:
        home_code = teams_list[i]
        away_code = teams_list[j]
        month, day = MATCH_DATES[(g, midx)]
        venue_key = get_match_venue(g, midx)
        all_matches.append((month, day, g, home_code, away_code, venue_key))

# 按日期排序
all_matches.sort(key=lambda x: (x[0], x[1]))

print(f"共 {len(all_matches)} 场小组赛")

# ══════════════════════════════════════════════════════════════
# 三、预测所有比赛 + 积分榜
# ══════════════════════════════════════════════════════════════

predictions = {}  # (home, away) -> result dict
group_points = defaultdict(lambda: defaultdict(lambda: {'p':0, 'd':0, 'f':0, 'g':0, 'w':0, 'l':0}))

for month, day, g, hc, ac, v in all_matches:
    played = is_match_played(month, day)
    actual = ACTUAL_RESULTS.get((hc, ac)) or ACTUAL_RESULTS.get((ac, hc))
    
    if actual and played:
        # 已赛——用实际结果算积分
        parts = actual.split('-')
        hg, ag = int(parts[0]), int(parts[1])
        # 判断主客 (根据ACTUAL_RESULTS键)
        if (hc, ac) in ACTUAL_RESULTS:
            real_hg, real_ag = hg, ag
        else:
            real_hg, real_ag = ag, hg
        
        pred_result = pred.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key=v)
        pred_result['actual_score'] = actual
        pred_result['actual_home_goals'] = real_hg
        pred_result['actual_away_goals'] = real_ag
        pred_result['is_played'] = True
        predictions[(hc, ac)] = pred_result
        
        # 积分
        gp = group_points[g]
        gp[hc]['f'] += real_hg; gp[hc]['d'] += real_hg - real_ag; gp[hc]['g'] += 1
        gp[ac]['f'] += real_ag; gp[ac]['d'] += real_ag - real_hg; gp[ac]['g'] += 1
        if real_hg > real_ag:
            gp[hc]['p'] += 3; gp[hc]['w'] += 1; gp[ac]['l'] += 1
        elif real_hg < real_ag:
            gp[ac]['p'] += 3; gp[ac]['w'] += 1; gp[hc]['l'] += 1
        else:
            gp[hc]['p'] += 1; gp[ac]['p'] += 1
    else:
        # 未赛——用模型预测
        pred_result = pred.predict_match(teams[hc], teams[ac], home_code=hc, away_code=ac, venue_key=v)
        pred_result['is_played'] = False
        predictions[(hc, ac)] = pred_result
        
        pw, pd = pred_result['p_home_win']/100, pred_result['p_draw']/100
        pa = pred_result['p_away_win']/100
        eh, ea = pred_result['expected_goals_home'], pred_result['expected_goals_away']
        
        gp = group_points[g]
        gp[hc]['p'] += pw*3 + pd*1
        gp[hc]['f'] += eh; gp[hc]['d'] += eh - ea; gp[hc]['g'] += 1
        gp[ac]['p'] += pa*3 + pd*1
        gp[ac]['f'] += ea; gp[ac]['d'] += ea - eh; gp[ac]['g'] += 1

print("小组赛预测完成")

# ══════════════════════════════════════════════════════════
# 淘汰赛预测 — FIFA官方对阵 + 淘汰赛5因子
# ══════════════════════════════════════════════════════════

# 淘汰赛专属因子数据
INJURY_KO={'BRA':-0.08,'NED':-0.07,'JPN':-0.06,'GER':-0.03,'USA':-0.02,'FRA':-0.01,'AUS':-0.02}
MOM_KO={'MEX':1.04,'FRA':1.08,'NED':1.06,'SUI':1.06,'ESP':1.06,'ENG':1.06,'COL':1.06,'BRA':1.06,'GER':1.04,'ARG':1.04,'BEL':1.03,'JPN':1.03,'CRO':1.04,'MAR':1.04,'USA':1.03,'CIV':1.03,'EGY':1.03,'NOR':1.03,'AUT':1.02,'POR':1.02,'CAN':1.01,'RSA':1.00,'AUS':1.00,'CPV':0.97,'COD':0.97,'SWE':0.96,'ECU':0.96,'GHA':0.96,'BIH':0.95,'PAR':0.95,'SEN':0.94,'ALG':0.94}
EXP_KO={'CRO':0.10,'ARG':0.08,'FRA':0.08,'MAR':0.06,'BRA':0.06,'NED':0.05,'ENG':0.05,'GER':0.05,'ESP':0.05,'POR':0.04,'BEL':0.04,'COL':0.04,'MEX':0.02,'USA':0.03,'SUI':0.03,'SEN':0.03}
PEN_KO={'CRO':0.12,'ARG':0.09,'GER':0.06,'FRA':0.05,'BRA':0.05,'ESP':0.04,'NED':0.03,'ENG':-0.04,'MEX':0.03}
UNI_KO={'FRA':0.04,'ESP':0.04,'COL':0.04,'MEX':0.02,'GER':0.03,'ARG':0.03,'NED':0.03,'POR':0.02,'ENG':0.02,'BEL':0.02,'JPN':-0.03,'USA':0.02,'CAN':0.02,'MAR':0.03,'CRO':0.03,'COD':0.03,'CPV':0.03,'ECU':0.03}

def _ko_advance_prob(t1_code, t2_code, venue_key):
    """计算淘汰赛晋级概率（含5因子）"""
    res = pred.predict_match(teams[t1_code], teams[t2_code], t1_code, t2_code, venue_key=venue_key)
    p1, pd, p2 = res['p_home_win'], res['p_draw'], res['p_away_win']
    eg1, eg2 = res['expected_goals_home'], res['expected_goals_away']
    
    # 基础晋级概率 (0-1 scale)
    r1 = (p1 + 0.5*pd) / 100
    r2 = (p2 + 0.5*pd) / 100
    
    # 淘汰赛因子
    b1 = INJURY_KO.get(t1_code,0) + (MOM_KO.get(t1_code,1)-1)*0.5 + EXP_KO.get(t1_code,0)*1.2 + UNI_KO.get(t1_code,0)*0.6
    b2 = INJURY_KO.get(t2_code,0) + (MOM_KO.get(t2_code,1)-1)*0.5 + EXP_KO.get(t2_code,0)*1.2 + UNI_KO.get(t2_code,0)*0.6
    if abs(p1-p2)/100 < 0.08:
        b1 += PEN_KO.get(t1_code,0)*0.6
        b2 += PEN_KO.get(t2_code,0)*0.6
    
    a1, a2 = r1+b1, r2+b2
    winner = t1_code if a1 > a2 else t2_code
    adv_pct = max(a1,a2)/(a1+a2)*100
    
    return {'home':t1_code,'away':t2_code,'eg_h':round(eg1,2),'eg_a':round(eg2,2),
            'pw':round(p1),'pd':round(pd),'pa':round(p2),
            'most_likely':f'{round(eg1)}-{round(eg2)}',
            'winner':winner,'adv_pct':round(adv_pct,1)}

def predict_knockout():
    """基于FIFA官方对阵 + 淘汰赛因子"""
    # 各组排名
    rankings = {}
    for g in sorted(GROUPS_2026):
        rk = sorted(group_points[g].items(), key=lambda x: (-x[1]['p'], -x[1]['d'], -x[1]['f']))
        rankings[g] = [t[0] for t in rk]
    
    # 最佳8个第三
    all_third = []
    for g in sorted(GROUPS_2026):
        c = rankings[g][2]
        pts = group_points[g][c]['p']
        gd = group_points[g][c]['d']
        all_third.append((c, pts, gd, g))
    all_third.sort(key=lambda x: (-x[1], -x[2]))
    best_thirds = {t[0]: t for t in all_third[:8]}
    
    # 按组分配third: A组→3rd(C/E/F/H/I), B组→3rd(E/F/G/I/J), etc.
    third_to_group = {
        'A': [c for (c,p,gd,g) in all_third if g in ('C','E','F','H','I')][:1],
        'B': [c for (c,p,gd,g) in all_third if g in ('E','F','G','I','J')][:1],
        'D': [c for (c,p,gd,g) in all_third if g in ('B','E','F','I','J')][:1],
        'E': [c for (c,p,gd,g) in all_third if g in ('A','B','C','D','F')][:1],
        'I': [c for (c,p,gd,g) in all_third if g in ('C','D','F','G','H')][:1],
        'L': [c for (c,p,gd,g) in all_third if g in ('E','H','I','J','K')][:1],
        'G': [c for (c,p,gd,g) in all_third if g in ('A','E','H','I','J')][:1],
        'K': [c for (c,p,gd,g) in all_third if g in ('D','E','I','J','L')][:1],
    }

    # FIFA官方R32对阵
    r32_pairs = [
        (rankings['A'][1], rankings['B'][1], 'Los Angeles'),         # M73
        (rankings['E'][0], third_to_group['E'][0], 'Boston'),         # M74
        (rankings['F'][0], rankings['C'][1], 'Monterrey'),            # M75
        (rankings['C'][0], rankings['F'][1], 'Houston'),              # M76
        (rankings['I'][0], third_to_group['I'][0], 'New York'),       # M77
        (rankings['E'][1], rankings['I'][1], 'Dallas'),               # M78
        (rankings['A'][0], third_to_group['A'][0], 'Mexico City'),   # M79
        (rankings['L'][0], third_to_group['L'][0], 'Atlanta'),        # M80
        (rankings['D'][0], third_to_group['D'][0], 'San Francisco'),  # M81
        (rankings['G'][0], third_to_group['G'][0], 'Seattle'),        # M82
        (rankings['K'][1], rankings['L'][1], 'Toronto'),              # M83
        (rankings['H'][0], rankings['J'][1], 'Los Angeles'),          # M84
        (rankings['B'][0], third_to_group['B'][0], 'Vancouver'),      # M85
        (rankings['J'][0], rankings['H'][1], 'Miami'),                # M86
        (rankings['K'][0], third_to_group['K'][0], 'Kansas City'),    # M87
        (rankings['D'][1], rankings['G'][1], 'Dallas'),               # M88
    ]
    
    r32 = [_ko_advance_prob(a, b, v) for (a, b, v) in r32_pairs]
    
    # R16 bracket: M89=W74vsW77, M90=W73vsW75, M91=W76vsW78, M92=W79vsW80,
    #               M93=W83vsW84, M94=W81vsW82, M95=W86vsW88, M96=W85vsW87
    W = [r['winner'] for r in r32]
    r16_pairs = [
        (W[1], W[4], 'NRG Stadium'), (W[0], W[2], 'Philadelphia'),
        (W[3], W[5], 'New York'), (W[6], W[7], 'Mexico City'),
        (W[10], W[11], 'Dallas'), (W[8], W[9], 'Seattle'),
        (W[13], W[15], 'Atlanta'), (W[12], W[14], 'Vancouver'),
    ]
    r16 = [_ko_advance_prob(a, b, v) for (a, b, v) in r16_pairs]
    
    # QF: M97=W89vsW90, M98=W93vsW94, M99=W91vsW92, M100=W95vsW96
    W16 = [r['winner'] for r in r16]
    qf_pairs = [
        (W16[0], W16[1], 'Boston'), (W16[4], W16[5], 'Los Angeles'),
        (W16[2], W16[3], 'Miami'), (W16[6], W16[7], 'Kansas City'),
    ]
    qf = [_ko_advance_prob(a, b, v) for (a, b, v) in qf_pairs]
    
    # SF
    W8 = [r['winner'] for r in qf]
    sf_pairs = [(W8[0], W8[1], 'Dallas'), (W8[2], W8[3], 'Atlanta')]
    sf = [_ko_advance_prob(a, b, v) for (a, b, v) in sf_pairs]
    
    # Final & 3rd
    W4 = [r['winner'] for r in sf]
    L4 = [r['away'] if r['winner'] == r['home'] else r['home'] for r in sf]
    third = [_ko_advance_prob(L4[0], L4[1], 'Miami')]
    final = [_ko_advance_prob(W4[0], W4[1], 'New York')]
    
    return {
        'r32': r32, 'r16': r16, 'qf': qf, 'sf': sf,
        'third_place': third, 'final': final,
        'champion': final[0]['winner'] if final else 'N/A',
        'rankings': rankings,  # group standings
    }

ko = predict_knockout()
print("淘汰赛预测完成")

# ══════════════════════════════════════════════════════════════
# 四、生成 HTML 报告
# ══════════════════════════════════════════════════════════════

def fmt_pct(v):
    return f'{v:.0f}%'
def fmt_eg(v):
    return f'{v:.2f}'

def winner_icon(pw, pd, pa):
    if pw >= 55: return 'H'
    elif pa >= 55: return 'A'
    elif pw >= 45 or pa >= 45: return 'E'
    else: return 'D'

html_parts = []
def H(s): html_parts.append(s)

H('''<!DOCTYPE html><html lang="zh-CN"><head>
<meta charset="UTF-8">
<title>2026世界杯 · 完整104场预测总表</title>
<style>
@page { size: A4 landscape; margin: 1.0cm 1.2cm; }
body { font-family: 'Noto Sans CJK SC', 'SimSun', serif; font-size: 8pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 14pt; text-align: center; margin: 0.1cm 0; letter-spacing: 1pt; }
.subtitle { text-align: center; font-size: 8.5pt; color: #666; margin-bottom: 0.3cm; }
h2 { font-size: 10pt; margin: 0.3cm 0 0.15cm 0; border-bottom: 1.5px solid #2c5282; padding-bottom: 2pt; color: #2c5282; }
h3 { font-size: 9pt; margin: 0.2cm 0 0.1cm 0; color: #444; }
table { width: 100%; border-collapse: collapse; margin: 0.15cm 0; font-size: 7.5pt; page-break-inside: avoid; }
th { background: #2c5282; color: white; padding: 3pt 4pt; text-align: center; font-weight: 600; font-size: 7pt; }
td { padding: 2pt 3pt; border: 1px solid #ccc; text-align: center; }
tr:nth-child(even) { background: #f8f9fa; }
.played { background: #e8f5e9 !important; }
.ko-win { background: #e3f2fd; font-weight: bold; }
.match-day { background: #e8eaf6; font-weight: bold; font-size: 8pt; }
.group-title { background: #fff3e0; font-weight: bold; }
.actual-score { color: #c62828; font-weight: bold; }
.pred-error { color: #e65100; font-size: 6.5pt; }
.ko-bracket { font-size: 7pt; line-height: 1.6; }
.champion-box { background: linear-gradient(135deg, #fff9c4, #ffe082); border: 2px solid #f9a825; border-radius: 4px; padding: 0.2cm; margin: 0.3cm 0; text-align: center; }
.champion-box h3 { margin: 0; color: #e65100; font-size: 11pt; }
.footnote { font-size: 6.5pt; color: #999; margin-top: 0.3cm; border-top: 1px solid #ddd; padding-top: 0.15cm; }
</style></head><body>

<h1> 2026世界杯 · 完整104场预测总表</h1>
<div class="subtitle">
  v3优化模型 · α=0.20 · 攻×1.10 · 守×1.20 · 球员级化学归一化[15,85]
  &nbsp;|&nbsp; 生成时间: 2026-06-15 &nbsp;|&nbsp; 已赛比赛显示 🟢实际比分
</div>

<h2>一、小组赛（72场）</h2>
''')

# ── 按日期分组输出 ──
current_month = 0
current_day = 0

for month, day, g, hc, ac, v in all_matches:
    if month != current_month or day != current_day:
        current_month, current_day = month, day
        H(f'<h3>▎{month}月{day}日</h3>')
        H('<table><tr><th>组</th><th>主队</th><th>EG</th><th>客队</th><th>H/D/A</th><th>胜/平/负</th><th>最可能</th><th>实际比分</th><th>场馆</th></tr>')
    
    pr = predictions.get((hc, ac)) or predictions.get((ac, hc))
    if pr is None:
        continue
    
    played = pr.get('is_played', False)
    pw = pr['p_home_win']; pd = pr['p_draw']; pa = pr['p_away_win']
    eh = pr['expected_goals_home']; ea = pr['expected_goals_away']
    ml = pr['most_likely_score']
    wi = winner_icon(pw, pd, pa)
    
    actual_str = pr.get('actual_score', '')
    venue_short = v.split('/')[0].strip() if '/' in v else v
    
    played_cls = 'played' if played else ''
    
    if actual_str:
        # 将实际比分按当前主客方向对齐
        act_parts = actual_str.split('-')
        if (hc, ac) in ACTUAL_RESULTS:
            # 实际比分键与本次对阵方向一致
            aligned_actual = actual_str
        else:
            # 实际比分键是反的，需要翻转
            aligned_actual = f'{act_parts[1]}-{act_parts[0]}'
        
        real_hg = int(act_parts[0]) if (hc, ac) in ACTUAL_RESULTS else int(act_parts[1])
        real_ag = int(act_parts[1]) if (hc, ac) in ACTUAL_RESULTS else int(act_parts[0])
        
        # 判断预测 vs 实际
        actual_hda = 'H' if real_hg > real_ag else ('D' if real_hg == real_ag else 'A')
        pred_hda = 'H' if pw > pa else ('D' if abs(pw-pa) <= 5 else 'A')
        correct_hda = actual_hda == pred_hda
        
        # HDA预测指示
        hda_label = f'<b>{pred_hda}</b>'
        if actual_str:
            hda_label += f' <small>(实{actual_hda})</small>' if not correct_hda else ''
        
        # 比分对比
        ml_clean = ml.replace(' ', '')
        if aligned_actual == ml_clean:
            score_status = '✅'
        elif correct_hda:
            score_status = '方向✓'
        else:
            score_status = '✗'
        
        actual_display = f'<span class="actual-score">{aligned_actual}</span> {score_status}'
        
        H(f'<tr class="{played_cls}"><td>{g}</td><td>{cn(hc)}</td><td>{fmt_eg(eh)}-{fmt_eg(ea)}</td><td>{cn(ac)}</td>'
          f'<td>{hda_label}</td><td>{fmt_pct(pw)}/{fmt_pct(pd)}/{fmt_pct(pa)}</td><td>{ml}</td>'
          f'<td>{actual_display}</td><td>{venue_short}</td></tr>')
    else:
        # 方向标识
        hda_label = f'<b>{"H" if pw>pa else "D" if abs(pw-pa)<=5 else "A"}</b>'
        H(f'<tr><td>{g}</td><td>{cn(hc)}</td><td>{fmt_eg(eh)}-{fmt_eg(ea)}</td><td>{cn(ac)}</td>'
          f'<td>{hda_label}</td><td>{fmt_pct(pw)}/{fmt_pct(pd)}/{fmt_pct(pa)}</td><td>{ml}</td>'
          f'<td>-</td><td>{venue_short}</td></tr>')

H('</table>')

# ── 小组积分榜 ──
H('<h2>二、小组积分榜</h2>')
H('<table><tr><th>组别</th><th>排名</th><th>球队</th><th>积分</th><th>净胜球</th><th>进球</th><th>出线</th></tr>')

for g in sorted(GROUPS_2026):
    rk = sorted(group_points[g].items(), key=lambda x: (-round(x[1]['p'],1), -x[1]['d'], -x[1]['f']))
    for i, (c, s) in enumerate(rk):
        pts = str(int(round(s['p'])))
        gd = f'{int(round(s["d"])):+d}' if abs(s["d"]) >= 0.5 else '0'
        qualified = 'Q' if i < 2 else '*'
        H(f'<tr><td>{g}</td><td>{i+1}</td><td>{cn(c)}</td><td>{pts}</td><td>{gd}</td><td>{int(round(s["f"]))}</td><td>{qualified}</td></tr>')

H('</table>')

# ── 淘汰赛 ──
H('<h2>三、淘汰赛预测（32强 → 决赛）</h2>')

def render_ko_round(matches, round_name):
    H(f'<h3>{round_name}</h3>')
    H('<table><tr><th>对阵</th><th>EG</th><th>H/D/A</th><th>胜/平/负</th><th>最可能</th><th>预测胜者</th></tr>')
    for m in matches:
        wi = winner_icon(m['pw'], m['pd'], m['pa'])
        hda_label = f'<b>{wi}</b>'
        winner_name = f'{cn(m["winner"])}'
        H(f'<tr><td>{cn(m["home"])} vs {cn(m["away"])}</td>'
          f'<td>{m["eg_h"]}-{m["eg_a"]}</td>'
          f'<td>{hda_label}</td>'
          f'<td>{m["pw"]}%/{m["pd"]}%/{m["pa"]}%</td>'
          f'<td>{m["most_likely"]}</td>'
          f'<td class="ko-win">{winner_name}</td></tr>')
    H('</table>')

render_ko_round(ko['r32'], '🔵 32强赛 (16场)')
render_ko_round(ko['r16'], '🔵 16强赛 (8场)')
render_ko_round(ko['qf'], '🔵 1/4决赛 (4场)')
render_ko_round(ko['sf'], '🔵 半决赛 (2场)')
render_ko_round(ko['third_place'], '🥉 季军赛')
render_ko_round(ko['final'], ' 决赛')

# 冠军预测
champion = ko['champion']
H(f'<div class="champion-box"><h3>[冠军] 预测冠军：{cn(champion)}</h3>')
# 亚军
final_match = ko['final'][0]
runner_up = final_match['home'] if final_match['winner'] == final_match['away'] else final_match['away']
H(f'<p>1st {cn(champion)} | 2nd {cn(runner_up)} | 3rd {cn(ko["third_place"][0]["winner"]) if ko["third_place"] else "--"}</p>')
H('</div>')

# 模型参数
H(f'<div class="footnote">')
H(f'<b>模型参数</b>: α={ALPHA} | 攻击权重×{ATK_S} | 防守权重×{DEF_S} | 球员级化学归一化[15,85]<br>')
H(f'<b>方法论</b>: 6层13因子泊松模型 · 场馆感知环境注入 · 非线性明星依赖 · 蒙特卡洛模拟<br>')
H(f'<b>验证</b>: 已赛{len(ACTUAL_RESULTS)}场 · 比分命中?/·场 · 灵栖Invest量化研究团队')
H('</div>')

H('</body></html>')

# 写入HTML
with open('/workspace/full_predictions.html', 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))
print("✅ HTML报告已生成: /workspace/full_predictions.html")

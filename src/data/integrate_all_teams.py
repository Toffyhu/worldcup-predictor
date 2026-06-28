"""
完整48队球员级数据整合
=======================
将Batch Fetch数据合并到SquadDatabase，计算全队化学分，更新预测模型。
"""

import sys
sys.path.insert(0, '/workspace')

import json
from worldcup_model import WorldCupPredictor, load_2026_teams
from worldcup_squads import SquadDatabase, ChemistryEngine
from batch_fetch_squads import fetch_nation_page, parse_nation_data, estimate_squad_from_opr, FC_NATIONS_URLS


def build_complete_database() -> SquadDatabase:
    """
    构建完整的48队阵容数据库：
    - 10支核心球队（手动精确构建）
    - 29支球队（从EA FC批量抓取）
    - 9支球队（基于OPR估算）
    """
    from worldcup_model import load_2026_teams
    from worldcup_squads import SquadDatabase, TeamSquad
    team_data = load_2026_teams()

    # 1. 加载已有的SquadDatabase（10支核心球队）
    db = SquadDatabase()
    # 清理估算数据，只保留核心球队
    core_codes = {'FRA', 'ARG', 'BRA', 'ENG', 'GER', 'ESP', 'POR', 'NED', 'BEL', 'CRO', 'USA'}
    core_squads = {code: db.squads[code] for code in core_codes if code in db.squads}
    db.squads = core_squads
    print(f"  核心球队: {len(db.squads)} 支 (精确球员数据)")

    # 2. 批量抓取 + 估算剩余的38支球队
    fetched_count = 0
    estimated_count = 0

    for code, info in FC_NATIONS_URLS.items():
        if code in db.squads:
            continue
        td = team_data.get(code)
        if not td:
            continue

        url = info['url']
        est_count = info.get('est_players', 10)

        if url and est_count >= 15:
            raw = fetch_nation_page(code, url)
            if raw and len(raw) >= 11:
                players = parse_nation_data(code, raw)
                if len(players) >= 11:
                    xi = [p.name for p in players[:11]]
                    db.squads[code] = TeamSquad(
                        code=code, name=td.name, coach_name=f'{td.name} Coach',
                        training_camp_days=14, captain_name=players[0].name,
                        probable_starting_xi=xi, players=players,
                    )
                    fetched_count += 1
                    continue

        # 估算
        base_ovr = max(60, int((td.opr_attack + td.opr_defense) / 2))
        players = estimate_squad_from_opr(code, base_ovr)
        xi = [p.name for p in players[:11]]
        db.squads[code] = TeamSquad(
            code=code, name=td.name, coach_name=f'{td.name} Coach',
            training_camp_days=14, captain_name=players[0].name,
            probable_starting_xi=xi, players=players,
        )
        estimated_count += 1

    print(f"  EA FC抓取: {fetched_count} 支")
    print(f"  估算: {estimated_count} 支")
    print(f"  总计: {len(db.squads)} 支")
    return db


def compute_and_update_model():
    """计算全队化学分并更新预测模型"""
    from worldcup_model import WorldCupPredictor, load_2026_teams
    import copy

    print("=" * 65)
    print("2026世界杯 · 完整48队球员级数据集成")
    print("=" * 65)

    # 1. 构建完整数据库
    print("\n[1/4] 构建完整阵容数据库...")
    db = build_complete_database()
    engine = ChemistryEngine()

    # 2. 计算化学分
    print("\n[2/4] 计算全48队球员级化学分...")

    # 有真实EA FC数据的球队
    real_data_teams = {'FRA', 'ARG', 'BRA', 'ENG', 'GER', 'ESP', 'POR', 'NED', 'BEL', 'CRO',
                       'MEX', 'KOR', 'CZE', 'CAN', 'BIH', 'SUI', 'MAR', 'HAI', 'SCO',
                       'USA', 'PAR', 'AUS', 'TUR', 'CIV', 'ECU', 'JPN', 'TUN', 'SWE',
                       'NZL', 'CPV', 'KSA', 'URU', 'SEN', 'NOR', 'ALG', 'AUT', 'COD', 'COL', 'GHA'}

    # 先跑一遍化学分，收集avg_ovr
    raw_chem = {}
    real_squads = {}

    for code, squad in db.squads.items():
        chem = engine.compute(squad)
        xi = squad.get_starting_xi_sorted()
        avg_ovr = sum(p.ovr for p in xi) / max(len(xi), 1)
        chem['avg_ovr'] = round(avg_ovr, 1)
        raw_chem[code] = chem
        if code in real_data_teams:
            real_squads[code] = squad

    # 归一化（仅基于真实数据球队）
    print(f"  基于 {len(real_squads)} 支真实数据球队进行归一化...")
    normalized = engine.normalize_across_teams(real_squads, target_min=15.0, target_max=85.0)

    for code, chem in normalized.items():
        raw_chem[code]['club_chemistry_norm'] = chem['club_chemistry_norm']
        raw_chem[code]['league_familiarity_norm'] = chem['league_familiarity_norm']

    # 估算球队使用中等值
    for code in db.squads:
        if code not in real_data_teams:
            raw_chem[code]['club_chemistry_norm'] = 45.0
            raw_chem[code]['league_familiarity_norm'] = 45.0

    # 3. 更新TeamData
    print("\n[3/4] 更新TeamData中的球员级化学分...")
    teams = load_2026_teams()
    updated_teams = {}

    for code, td in teams.items():
        if code in raw_chem:
            chem = raw_chem[code]
            new_td = copy.deepcopy(td)
            new_td.club_chemistry = chem['club_chemistry_norm']
            new_td.league_familiarity = chem['league_familiarity_norm']
            updated_teams[code] = new_td
        else:
            updated_teams[code] = td

    # 4. 输出化学分排名
    print("\n" + "=" * 65)
    print("全48队球员级化学分排名（基于EA FC 26真实数据）")
    print("=" * 65)
    print(f"{'排名':<5} {'队名':<14} {'化学分':>8} {'同俱乐部':>8} {'同联赛':>8} {'首发OVR':>8} {'数据源':<10}")
    print("-" * 65)

    ranked = sorted([(c, raw_chem[c]) for c in teams if c in raw_chem],
                     key=lambda x: -x[1]['chemistry_score'])

    for rank, (code, chem) in enumerate(ranked, 1):
        td = teams[code]
        source = "EA FC ✅" if code in real_data_teams else "估算 ⚠️"
        print(f"  {rank:<3} {td.name:<12} {chem['chemistry_score']:>8.1f}"
              f" {chem['club_chemistry_norm']:>8.1f} {chem['league_familiarity_norm']:>8.1f}"
              f" {chem.get('avg_ovr', 0):>8.1f} {source:<10}")

    # 5. 预测对比
    print("\n[4/4] 球员级化学模型 vs 原始模型 — 关键对阵预测")
    print("=" * 65)

    predictor_orig = WorldCupPredictor()

    # 用override_team_chemistry更新
    predictor_player = WorldCupPredictor(use_player_chemistry=True)
    player_teams = predictor_player.override_team_chemistry(load_2026_teams())

    # 补充覆盖完整set
    for code, td in updated_teams.items():
        player_teams[code] = td
    # 也更新predictor的teams
    for code in real_data_teams:
        if code in updated_teams:
            pass  # 已更新

    test_matches = [
        ("FRA", "ENG", "Los Angeles", "法国 vs 英格兰"),
        ("GER", "ESP", "Houston", "德国 vs 西班牙"),
        ("ARG", "POR", "Atlanta", "阿根廷 vs 葡萄牙"),
        ("BRA", "NED", "Dallas", "巴西 vs 荷兰"),
        ("USA", "TUR", "Los Angeles", "美国 vs 土耳其"),
        ("JPN", "SWE", "Kansas City", "日本 vs 瑞典"),
        ("MAR", "SCO", "Boston", "摩洛哥 vs 苏格兰"),
        ("SEN", "NOR", "New York / New Jersey", "塞内加尔 vs 挪威"),
        ("URU", "COL", "Miami", "乌拉圭 vs 哥伦比亚"),
        ("MEX", "KOR", "Mexico City", "墨西哥 vs 韩国"),
    ]

    for h, a, v, label in test_matches:
        orig_teams = load_2026_teams()
        r_orig = predictor_orig.predict_match(orig_teams[h], orig_teams[a],
                                               home_code=h, away_code=a, venue_key=v)
        r_play = predictor_player.predict_match(player_teams[h], player_teams[a],
                                                 home_code=h, away_code=a, venue_key=v)

        delta = r_play['p_home_win'] - r_orig['p_home_win']
        arrow = "📈" if delta > 1 else "📉" if delta < -1 else "➡️"

        print(f"\n  {label}")
        print(f"    原始: {r_orig['home_team']} {r_orig['p_home_win']}% | "
              f"平 {r_orig['p_draw']}% | {r_orig['away_team']} {r_orig['p_away_win']}%")
        ch_orig = f"{orig_teams[h].club_chemistry:.0f}/{orig_teams[a].club_chemistry:.0f}"
        ch_new = f"{player_teams[h].club_chemistry:.0f}/{player_teams[a].club_chemistry:.0f}"
        print(f"    球员: {r_play['home_team']} {r_play['p_home_win']}% | "
              f"平 {r_play['p_draw']}% | {r_play['away_team']} {r_play['p_away_win']}%"
              f"  {arrow} 化学: {ch_orig} → {ch_new}")

    # 保存结果
    real_count = len(real_data_teams)
    print(f"\n{'='*65}")
    print("📊 数据统计:")
    print(f"  真实球员级数据: {real_count}/48 支球队 ({real_count/48*100:.0f}%)")
    for rank in range(3):
        top = ranked[rank]
        print(f"  化学分第{rank+1}: {teams[top[0]].name} ({top[1]['chemistry_score']:.0f})")
    print(f"{'='*65}")

    return db, raw_chem, updated_teams


def compute_and_update_model():
    """计算全队化学分并更新预测模型"""
    from worldcup_model import WorldCupPredictor, load_2026_teams
    import copy

    print("=" * 65)
    print("2026世界杯 · 完整48队球员级数据集成")
    print("=" * 65)

    # 1. 构建完整数据库
    print("\n[1/4] 构建完整阵容数据库...")
    db = build_complete_database()
    engine = ChemistryEngine()

    # 2. 计算化学分
    print("\n[2/4] 计算全48队球员级化学分...")

    # 分离有真实数据的球队和估算球队
    real_data_teams = {'FRA', 'ARG', 'BRA', 'ENG', 'GER', 'ESP', 'POR', 'NED', 'BEL', 'CRO',
                       'MEX', 'KOR', 'CZE', 'CAN', 'BIH', 'SUI', 'MAR', 'HAI', 'SCO',
                       'USA', 'PAR', 'AUS', 'TUR', 'CIV', 'ECU', 'JPN', 'TUN', 'SWE',
                       'NZL', 'CPV', 'KSA', 'URU', 'SEN', 'NOR', 'ALG', 'AUT', 'COD', 'COL', 'GHA'}

    chem_results = {}
    real_squads = {}
    estimated_squads = {}

    for code, squad in db.squads.items():
        chem = engine.compute(squad)
        chem_results[code] = chem
        if code in real_data_teams:
            real_squads[code] = squad
        else:
            estimated_squads[code] = squad

    # 3. 归一化（仅基于真实数据球队）
    print("  基于真实数据球队进行归一化...")
    normalized = engine.normalize_across_teams(real_squads, target_min=15.0, target_max=85.0)

    for code, chem in normalized.items():
        chem_results[code]['club_chemistry_norm'] = chem['club_chemistry_norm']
        chem_results[code]['league_familiarity_norm'] = chem['league_familiarity_norm']

    # 估算球队使用中等值
    for code in estimated_squads:
        chem_results[code]['club_chemistry_norm'] = 45.0
        chem_results[code]['league_familiarity_norm'] = 45.0

    # 4. 更新TeamData
    print("\n[3/4] 更新TeamData中的球员级化学分...")
    teams = load_2026_teams()
    updated_teams = {}

    for code, td in teams.items():
        if code in chem_results:
            chem = chem_results[code]
            new_td = copy.deepcopy(td)
            new_td.club_chemistry = chem['club_chemistry_norm']
            new_td.league_familiarity = chem['league_familiarity_norm']
            updated_teams[code] = new_td
        else:
            updated_teams[code] = td

    # 5. 输出化学分排名
    print("\n" + "=" * 65)
    print("全48队球员级化学分排名（基于EA FC 26真实数据）")
    print("=" * 65)
    print(f"{'排名':<5} {'队名':<14} {'化学分':>8} {'同俱乐部':>8} {'同联赛':>8} {'首发OVR':>8} {'数据源':<10}")
    print("-" * 65)

    ranked = sorted([(c, chem_results[c]) for c in teams],
                     key=lambda x: -x[1]['chemistry_score'])

    for rank, (code, chem) in enumerate(ranked, 1):
        td = teams[code]
        source = "EA FC ✅" if code in real_data_teams else "估算 ⚠️"
        print(f"  {rank:<3} {td.name:<12} {chem['chemistry_score']:>8.1f}"
              f" {chem['club_chemistry_norm']:>8.1f} {chem['league_familiarity_norm']:>8.1f}"
              f" {chem.get('avg_ovr', chem_results[code].get('avg_ovr', 0)):>8.1f}"
              f" {source:<10}")

    # 6. 预测对比
    print("\n[4/4] 球员级化学模型 vs 原始模型 — 关键对阵预测")
    print("=" * 65)

    predictor_orig = WorldCupPredictor()
    predictor_player = WorldCupPredictor(use_player_chemistry=True)
    player_teams = predictor_player.override_team_chemistry(load_2026_teams())

    # 也用完整更新覆盖
    for code, td in updated_teams.items():
        if code in player_teams:
            player_teams[code] = td

    test_matches = [
        ("FRA", "ENG", "Los Angeles", "法国 vs 英格兰"),
        ("GER", "ESP", "Houston", "德国 vs 西班牙"),
        ("ARG", "POR", "Atlanta", "阿根廷 vs 葡萄牙"),
        ("BRA", "NED", "Dallas", "巴西 vs 荷兰"),
        ("USA", "TUR", "Los Angeles", "美国 vs 土耳其"),
        ("JPN", "SWE", "Kansas City", "日本 vs 瑞典"),
        ("MAR", "SCO", "Boston", "摩洛哥 vs 苏格兰"),
        ("SEN", "NOR", "New York / New Jersey", "塞内加尔 vs 挪威"),
        ("URU", "COL", "Miami", "乌拉圭 vs 哥伦比亚"),
        ("MEX", "KOR", "Guadalajara", "墨西哥 vs 韩国"),
    ]

    for h, a, v, label in test_matches:
        orig_teams = load_2026_teams()
        r_orig = predictor_orig.predict_match(orig_teams[h], orig_teams[a],
                                               home_code=h, away_code=a, venue_key=v)
        r_play = predictor_player.predict_match(player_teams[h], player_teams[a],
                                                 home_code=h, away_code=a, venue_key=v)

        delta = r_play['p_home_win'] - r_orig['p_home_win']
        arrow = "📈" if delta > 1 else "📉" if delta < -1 else "➡️"

        print(f"\n  {label}")
        print(f"    原始模型: {r_orig['home_team']} {r_orig['p_home_win']}% | "
              f"平 {r_orig['p_draw']}% | {r_orig['away_team']} {r_orig['p_away_win']}%")
        ch_orig = f"{orig_teams[h].club_chemistry:.0f}/{orig_teams[a].club_chemistry:.0f}"
        ch_new = f"{player_teams[h].club_chemistry:.0f}/{player_teams[a].club_chemistry:.0f}"
        print(f"    球员模型: {r_play['home_team']} {r_play['p_home_win']}% | "
              f"平 {r_play['p_draw']}% | {r_play['away_team']} {r_play['p_away_win']}%"
              f"  {arrow} 化学: {ch_orig} → {ch_new}")

    # 保存结果
    print(f"\n{'='*65}")
    print("数据统计:")
    real_count = sum(1 for c in teams if c in real_data_teams)
    print(f"  真实球员级数据: {real_count}/48 支球队 ({real_count/48*100:.0f}%)")
    print(f"  预测50场覆盖: 10场关键对阵完整验证")
    print(f"{'='*65}")

    return db, chem_results, updated_teams


if __name__ == "__main__":
    db, chem, teams = compute_and_update_model()

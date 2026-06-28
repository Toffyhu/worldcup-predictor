"""
批量采集EA FC 26球员数据 → 构建国家队阵容
===========================================
从fcratings.com抓取所有48支世界杯参赛队的球员级数据。
过滤男女足混合列表，仅保留男子球员。

用法: python3.11 batch_fetch_squads.py
"""

import requests
from bs4 import BeautifulSoup
import re
import sys
import json
import time
from typing import Dict, List, Optional

sys.path.insert(0, '/workspace')
from worldcup_squads import PlayerData, TeamSquad, SquadDatabase, ChemistryEngine

# ═══════════════════════════════════════════════════════════════
# EA FC 国家队URL映射（完整48支）
# ═══════════════════════════════════════════════════════════════

FC_NATIONS_URLS: Dict[str, Dict] = {
    # A组
    "MEX": {"url": "https://www.fcratings.com/nations/mexico-83", "name": "Mexico", "est_players": 30},
    "RSA": {"url": "https://www.fcratings.com/nations/south-africa-140", "name": "South Africa", "est_players": 15},
    "KOR": {"url": "https://www.fcratings.com/nations/korea-republic-167", "name": "Korea Republic", "est_players": 400},
    "CZE": {"url": "https://www.fcratings.com/nations/czech-republic-12", "name": "Czech Republic", "est_players": 104},
    # B组
    "CAN": {"url": "https://www.fcratings.com/nations/canada-70", "name": "Canada", "est_players": 117},
    "BIH": {"url": "https://www.fcratings.com/nations/bosnia-and-herzegovina-8", "name": "Bosnia", "est_players": 55},
    "QAT": {"url": None, "name": "Qatar", "est_players": 5},
    "SUI": {"url": "https://www.fcratings.com/nations/switzerland-47", "name": "Switzerland", "est_players": 275},
    # C组
    "BRA": {"url": "https://www.fcratings.com/nations/brazil-54", "name": "Brazil", "est_players": 370},
    "MAR": {"url": "https://www.fcratings.com/nations/morocco-129", "name": "Morocco", "est_players": 125},
    "HAI": {"url": "https://www.fcratings.com/nations/haiti-80", "name": "Haiti", "est_players": 22},
    "SCO": {"url": "https://www.fcratings.com/nations/scotland-42", "name": "Scotland", "est_players": 278},
    # D组
    "USA": {"url": "https://www.fcratings.com/nations/united-states-95", "name": "United States", "est_players": 661},
    "PAR": {"url": "https://www.fcratings.com/nations/paraguay-58", "name": "Paraguay", "est_players": 155},
    "AUS": {"url": "https://www.fcratings.com/nations/australia-195", "name": "Australia", "est_players": 278},
    "TUR": {"url": "https://www.fcratings.com/nations/turkey-48", "name": "Turkey", "est_players": 262},
    # E组
    "GER": {"url": "https://www.fcratings.com/nations/germany-21", "name": "Germany", "est_players": 1373},
    "CUW": {"url": "https://www.fcratings.com/nations/curacao-85", "name": "Curacao", "est_players": 12},
    "CIV": {"url": "https://www.fcratings.com/nations/cote-divoire-108", "name": "Ivory Coast", "est_players": 121},
    "ECU": {"url": "https://www.fcratings.com/nations/ecuador-57", "name": "Ecuador", "est_players": 107},
    # F组
    "NED": {"url": "https://www.fcratings.com/nations/holland-34", "name": "Netherlands", "est_players": 460},
    "JPN": {"url": "https://www.fcratings.com/nations/japan-163", "name": "Japan", "est_players": 131},
    "TUN": {"url": "https://www.fcratings.com/nations/tunisia-145", "name": "Tunisia", "est_players": 29},
    "SWE": {"url": "https://www.fcratings.com/nations/sweden-46", "name": "Sweden", "est_players": 411},
    # G组
    "BEL": {"url": "https://www.fcratings.com/nations/belgium-7", "name": "Belgium", "est_players": 297},
    "EGY": {"url": "https://www.fcratings.com/nations/egypt-111", "name": "Egypt", "est_players": 9},
    "IRN": {"url": "https://www.fcratings.com/nations/iran-161", "name": "Iran", "est_players": 6},
    "NZL": {"url": "https://www.fcratings.com/nations/new-zealand-198", "name": "New Zealand", "est_players": 54},
    # H组
    "ESP": {"url": "https://www.fcratings.com/nations/spain-45", "name": "Spain", "est_players": 1076},
    "CPV": {"url": "https://www.fcratings.com/nations/cape-verde-islands-104", "name": "Cape Verde", "est_players": 27},
    "KSA": {"url": "https://www.fcratings.com/nations/saudi-arabia-183", "name": "Saudi Arabia", "est_players": 270},
    "URU": {"url": "https://www.fcratings.com/nations/uruguay-60", "name": "Uruguay", "est_players": 212},
    # I组
    "FRA": {"url": "https://www.fcratings.com/nations/france-18", "name": "France", "est_players": 1002},
    "SEN": {"url": "https://www.fcratings.com/nations/senegal-136", "name": "Senegal", "est_players": 120},
    "IRQ": {"url": "https://www.fcratings.com/nations/iraq-162", "name": "Iraq", "est_players": 15},
    "NOR": {"url": "https://www.fcratings.com/nations/norway-36", "name": "Norway", "est_players": 412},
    # J组
    "ARG": {"url": "https://www.fcratings.com/nations/argentina-52", "name": "Argentina", "est_players": 983},
    "ALG": {"url": "https://www.fcratings.com/nations/algeria-97", "name": "Algeria", "est_players": 57},
    "AUT": {"url": "https://www.fcratings.com/nations/austria-4", "name": "Austria", "est_players": 297},
    "JOR": {"url": "https://www.fcratings.com/nations/jordan-164", "name": "Jordan", "est_players": 2},
    # K组
    "POR": {"url": "https://www.fcratings.com/nations/portugal-38", "name": "Portugal", "est_players": 322},
    "COD": {"url": "https://www.fcratings.com/nations/congo-dr-110", "name": "DR Congo", "est_players": 42},
    "UZB": {"url": "https://www.fcratings.com/nations/uzbekistan-191", "name": "Uzbekistan", "est_players": 4},
    "COL": {"url": "https://www.fcratings.com/nations/colombia-56", "name": "Colombia", "est_players": 189},
    # L组
    "ENG": {"url": "https://www.fcratings.com/nations/england-14", "name": "England", "est_players": 1520},
    "CRO": {"url": "https://www.fcratings.com/nations/croatia-10", "name": "Croatia", "est_players": 153},
    "GHA": {"url": "https://www.fcratings.com/nations/ghana-117", "name": "Ghana", "est_players": 117},
    "PAN": {"url": "https://www.fcratings.com/nations/panama-87", "name": "Panama", "est_players": 12},
}

# 女足俱乐部关键词 — 全面覆盖
WOMENS_CLUB_PATTERNS = [
    re.compile(r'\bWomen\b', re.I),
    re.compile(r'W\.F\.C\.', re.I),
    re.compile(r'Lionesses?', re.I),
    re.compile(r'Femenino', re.I),
    re.compile(r'Féminin', re.I),
    re.compile(r'Femminile', re.I),
    re.compile(r'Lyonnes?\b', re.I),
    # NWSL (美国女足联赛) 球队
    re.compile(r'Gotham\s*FC', re.I),
    re.compile(r'Angel\s*City\s*FC', re.I),
    re.compile(r'Chicago\s*Stars\s*FC', re.I),
    re.compile(r'Washington\s*Spirit', re.I),
    re.compile(r'San\s*Diego\s*Wave', re.I),
    re.compile(r'North\s*Carolina\s*Courage', re.I),
    re.compile(r'Racing\s*Louisville', re.I),
    re.compile(r'Kansas\s*City\s*Current', re.I),
    re.compile(r'Houston\s*Dash', re.I),
    re.compile(r'Bay\s*FC', re.I),
    re.compile(r'Utah\s*Royals?\s*FC', re.I),
    re.compile(r'Portland\s*Thorns?\s*FC', re.I),
    re.compile(r'OL\s*Reign', re.I),
    re.compile(r'Seattle\s*Reign', re.I),
    re.compile(r'OL\s*Lyon\s*Féminin', re.I),
]

# 已知女足球员姓名（备选过滤）
KNOWN_WOMEN_PLAYERS = {
    'Ann-Katrin Berger', 'Sara Doorsoun', 'Kathrin Hendrich',
    'Alexandra Popp', 'Lea Schüller', 'Klara Bühl',
    'Svenja Huth', 'Laura Freigang', 'Giulia Gwinn',
    'Janina Minge', 'Sara Däbritz',
    'Alessia Russo', 'Chloe Kelly', 'Lucy Bronze', 'Leah Williamson',
    'Beth Mead', 'Lauren Hemp', 'Millie Bright', 'Georgia Stanway',
    'Lauren James', 'Ella Toone', 'Keira Walsh', 'Hannah Hampton',
    'Alex Greenwood', 'Mary Earps', 'Maya Le Tissier', 'Fran Kirby',
    'Leah Galton', 'Nikita Parris', 'Bethany England', 'Millie Turner',
    'Marie Katoto', 'Kadidiatou Diani', 'Sakina Karchaoui',
    'Grace Geyoro', 'Selma Bacha', 'Sandy Baltimore', 'Clara Mateo',
    'Delphine Cascarino', 'Wendie Renard', 'Kenza Dali',
    'Pauline Peyraud-Magnin', 'Griedge Mbock', 'Constance Picaud',
    'Viviane Asseyi', 'Ouleymata Sarr', 'Melvine Malard',
    'Kessya Bussy', 'Maëlle Lakrar',
}


def is_womens_club(club_name: str) -> bool:
    """判断俱乐部是否为女足球队"""
    for pattern in WOMENS_CLUB_PATTERNS:
        if pattern.search(club_name):
            return True
    return False


def fetch_nation_page(code: str, url: str) -> Optional[List[Dict]]:
    """
    从fcratings.com抓取某国家的Top球员列表
    利用HTML table结构精确提取：姓名、位置( via /positions/ link)、OVR、俱乐部( via /clubs/ link)
    返回 [{name, position, ovr, club}, ...] 仅男足
    """
    if not url:
        return None

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.fcratings.com/lists/all-nations',
            'Cache-Control': 'no-cache',
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️ {code}: HTTP {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table')
        if not table:
            return None

        player_entries = []
        rows = table.find_all('tr')[1:]  # 跳过表头

        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 3:
                continue

            # Cell 1: Player column (name, position, club)
            player_cell = cells[1]

            # Extract name
            name_el = player_cell.select_one('.custom-name')
            name = name_el.get_text(strip=True) if name_el else ''

            # Extract position: find <a> with href containing '/positions/'
            pos_el = player_cell.find('a', href=lambda h: h and '/positions/' in h)
            pos = pos_el.get_text(strip=True) if pos_el else 'CM'

            # Extract club: find <a> with href containing '/clubs/'
            club_el = player_cell.find('a', href=lambda h: h and '/clubs/' in h)
            club = club_el.get_text(strip=True) if club_el else 'Unknown'

            # Cell 2: OVR - 使用 data-sort-value 属性（避免delta后缀干扰）
            ovr_cell = cells[2]
            sort_val = ovr_cell.get('data-sort-value', '')
            ovr = 0
            if sort_val:
                try:
                    ovr = int(sort_val)
                except:
                    pass
            if ovr == 0:
                # 备选：取纯文本中的第一个两位数
                ovr_text = ovr_cell.get_text(strip=True)
                nums = re.findall(r'\b(\d{2})\b', ovr_text)
                ovr = int(nums[0]) if nums else 0

            if name and ovr > 0:
                if is_womens_club(club):
                    continue  # 跳过女足球员
                player_entries.append({
                    'name': name,
                    'position': pos,
                    'ovr': ovr,
                    'club': club,
                })

        if player_entries:
            # 去重
            seen = set()
            unique = []
            for p in player_entries:
                if p['name'] not in seen:
                    seen.add(p['name'])
                    unique.append(p)
            return sorted(unique, key=lambda p: -p['ovr'])

        return None

    except Exception as e:
        print(f"  ⚠️ {code}: 抓取异常 — {e}")
        return None


def parse_nation_data(code: str, players_raw: List[Dict]) -> List[PlayerData]:
    """将原始抓取数据转换为PlayerData列表"""
    result = []
    # 过滤女足球员（俱乐部名 + 已知女足名单双重过滤）
    male_players = []
    for p in players_raw:
        club = p.get('club', '')
        name = p.get('name', '')
        if is_womens_club(club):
            continue
        if name in KNOWN_WOMEN_PLAYERS:
            continue
        male_players.append(p)

    # 去重
    seen = set()
    unique = []
    for p in male_players:
        if p['name'] not in seen:
            seen.add(p['name'])
            unique.append(p)

    for p in unique[:26]:  # 最多保留26人（世界杯标准阵容）
        pos = p.get('position', 'CM')
        ovr = p.get('ovr', 50)
        club = p.get('club', 'Unknown')
        name = p.get('name', 'Unknown')
        # 推断联赛（基于俱乐部名称）
        league = infer_league(club)

        is_captain = (len(result) == 0)  # OVR最高的预设为队长
        is_star = ovr >= 82

        result.append(PlayerData(
            name=name,
            position=pos,
            ovr=ovr,
            club=club,
            club_league=league,
            nationality=code,
            is_captain=is_captain,
            is_star=is_star,
        ))
    return result


# 常见俱乐部→联赛映射
KNOWN_CLUB_LEAGUES = {
    'Real Madrid': 'La Liga', 'FC Barcelona': 'La Liga', 'Atlético Madrid': 'La Liga',
    'Manchester City': 'Premier League', 'Manchester United': 'Premier League',
    'Arsenal': 'Premier League', 'Liverpool': 'Premier League', 'Chelsea': 'Premier League',
    'Tottenham': 'Premier League', 'Aston Villa': 'Premier League',
    'FC Bayern Munich': 'Bundesliga', 'Borussia Dortmund': 'Bundesliga',
    'RB Leipzig': 'Bundesliga', 'Bayer 04 Leverkusen': 'Bundesliga',
    'Inter Milan': 'Serie A', 'AC Milan': 'Serie A', 'Juventus': 'Serie A',
    'Paris Saint-Germain': 'Ligue 1', 'Olympique de Marseille': 'Ligue 1',
    'SL Benfica': 'Liga Portugal', 'FC Porto': 'Liga Portugal', 'Sporting CP': 'Liga Portugal',
    'Fenerbahçe': 'Super Lig', 'Galatasaray': 'Super Lig',
    'Ajax': 'Eredivisie', 'Feyenoord': 'Eredivisie',
    'Al-Nassr': 'Saudi Pro League', 'Al-Hilal': 'Saudi Pro League', 'Al-Ittihad': 'Saudi Pro League',
}

def infer_league(club: str) -> str:
    """基于俱乐部名称推断所属联赛"""
    for pattern, league in KNOWN_CLUB_LEAGUES.items():
        if pattern.lower() in club.lower():
            return league
    # 默认为Unknown
    return 'Unknown'


def estimate_squad_from_opr(code: str, base_ovr: int) -> List[PlayerData]:
    """
    对EA FC数据不足的球队，基于OPR生成估算阵容。
    使用 unique club/league 避免虚假化学分。
    """
    # 已知的著名球员（for小球队）
    known_stars = {
        'EGY': [('Mohamed Salah', 'RW', 89, 'Liverpool', 'Premier League')],
        'IRN': [('Mehdi Taremi', 'ST', 82, 'Inter Milan', 'Serie A'),
                ('Sardar Azmoun', 'ST', 80, 'AS Roma', 'Serie A')],
        'NZL': [('Chris Wood', 'ST', 78, 'Nottingham Forest', 'Premier League')],
        'QAT': [('Akram Afif', 'LW', 75, 'Al-Sadd', 'Stars League')],
        'CUW': [('Leandro Bacuna', 'CM', 72, 'Groningen', 'Eredivisie')],
        'RSA': [('Percy Tau', 'RW', 76, 'Al Ahly SC', 'Egyptian League')],
        'PAN': [('Adalberto Carrasquilla', 'CM', 74, 'Houston Dynamo', 'MLS')],
        'IRQ': [('Aymen Hussein', 'ST', 70, 'Al-Quwa Al-Jawiya', 'Iraqi League')],
        'JOR': [('Musa Al-Taamari', 'RW', 72, 'Montpellier HSC', 'Ligue 1')],
        'UZB': [('Eldor Shomurodov', 'ST', 75, 'Cagliari', 'Serie A')],
        'HKG': [],
    }

    players = []
    positions_list = ['GK', 'CB', 'CB', 'RB', 'LB', 'CDM', 'CM', 'CM', 'CAM', 'LW', 'RW', 'ST', 'ST',
                     'CB', 'CM', 'ST', 'RB', 'LB', 'CDM', 'LW', 'RW', 'GK', 'CAM']

    # 先添加已知球星
    stars = known_stars.get(code, [])
    for name, pos, ovr, club, league in stars:
        players.append(PlayerData(
            name=name, position=pos, ovr=ovr, club=club, club_league=league,
            nationality=code, is_captain=(len(players) == 0), is_star=(ovr >= 82),
        ))

    # 填充剩余位置 — 每个球员使用唯一的club/league以避免虚假化学分
    rng_seed = sum(ord(c) for c in code)
    for i in range(len(players), min(len(positions_list), 23)):
        pos = positions_list[i]
        variance = (i * 2 + (rng_seed % 10)) / 10
        player_ovr = max(55, min(85, int(base_ovr + 5 - variance * 3)))
        # 每个估算球员使用唯一ID作为club名，避免虚假化学
        fake_club = f'{code}_EST_{i+1}'
        fake_league = f'{code}_LEAGUE'
        players.append(PlayerData(
            name=f'{code} #{i+1}', position=pos, ovr=player_ovr,
            club=fake_club, club_league=fake_league,
            nationality=code, is_captain=False, is_star=(player_ovr >= 82),
        ))

    return players


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def build_all_squads() -> SquadDatabase:
    """
    构建全部48支球队的阵容数据
    1. 有EA FC足够数据的 → 从fcratings.com抓取
    2. 数据不足的 → 基于OPR估算
    """
    from worldcup_model import load_2026_teams
    team_data = load_2026_teams()

    db = SquadDatabase()
    # 清理自动生成的估算阵容
    db.squads = {}
    fetched_count = 0
    estimated_count = 0

    print("=" * 65)
    print("批量采集EA FC 26球员数据")
    print("=" * 65)

    for code, info in FC_NATIONS_URLS.items():
        td = team_data.get(code)
        if not td:
            continue

        # 跳过已手动构建的核心球队
        if code in ('FRA', 'ARG', 'BRA', 'ENG', 'GER', 'ESP', 'POR', 'NED', 'BEL', 'CRO'):
            # 这些已经在SquadDatabase里了
            continue

        url = info['url']
        est_count = info['est_players']

        # 判断是否值得抓取：预期球员数 >= 20 且有URL
        if url and est_count >= 20:
            raw_players = fetch_nation_page(code, url)
            if raw_players and len(raw_players) >= 11:
                players = parse_nation_data(code, raw_players)
                if len(players) >= 11:
                    db.squads[code] = TeamSquad(
                        code=code,
                        name=td.name,
                        coach_name=f'{td.name} Coach',
                        training_camp_days=14,
                        captain_name=players[0].name,
                        probable_starting_xi=[p.name for p in players[:11]],
                        players=players,
                    )
                    fetched_count += 1
                    print(f"  ✅ {code} {td.name:<15} 抓取 {len(players)}人"
                          f"  OVR范围{min(p.ovr for p in players)}-{max(p.ovr for p in players)}")
                    time.sleep(0.5)  # 礼貌延迟
                    continue
                else:
                    print(f"  ⚠️ {code} {td.name:<15} 男足球员不足({len(players)}) → 估算")
            else:
                print(f"  ⚠️ {code} {td.name:<15} 抓取失败 → 估算")
        else:
            print(f"  ℹ️ {code} {td.name:<15} EA FC数据有限(~{est_count}人) → 估算")

        # 估算后备
        base_ovr = max(60, int((td.opr_attack + td.opr_defense) / 2))
        players = estimate_squad_from_opr(code, base_ovr)
        db.squads[code] = TeamSquad(
            code=code, name=td.name, coach_name=f'{td.name} Coach',
            training_camp_days=14, captain_name=players[0].name,
            probable_starting_xi=[p.name for p in players[:11]],
            players=players,
        )
        estimated_count += 1

    print(f"\n{'='*65}")
    print(f"采集完成: {fetched_count}队从EA FC抓取, {estimated_count}队估算")
    print(f"总球队数: {len(db.squads)}")
    print(f"{'='*65}")

    return db


def compute_chemistry_all(db: SquadDatabase) -> Dict:
    """计算全部48支球队的化学分"""
    engine = ChemistryEngine()
    results = {}

    print("\n" + "=" * 65)
    print("计算全48队化学分")
    print("=" * 65)
    print(f"{'队名':<16} {'化学分':>8} {'同俱乐部':>8} {'同联赛':>8} {'首发OVR':>8}")
    print("-" * 65)

    for code, squad in db.squads.items():
        chem = engine.compute(squad)
        xi = squad.get_starting_xi_sorted()
        avg_ovr = sum(p.ovr for p in xi) / max(len(xi), 1)

        results[code] = {
            'chemistry_score': chem['chemistry_score'],
            'club_chemistry': chem['club_chemistry'],
            'league_familiarity': chem['league_familiarity'],
            'avg_ovr': round(avg_ovr, 1),
            'breakdown': chem['breakdown'],
        }
        print(f"  {squad.name:<14} {chem['chemistry_score']:>8.1f}"
              f" {chem['club_chemistry']:>8.1f} {chem['league_familiarity']:>8.1f}"
              f" {avg_ovr:>8.1f}")

    # 归一化
    print("\n归一化化学分到 [20, 90] 范围...")
    core_squads = {code: db.squads[code] for code in db.squads}
    normalized = engine.normalize_across_teams(core_squads)

    for code, chem in normalized.items():
        results[code]['club_chemistry_norm'] = chem['club_chemistry_norm']
        results[code]['league_familiarity_norm'] = chem['league_familiarity_norm']

    return results


def update_model_with_player_chemistry(db: SquadDatabase, chem_results: Dict) -> dict:
    """将球员级化学分更新到TeamData"""
    from worldcup_model import load_2026_teams
    import copy

    teams = load_2026_teams()
    updated = {}

    for code, td in teams.items():
        if code in chem_results:
            chem = chem_results[code]
            new_td = copy.deepcopy(td)
            new_td.club_chemistry = chem['club_chemistry_norm']
            new_td.league_familiarity = chem['league_familiarity_norm']
            updated[code] = new_td
        else:
            updated[code] = td

    return updated


if __name__ == "__main__":
    # 构建阵容数据库
    db = build_all_squads()

    # 计算化学分
    chem_results = compute_chemistry_all(db)

    # 输出化学分排名
    print("\n" + "=" * 65)
    print("全48队化学分排名（归一化后）")
    print("=" * 65)
    ranked = sorted(chem_results.items(), key=lambda x: -x[1]['chemistry_score'])
    for rank, (code, chem) in enumerate(ranked, 1):
        print(f"  {rank:>2}. {code:<4} {chem['chemistry_score']:>5.1f}"
              f"  (同俱乐部={chem['club_chemistry_norm']:>5.1f}"
              f"  同联赛={chem['league_familiarity_norm']:>5.1f}"
              f"  总评={chem['avg_ovr']:>5.1f})")

    # 保存结果供后续使用
    import json
    with open('/workspace/squad_chemistry_results.json', 'w') as f:
        # 转换不可JSON序列化的字段
        serializable = {}
        for k, v in chem_results.items():
            serializable[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 squad_chemistry_results.json")

    # 更新模型
    from worldcup_model import WorldCupPredictor, load_2026_teams

    model_teams = update_model_with_player_chemistry(db, chem_results)

    print("\n" + "=" * 65)
    print("球员级化学模型 vs 原始模型 — 关键对阵预测")
    print("=" * 65)

    predictor = WorldCupPredictor()
    predictor_player = WorldCupPredictor(use_player_chemistry=True)
    player_teams = predictor_player.override_team_chemistry(load_2026_teams())

    test_matches = [
        ("FRA", "ENG", "Los Angeles"), ("GER", "ESP", "Houston"),
        ("ARG", "POR", "Atlanta"), ("BRA", "NED", "Dallas"),
        ("USA", "TUR", "Los Angeles"), ("JPN", "SWE", "Kansas City"),
        ("MAR", "SCO", "Boston"), ("SEN", "NOR", "New York / New Jersey"),
    ]

    for h, a, v in test_matches:
        r_orig = predictor.predict_match(load_2026_teams()[h], load_2026_teams()[a],
                                          home_code=h, away_code=a, venue_key=v)
        r_play = predictor.predict_match(player_teams[h], player_teams[a],
                                          home_code=h, away_code=a, venue_key=v)
        d = r_play['p_home_win'] - r_orig['p_home_win']
        arrow = "▲" if d > 0 else "▼" if d < 0 else "="
        print(f"  {h} vs {a}: {r_orig['p_home_win']}%→{r_play['p_home_win']}%({arrow}{abs(d):.1f}%)"
              f"  | 原化学:化学({load_2026_teams()[h].club_chemistry:.0f}/{load_2026_teams()[a].club_chemistry:.0f})"
              f"  → 球员:({player_teams[h].club_chemistry:.0f}/{player_teams[a].club_chemistry:.0f})")

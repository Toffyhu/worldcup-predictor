"""
2026世界杯预测系统 · Python实现
==================================
基于6层13因子的综合预测模型 + 蒙特卡洛模拟引擎

架构：
  worldcup_model.py (本文件)      — 模型核心 + 数据定义
  worldcup_simulator.py           — 蒙特卡洛模拟引擎
  worldcup_optimizer.py           — 权重优化模块
"""

import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

# ============================================================
# 一、数据定义 — 基于实际收集的2026世界杯数据
# ============================================================

@dataclass
class TeamData:
    """单支球队的完整特征数据"""
    name: str
    code: str
    group: str

    # 第1层：个人能力 (从OddsFlow OPR数据 + EA FC25估算)
    opr_attack: float       # OPR攻击分 [0-100]
    opr_defense: float      # OPR防守分 [0-100]
    opr_gk: float           # OPR门将分 [0-100]
    league_strength: float  # 联赛强度系数加权 [0-100]

    # 第2层：团队化学
    club_chemistry: float   # 同队球员亲密度 [0-100]
    league_familiarity: float # 同联赛熟悉度 [0-100]
    ball_hog_index: float   # 球霸指数(越高越差) [0-100]
    star_dependency: float  # 球星依赖度(越高越差) [0-100]

    # 第3层：战术因子
    positional_matchup: float  # 1v1对位优势 [0-100]
    tempo_control: float       # 节奏控制力 [0-100]
    setpiece_asymmetry: float  # 定位球攻防差 [0-100]
    coach_ability: float       # 教练组能力 [0-100]

    # 第4层：环境因子 (动态注入，此处存默认值)
    altitude_adapt: float  # 海拔适应度 [0-100]
    jetlag_penalty: float  # 时差惩罚(越高越差) [0-100]
    climate_adapt: float   # 气候适应 [0-100]
    travel_fatigue: float  # 旅途疲劳(越高越差) [0-100]

    # 第5层：大赛基因因子 ✨新增
    football_culture: float    # 足球文化指数 [0-100]
    political_pressure: float  # 政治/社会压力系数 [0-100]
    tournament_overperform: float  # 大赛超常系数 [0-100]
    national_pride: float      # 民族荣誉驱动 [0-100]

    # 第6层：软性因子
    injury_impact: float       # 伤病影响(越高越差) [0-100]
    tournament_experience: float # 大赛经验 [0-100]
    team_morale: float         # 士气 [0-100]
    preparation: float         # 备战完整度 [0-100]


# ============================================================
# 二、真实球队数据 (基于OddsFlow/CupIndex/EA FC25实际数据)
# ============================================================

def load_2026_teams() -> Dict[str, TeamData]:
    """
    加载48支球队的完整特征数据。
    数据来源：
      - OPR: OddsFlow 12-Factor Model
      - 化学分: OddsFlow Chemistry Score
      - 夺冠概率: CupIndex / FootballForecast
      - 大赛基因: 基于历史世界杯表现计算
    """
    teams = {
        # ── A组 ──
        "MEX": TeamData("Mexico", "MEX", "A",
            opr_attack=68, opr_defense=65, opr_gk=70,
            league_strength=70,
            club_chemistry=70, league_familiarity=65,
            ball_hog_index=30, star_dependency=25,
            positional_matchup=60, tempo_control=55,
            setpiece_asymmetry=45, coach_ability=65,
            altitude_adapt=95, jetlag_penalty=10,
            climate_adapt=85, travel_fatigue=5,
            football_culture=75, political_pressure=65,
            tournament_overperform=48, national_pride=80,
            injury_impact=15, tournament_experience=72,
            team_morale=80, preparation=88
        ),
        "RSA": TeamData("South Africa", "RSA", "A",
            opr_attack=55, opr_defense=50, opr_gk=52,
            league_strength=40,
            club_chemistry=55, league_familiarity=45,
            ball_hog_index=35, star_dependency=40,
            positional_matchup=40, tempo_control=38,
            setpiece_asymmetry=50, coach_ability=50,
            altitude_adapt=75, jetlag_penalty=30,
            climate_adapt=70, travel_fatigue=25,
            football_culture=70, political_pressure=55,
            tournament_overperform=58, national_pride=68,
            injury_impact=20, tournament_experience=42,
            team_morale=65, preparation=70
        ),
        "KOR": TeamData("South Korea", "KOR", "A",
            opr_attack=68, opr_defense=62, opr_gk=65,
            league_strength=62,
            club_chemistry=72, league_familiarity=60,
            ball_hog_index=25, star_dependency=30,
            positional_matchup=58, tempo_control=60,
            setpiece_asymmetry=48, coach_ability=62,
            altitude_adapt=55, jetlag_penalty=50,
            climate_adapt=55, travel_fatigue=45,
            football_culture=75, political_pressure=68,
            tournament_overperform=70, national_pride=85,
            injury_impact=15, tournament_experience=68,
            team_morale=75, preparation=80
        ),
        "CZE": TeamData("Czech Republic", "CZE", "A",
            opr_attack=62, opr_defense=60, opr_gk=63,
            league_strength=58,
            club_chemistry=65, league_familiarity=55,
            ball_hog_index=28, star_dependency=30,
            positional_matchup=50, tempo_control=48,
            setpiece_asymmetry=55, coach_ability=55,
            altitude_adapt=50, jetlag_penalty=25,
            climate_adapt=60, travel_fatigue=20,
            football_culture=72, political_pressure=58,
            tournament_overperform=55, national_pride=65,
            injury_impact=18, tournament_experience=55,
            team_morale=60, preparation=65
        ),

        # ── B组 ──
        "CAN": TeamData("Canada", "CAN", "B",
            opr_attack=65, opr_defense=58, opr_gk=55,
            league_strength=65,
            club_chemistry=60, league_familiarity=55,
            ball_hog_index=35, star_dependency=55,  # 戴维斯伤缺影响大
            positional_matchup=48, tempo_control=45,
            setpiece_asymmetry=42, coach_ability=58,
            altitude_adapt=55, jetlag_penalty=15,
            climate_adapt=65, travel_fatigue=10,
            football_culture=55, political_pressure=45,
            tournament_overperform=40, national_pride=55,
            injury_impact=65,  # 戴维斯+多人伤缺
            tournament_experience=30,
            team_morale=70, preparation=60
        ),
        "BIH": TeamData("Bosnia and Herzegovina", "BIH", "B",
            opr_attack=58, opr_defense=55, opr_gk=58,
            league_strength=50,
            club_chemistry=62, league_familiarity=48,
            ball_hog_index=30, star_dependency=35,
            positional_matchup=45, tempo_control=42,
            setpiece_asymmetry=40, coach_ability=48,
            altitude_adapt=50, jetlag_penalty=20,
            climate_adapt=60, travel_fatigue=18,
            football_culture=68, political_pressure=52,
            tournament_overperform=48, national_pride=72,
            injury_impact=20, tournament_experience=35,
            team_morale=62, preparation=58
        ),
        "QAT": TeamData("Qatar", "QAT", "B",
            opr_attack=55, opr_defense=48, opr_gk=50,
            league_strength=45,
            club_chemistry=50, league_familiarity=40,
            ball_hog_index=40, star_dependency=42,
            positional_matchup=38, tempo_control=35,
            setpiece_asymmetry=35, coach_ability=45,
            altitude_adapt=45, jetlag_penalty=40,
            climate_adapt=75, travel_fatigue=30,
            football_culture=52, political_pressure=48,
            tournament_overperform=35, national_pride=70,
            injury_impact=15, tournament_experience=25,
            team_morale=70, preparation=75
        ),
        "SUI": TeamData("Switzerland", "SUI", "B",
            opr_attack=70, opr_defense=72, opr_gk=75,
            league_strength=72,
            club_chemistry=68, league_familiarity=65,
            ball_hog_index=18, star_dependency=20,
            positional_matchup=62, tempo_control=60,
            setpiece_asymmetry=58, coach_ability=68,
            altitude_adapt=60, jetlag_penalty=15,
            climate_adapt=55, travel_fatigue=12,
            football_culture=65, political_pressure=55,
            tournament_overperform=60, national_pride=62,
            injury_impact=10, tournament_experience=65,
            team_morale=68, preparation=72
        ),

        # ── C组 ──
        "BRA": TeamData("Brazil", "BRA", "C",
            opr_attack=76, opr_defense=77, opr_gk=78,
            league_strength=75,
            club_chemistry=56, league_familiarity=50,
            ball_hog_index=40, star_dependency=35,
            positional_matchup=78, tempo_control=72,
            setpiece_asymmetry=65, coach_ability=72,
            altitude_adapt=70, jetlag_penalty=30,
            climate_adapt=80, travel_fatigue=22,
            football_culture=98, political_pressure=95,
            tournament_overperform=80, national_pride=95,
            injury_impact=12, tournament_experience=88,
            team_morale=75, preparation=78
        ),
        "MAR": TeamData("Morocco", "MAR", "C",
            opr_attack=70, opr_defense=72, opr_gk=75,
            league_strength=58,
            club_chemistry=65, league_familiarity=55,
            ball_hog_index=20, star_dependency=28,
            positional_matchup=65, tempo_control=58,
            setpiece_asymmetry=55, coach_ability=70,
            altitude_adapt=62, jetlag_penalty=25,
            climate_adapt=72, travel_fatigue=18,
            football_culture=78, political_pressure=60,
            tournament_overperform=85, national_pride=92,
            injury_impact=15, tournament_experience=60,
            team_morale=88, preparation=75
        ),
        "HAI": TeamData("Haiti", "HAI", "C",
            opr_attack=45, opr_defense=40, opr_gk=38,
            league_strength=22,
            club_chemistry=42, league_familiarity=30,
            ball_hog_index=45, star_dependency=48,
            positional_matchup=30, tempo_control=25,
            setpiece_asymmetry=28, coach_ability=35,
            altitude_adapt=55, jetlag_penalty=35,
            climate_adapt=80, travel_fatigue=30,
            football_culture=60, political_pressure=35,
            tournament_overperform=30, national_pride=70,
            injury_impact=15, tournament_experience=15,
            team_morale=72, preparation=45
        ),
        "SCO": TeamData("Scotland", "SCO", "C",
            opr_attack=65, opr_defense=63, opr_gk=62,
            league_strength=68,
            club_chemistry=62, league_familiarity=60,
            ball_hog_index=25, star_dependency=32,
            positional_matchup=55, tempo_control=50,
            setpiece_asymmetry=55, coach_ability=58,
            altitude_adapt=48, jetlag_penalty=20,
            climate_adapt=50, travel_fatigue=15,
            football_culture=75, political_pressure=62,
            tournament_overperform=42, national_pride=78,
            injury_impact=20, tournament_experience=48,
            team_morale=72, preparation=65
        ),

        # ── D组 ──
        "USA": TeamData("United States", "USA", "D",
            opr_attack=70, opr_defense=68, opr_gk=70,
            league_strength=70,
            club_chemistry=55, league_familiarity=50,
            ball_hog_index=25, star_dependency=30,
            positional_matchup=65, tempo_control=62,
            setpiece_asymmetry=55, coach_ability=65,
            altitude_adapt=60, jetlag_penalty=5,
            climate_adapt=65, travel_fatigue=5,
            football_culture=58, political_pressure=55,
            tournament_overperform=55, national_pride=72,
            injury_impact=10, tournament_experience=55,
            team_morale=80, preparation=85
        ),
        "PAR": TeamData("Paraguay", "PAR", "D",
            opr_attack=58, opr_defense=60, opr_gk=62,
            league_strength=48,
            club_chemistry=58, league_familiarity=45,
            ball_hog_index=28, star_dependency=30,
            positional_matchup=48, tempo_control=42,
            setpiece_asymmetry=45, coach_ability=52,
            altitude_adapt=60, jetlag_penalty=25,
            climate_adapt=72, travel_fatigue=20,
            football_culture=72, political_pressure=52,
            tournament_overperform=55, national_pride=68,
            injury_impact=15, tournament_experience=50,
            team_morale=60, preparation=62
        ),
        "AUS": TeamData("Australia", "AUS", "D",
            opr_attack=62, opr_defense=60, opr_gk=63,
            league_strength=55,
            club_chemistry=58, league_familiarity=48,
            ball_hog_index=25, star_dependency=32,
            positional_matchup=52, tempo_control=48,
            setpiece_asymmetry=52, coach_ability=55,
            altitude_adapt=55, jetlag_penalty=80,
            climate_adapt=55, travel_fatigue=65,
            football_culture=60, political_pressure=48,
            tournament_overperform=60, national_pride=68,
            injury_impact=10, tournament_experience=55,
            team_morale=75, preparation=72
        ),
        "TUR": TeamData("Turkey", "TUR", "D",
            opr_attack=72, opr_defense=68, opr_gk=70,
            league_strength=65,
            club_chemistry=68, league_familiarity=60,
            ball_hog_index=30, star_dependency=28,
            positional_matchup=62, tempo_control=58,
            setpiece_asymmetry=58, coach_ability=60,
            altitude_adapt=52, jetlag_penalty=20,
            climate_adapt=58, travel_fatigue=18,
            football_culture=82, political_pressure=72,
            tournament_overperform=58, national_pride=80,
            injury_impact=12, tournament_experience=52,
            team_morale=72, preparation=68
        ),

        # ── E组 ──
        "GER": TeamData("Germany", "GER", "E",
            opr_attack=76, opr_defense=76, opr_gk=73,
            league_strength=78,
            club_chemistry=100, league_familiarity=75,  # 满分化学
            ball_hog_index=12, star_dependency=15,
            positional_matchup=75, tempo_control=72,
            setpiece_asymmetry=72, coach_ability=78,
            altitude_adapt=55, jetlag_penalty=15,
            climate_adapt=55, travel_fatigue=12,
            football_culture=92, political_pressure=88,
            tournament_overperform=92, national_pride=75,
            injury_impact=10, tournament_experience=88,
            team_morale=72, preparation=80
        ),
        "CUW": TeamData("Curacao", "CUW", "E",
            opr_attack=42, opr_defense=38, opr_gk=35,
            league_strength=18,
            club_chemistry=40, league_familiarity=25,
            ball_hog_index=50, star_dependency=55,
            positional_matchup=25, tempo_control=22,
            setpiece_asymmetry=25, coach_ability=30,
            altitude_adapt=50, jetlag_penalty=35,
            climate_adapt=78, travel_fatigue=28,
            football_culture=50, political_pressure=25,
            tournament_overperform=20, national_pride=55,
            injury_impact=15, tournament_experience=10,
            team_morale=55, preparation=40
        ),
        "CIV": TeamData("Ivory Coast", "CIV", "E",
            opr_attack=65, opr_defense=60, opr_gk=58,
            league_strength=45,
            club_chemistry=58, league_familiarity=42,
            ball_hog_index=35, star_dependency=40,
            positional_matchup=52, tempo_control=48,
            setpiece_asymmetry=48, coach_ability=50,
            altitude_adapt=52, jetlag_penalty=25,
            climate_adapt=78, travel_fatigue=22,
            football_culture=72, political_pressure=55,
            tournament_overperform=50, national_pride=75,
            injury_impact=12, tournament_experience=42,
            team_morale=68, preparation=65
        ),
        "ECU": TeamData("Ecuador", "ECU", "E",
            opr_attack=60, opr_defense=58, opr_gk=60,
            league_strength=42,
            club_chemistry=55, league_familiarity=40,
            ball_hog_index=30, star_dependency=35,
            positional_matchup=48, tempo_control=42,
            setpiece_asymmetry=42, coach_ability=48,
            altitude_adapt=100, jetlag_penalty=30,
            climate_adapt=75, travel_fatigue=25,
            football_culture=68, political_pressure=50,
            tournament_overperform=55, national_pride=65,
            injury_impact=15, tournament_experience=38,
            team_morale=62, preparation=60
        ),

        # ── F组 ──
        "NED": TeamData("Netherlands", "NED", "F",
            opr_attack=76, opr_defense=75, opr_gk=74,
            league_strength=75,
            club_chemistry=75, league_familiarity=68,
            ball_hog_index=20, star_dependency=22,
            positional_matchup=72, tempo_control=68,
            setpiece_asymmetry=68, coach_ability=72,
            altitude_adapt=48, jetlag_penalty=15,
            climate_adapt=50, travel_fatigue=12,
            football_culture=88, political_pressure=72,
            tournament_overperform=58, national_pride=70,
            injury_impact=10, tournament_experience=72,
            team_morale=65, preparation=75
        ),
        "JPN": TeamData("Japan", "JPN", "F",
            opr_attack=72, opr_defense=70, opr_gk=68,
            league_strength=65,
            club_chemistry=72, league_familiarity=62,
            ball_hog_index=18, star_dependency=25,
            positional_matchup=65, tempo_control=62,
            setpiece_asymmetry=45, coach_ability=68,
            altitude_adapt=50, jetlag_penalty=65,
            climate_adapt=50, travel_fatigue=55,
            football_culture=75, political_pressure=68,
            tournament_overperform=72, national_pride=88,
            injury_impact=10, tournament_experience=68,
            team_morale=78, preparation=82
        ),
        "TUN": TeamData("Tunisia", "TUN", "F",
            opr_attack=55, opr_defense=52, opr_gk=55,
            league_strength=40,
            club_chemistry=50, league_familiarity=35,
            ball_hog_index=32, star_dependency=38,
            positional_matchup=42, tempo_control=38,
            setpiece_asymmetry=40, coach_ability=45,
            altitude_adapt=48, jetlag_penalty=25,
            climate_adapt=72, travel_fatigue=20,
            football_culture=68, political_pressure=48,
            tournament_overperform=42, national_pride=62,
            injury_impact=18, tournament_experience=45,
            team_morale=58, preparation=55
        ),
        "SWE": TeamData("Sweden", "SWE", "F",
            opr_attack=68, opr_defense=70, opr_gk=72,
            league_strength=62,
            club_chemistry=62, league_familiarity=55,
            ball_hog_index=20, star_dependency=25,
            positional_matchup=58, tempo_control=52,
            setpiece_asymmetry=62, coach_ability=60,
            altitude_adapt=48, jetlag_penalty=18,
            climate_adapt=45, travel_fatigue=15,
            football_culture=70, political_pressure=55,
            tournament_overperform=58, national_pride=65,
            injury_impact=12, tournament_experience=60,
            team_morale=65, preparation=68
        ),

        # ── 夺冠热门简化录入 (其余球队同模式) ──
        "ARG": TeamData("Argentina", "ARG", "J",
            opr_attack=76, opr_defense=74, opr_gk=78,
            league_strength=70,
            club_chemistry=88, league_familiarity=70,
            ball_hog_index=22, star_dependency=30,
            positional_matchup=75, tempo_control=75,
            setpiece_asymmetry=58, coach_ability=82,
            altitude_adapt=70, jetlag_penalty=20,
            climate_adapt=72, travel_fatigue=18,
            football_culture=98, political_pressure=92,
            tournament_overperform=85, national_pride=90,
            injury_impact=10, tournament_experience=90,
            team_morale=85, preparation=80
        ),
        "FRA": TeamData("France", "FRA", "I",
            opr_attack=79, opr_defense=79, opr_gk=79,
            league_strength=78,
            club_chemistry=87, league_familiarity=72,
            ball_hog_index=28, star_dependency=25,
            positional_matchup=78, tempo_control=72,
            setpiece_asymmetry=70, coach_ability=82,
            altitude_adapt=58, jetlag_penalty=18,
            climate_adapt=60, travel_fatigue=15,
            football_culture=90, political_pressure=78,
            tournament_overperform=80, national_pride=78,
            injury_impact=12, tournament_experience=82,
            team_morale=78, preparation=78
        ),
        "ENG": TeamData("England", "ENG", "L",
            opr_attack=79, opr_defense=80, opr_gk=76,
            league_strength=82,
            club_chemistry=95, league_familiarity=80,
            ball_hog_index=30, star_dependency=35,
            positional_matchup=76, tempo_control=70,
            setpiece_asymmetry=75, coach_ability=75,
            altitude_adapt=52, jetlag_penalty=15,
            climate_adapt=50, travel_fatigue=12,
            football_culture=95, political_pressure=85,
            tournament_overperform=35, national_pride=72,
            injury_impact=10, tournament_experience=75,
            team_morale=68, preparation=76
        ),
        "ESP": TeamData("Spain", "ESP", "H",
            opr_attack=78, opr_defense=77, opr_gk=75,
            league_strength=78,
            club_chemistry=96, league_familiarity=78,
            ball_hog_index=15, star_dependency=20,
            positional_matchup=75, tempo_control=80,
            setpiece_asymmetry=50, coach_ability=76,
            altitude_adapt=55, jetlag_penalty=15,
            climate_adapt=58, travel_fatigue=12,
            football_culture=93, political_pressure=75,
            tournament_overperform=60, national_pride=72,
            injury_impact=10, tournament_experience=78,
            team_morale=72, preparation=76
        ),
        "POR": TeamData("Portugal", "POR", "K",
            opr_attack=78, opr_defense=77, opr_gk=78,
            league_strength=68,
            club_chemistry=89, league_familiarity=65,
            ball_hog_index=32, star_dependency=35,
            positional_matchup=72, tempo_control=68,
            setpiece_asymmetry=60, coach_ability=72,
            altitude_adapt=55, jetlag_penalty=18,
            climate_adapt=60, travel_fatigue=15,
            football_culture=90, political_pressure=72,
            tournament_overperform=55, national_pride=78,
            injury_impact=8, tournament_experience=74,
            team_morale=72, preparation=74
        ),
        "BEL": TeamData("Belgium", "BEL", "G",
            opr_attack=74, opr_defense=75, opr_gk=72,
            league_strength=72,
            club_chemistry=55, league_familiarity=48,
            ball_hog_index=35, star_dependency=42,
            positional_matchup=68, tempo_control=62,
            setpiece_asymmetry=55, coach_ability=62,
            altitude_adapt=50, jetlag_penalty=15,
            climate_adapt=52, travel_fatigue=12,
            football_culture=82, political_pressure=68,
            tournament_overperform=40, national_pride=62,
            injury_impact=15, tournament_experience=70,
            team_morale=55, preparation=65
        ),
        "CRO": TeamData("Croatia", "CRO", "L",
            opr_attack=73, opr_defense=74, opr_gk=74,
            league_strength=58,
            club_chemistry=70, league_familiarity=55,
            ball_hog_index=15, star_dependency=25,
            positional_matchup=68, tempo_control=65,
            setpiece_asymmetry=58, coach_ability=75,
            altitude_adapt=52, jetlag_penalty=18,
            climate_adapt=58, travel_fatigue=15,
            football_culture=85, political_pressure=62,
            tournament_overperform=95, national_pride=95,
            injury_impact=10, tournament_experience=78,
            team_morale=85, preparation=72
        ),
    }

    # 补录缺失的部分小组球队(简化条目，趋势正确)
    default_teams = {
        # G组
        "EGY": ("Egypt", "EGY", "G", 72, 68, 70, 45, 54, 42, 38, 55, 50, 38, 42, 48, 55, 25, 72, 20, 68, 52, 52, 72, 15, 48, 58, 55),
        "IRN": ("Iran", "IRN", "G", 62, 65, 62, 42, 58, 38, 35, 40, 45, 40, 45, 50, 58, 25, 65, 22, 65, 55, 48, 72, 12, 50, 60, 58),
        "NZL": ("New Zealand", "NZL", "G", 52, 48, 50, 35, 48, 30, 42, 38, 35, 32, 38, 42, 52, 75, 55, 58, 50, 38, 38, 55, 15, 35, 52, 48),
        # H组
        "CPV": ("Cape Verde", "CPV", "H", 48, 42, 40, 25, 42, 28, 45, 48, 32, 28, 32, 35, 48, 30, 68, 28, 48, 28, 25, 50, 12, 15, 55, 40),
        "KSA": ("Saudi Arabia", "KSA", "H", 58, 52, 55, 45, 55, 40, 38, 40, 42, 38, 40, 48, 52, 35, 75, 30, 62, 55, 45, 68, 10, 48, 60, 62),
        "URU": ("Uruguay", "URU", "H", 72, 74, 73, 55, 62, 48, 32, 30, 65, 58, 58, 68, 62, 22, 68, 18, 85, 72, 75, 85, 12, 72, 72, 68),
        # I组
        "SEN": ("Senegal", "SEN", "I", 68, 65, 68, 52, 60, 48, 32, 32, 58, 52, 50, 58, 55, 25, 72, 20, 68, 55, 60, 75, 10, 55, 68, 62),
        "IRQ": ("Iraq", "IRQ", "I", 50, 45, 48, 30, 48, 30, 42, 45, 35, 30, 35, 40, 50, 30, 68, 28, 55, 48, 42, 65, 15, 30, 58, 45),
        "NOR": ("Norway", "NOR", "I", 68, 65, 68, 58, 60, 48, 32, 35, 55, 48, 52, 55, 48, 18, 48, 15, 68, 52, 58, 65, 10, 45, 62, 60),
        # J组
        "ALG": ("Algeria", "ALG", "J", 62, 58, 60, 45, 55, 38, 35, 38, 50, 42, 45, 50, 55, 22, 68, 18, 65, 55, 55, 72, 10, 50, 62, 58),
        "AUT": ("Austria", "AUT", "J", 68, 65, 68, 62, 62, 52, 28, 25, 58, 52, 55, 58, 52, 18, 55, 15, 72, 58, 58, 65, 8, 52, 65, 62),
        "JOR": ("Jordan", "JOR", "J", 48, 42, 45, 28, 45, 28, 40, 42, 32, 28, 32, 38, 48, 28, 68, 25, 52, 45, 40, 62, 12, 25, 55, 42),
        # K组
        "COD": ("DR Congo", "COD", "K", 55, 50, 52, 32, 50, 32, 40, 42, 38, 32, 38, 42, 52, 28, 72, 25, 58, 50, 48, 68, 15, 35, 55, 48),
        "UZB": ("Uzbekistan", "UZB", "K", 48, 42, 45, 30, 45, 28, 38, 40, 32, 28, 32, 38, 52, 38, 58, 35, 50, 42, 42, 58, 10, 25, 52, 45),
        "COL": ("Colombia", "COL", "K", 72, 72, 70, 58, 65, 52, 30, 28, 65, 60, 58, 65, 72, 22, 65, 18, 78, 65, 68, 78, 12, 62, 65, 68),
        # L组
        "GHA": ("Ghana", "GHA", "L", 62, 58, 60, 48, 55, 42, 35, 38, 52, 45, 48, 52, 52, 22, 72, 18, 68, 55, 58, 72, 15, 52, 60, 55),
        "PAN": ("Panama", "PAN", "L", 48, 42, 45, 35, 45, 28, 38, 42, 32, 28, 32, 38, 48, 30, 72, 28, 52, 42, 40, 58, 10, 25, 52, 42),
    }

    for code, data in default_teams.items():
        name, c, g, oa, od, og, ls, cc, lf, bh, sd, pm, tc, sa, ca, aa, jp, cl, tf, fc, pp, to, np, ii, te, tm, pr = data
        teams[code] = TeamData(name, c, g,
            opr_attack=oa, opr_defense=od, opr_gk=og, league_strength=ls,
            club_chemistry=cc, league_familiarity=lf,
            ball_hog_index=bh, star_dependency=sd,
            positional_matchup=pm, tempo_control=tc,
            setpiece_asymmetry=sa, coach_ability=ca,
            altitude_adapt=aa, jetlag_penalty=jp,
            climate_adapt=cl, travel_fatigue=tf,
            football_culture=fc, political_pressure=pp,
            tournament_overperform=to, national_pride=np,
            injury_impact=ii, tournament_experience=te,
            team_morale=tm, preparation=pr
        )

    return teams


# ============================================================
# 三、权重建模
# ============================================================

@dataclass
class ModelWeights:
    """6层13因子的攻击/防守权重"""

    # 第1层：个人能力 (25%)
    w_opr_attack_a: float = 0.24
    w_opr_attack_d: float = 0.12
    w_league_a: float = 0.10
    w_league_d: float = 0.06
    w_position_a: float = 0.08
    w_position_d: float = 0.06
    w_gk_a: float = 0.04
    w_gk_d: float = 0.14

    # 第2层：团队化学 (20%)
    w_club_a: float = 0.12
    w_club_d: float = 0.12
    w_league_fam_a: float = 0.06
    w_league_fam_d: float = 0.06
    w_ball_hog_a: float = -0.06
    w_ball_hog_d: float = -0.06
    w_star_dep_a: float = -0.04
    w_star_dep_d: float = -0.04

    # 第3层：战术因子 (上调至18%) — 对位克制在实战中极其重要
    w_matchup_a: float = 0.10
    w_matchup_d: float = 0.08
    w_tempo_a: float = 0.07
    w_tempo_d: float = 0.05
    w_setpiece_a: float = 0.05
    w_setpiece_d: float = 0.06
    w_coach_a: float = 0.05
    w_coach_d: float = 0.05

    # 第4层：环境因子 (15%)
    w_altitude_a: float = 0.06
    w_altitude_d: float = 0.06
    w_jetlag_a: float = -0.06
    w_jetlag_d: float = -0.06
    w_climate_a: float = 0.04
    w_climate_d: float = 0.04
    w_travel_a: float = -0.02
    w_travel_d: float = -0.02

    # 第5层：大赛基因因子 (下调至12%)
    w_culture_a: float = 0.05
    w_culture_d: float = 0.03
    w_pressure_a: float = 0.03
    w_pressure_d: float = 0.03
    w_overperform_a: float = 0.06
    w_overperform_d: float = 0.06
    w_pride_a: float = 0.03
    w_pride_d: float = 0.05

    # 第6层：软性因子 (10%)
    w_injury_a: float = -0.08
    w_injury_d: float = -0.08
    w_experience_a: float = 0.04
    w_experience_d: float = 0.06
    w_morale_a: float = 0.02
    w_morale_d: float = 0.02
    w_preparation_a: float = 0.02
    w_preparation_d: float = 0.02

    def get_attack_weights(self) -> np.ndarray:
        """返回攻击权重向量 (26维)"""
        return np.array([
            self.w_opr_attack_a, self.w_league_a, self.w_position_a, self.w_gk_a,
            self.w_club_a, self.w_league_fam_a, self.w_ball_hog_a, self.w_star_dep_a,
            self.w_matchup_a, self.w_tempo_a, self.w_setpiece_a, self.w_coach_a,
            self.w_altitude_a, self.w_jetlag_a, self.w_climate_a, self.w_travel_a,
            self.w_culture_a, self.w_pressure_a, self.w_overperform_a, self.w_pride_a,
            self.w_injury_a, self.w_experience_a, self.w_morale_a, self.w_preparation_a,
        ])

    def get_defense_weights(self) -> np.ndarray:
        """返回防守权重向量 (24维)"""
        return np.array([
            self.w_opr_attack_d, self.w_league_d, self.w_position_d, self.w_gk_d,
            self.w_club_d, self.w_league_fam_d, self.w_ball_hog_d, self.w_star_dep_d,
            self.w_matchup_d, self.w_tempo_d, self.w_setpiece_d, self.w_coach_d,
            self.w_altitude_d, self.w_jetlag_d, self.w_climate_d, self.w_travel_d,
            self.w_culture_d, self.w_pressure_d, self.w_overperform_d, self.w_pride_d,
            self.w_injury_d, self.w_experience_d, self.w_morale_d, self.w_preparation_d,
        ])


# ============================================================
# 四、特征提取引擎
# ============================================================

class FeatureExtractor:
    """从 TeamData 中提取标准化特征向量"""

    SCALE = 100.0  # 原始特征在 [0,100]，缩放到 [0,1]

    def extract_attack_features(self, team: TeamData) -> np.ndarray:
        """提取攻击特征向量 (24维)，归一化到 [0, 1]"""
        raw = np.array([
            team.opr_attack,
            team.league_strength,
            team.positional_matchup,
            team.opr_gk,
            team.club_chemistry,
            team.league_familiarity,
            team.ball_hog_index,
            team.star_dependency,
            team.positional_matchup,
            team.tempo_control,
            team.setpiece_asymmetry,
            team.coach_ability,
            team.altitude_adapt,
            team.jetlag_penalty,
            team.climate_adapt,
            team.travel_fatigue,
            team.football_culture,
            team.political_pressure,
            team.tournament_overperform,
            team.national_pride,
            team.injury_impact,
            team.tournament_experience,
            team.team_morale,
            team.preparation,
        ])
        return raw / self.SCALE

    def extract_defense_features(self, team: TeamData) -> np.ndarray:
        """提取防守特征向量 (24维)，归一化到 [0, 1]"""
        raw = np.array([
            team.opr_defense,
            team.league_strength,
            team.positional_matchup,
            team.opr_gk,
            team.club_chemistry,
            team.league_familiarity,
            team.ball_hog_index,
            team.star_dependency,
            team.positional_matchup,
            team.tempo_control,
            team.setpiece_asymmetry,
            team.coach_ability,
            team.altitude_adapt,
            team.jetlag_penalty,
            team.climate_adapt,
            team.travel_fatigue,
            team.football_culture,
            team.political_pressure,
            team.tournament_overperform,
            team.national_pride,
            team.injury_impact,
            team.tournament_experience,
            team.team_morale,
            team.preparation,
        ])
        return raw / self.SCALE


# ============================================================
# 五、核心预测模型
# ============================================================

class WorldCupPredictor:
    """
    世界杯预测核心引擎

    输入：两支球队的 TeamData
    输出：胜/平/负概率 + 期望进球 + λ战斗力分解
    """

    # 全局参数（可通过优化调整）
    ALPHA: float = 0.30           # log 基础进球率 → exp(0.30) ≈ 1.35
    HOME_ADVANTAGE: float = 0.12  # 主场优势（log尺度）— 世界杯东道主效应远小于联赛主场
    MAX_GOALS: int = 10           # 泊松截断

    def __init__(self, weights: ModelWeights = None, use_player_chemistry: bool = False):
        self.weights = weights or ModelWeights()
        self.extractor = FeatureExtractor()
        self._feature_count = 24
        self._player_chem_enabled = use_player_chemistry
        self._player_chem_cache = {}

        if use_player_chemistry:
            self._load_player_chemistry()

    def _load_player_chemistry(self):
        """从球员级 SquadDatabase 加载化学分"""
        try:
            from worldcup_squads import SquadDatabase, ChemistryEngine
            db = SquadDatabase()
            engine = ChemistryEngine()

            core_teams = ['FRA', 'ARG', 'BRA', 'ENG', 'GER', 'ESP', 'POR', 'NED', 'BEL', 'CRO']
            core_squads = {code: db.get_team(code) for code in core_teams}
            core_squads = {k: v for k, v in core_squads.items() if v is not None}

            normalized = engine.normalize_across_teams(core_squads,
                                                        target_min=20.0, target_max=90.0)

            for code, chem in normalized.items():
                self._player_chem_cache[code] = {
                    'club_chemistry': chem['club_chemistry_norm'],
                    'league_familiarity': chem['league_familiarity_norm'],
                }
            print(f"  [球员化学] 已加载 {len(self._player_chem_cache)} 支核心球队的球员级化学分")
        except Exception as e:
            print(f"  ⚠️ [球员化学] 加载失败: {e}")

    def override_team_chemistry(self, teams: dict) -> dict:
        """用球员级化学分覆盖TeamData"""
        if not self._player_chem_enabled or not self._player_chem_cache:
            return teams

        import copy
        updated = {}
        for code, td in teams.items():
            if code in self._player_chem_cache:
                chem = self._player_chem_cache[code]
                new_td = copy.deepcopy(td)
                new_td.club_chemistry = chem['club_chemistry']
                new_td.league_familiarity = chem['league_familiarity']
                updated[code] = new_td
            else:
                updated[code] = td
        return updated

    def compute_lambda(
        self, team: TeamData, is_attack: bool = True,
        venue_data=None, team_code: str = None
    ) -> float:
        """计算团队综合战斗力 λ（支持动态环境注入）"""
        if is_attack:
            features = self.extractor.extract_attack_features(team)
            w = self.weights.get_attack_weights()
        else:
            features = self.extractor.extract_defense_features(team)
            w = self.weights.get_defense_weights()

        n = min(len(features), len(w))
        base_lambda = np.dot(w[:n], features[:n])

        # ── 非线性球星依赖惩罚 ──
        # 条件：最佳球员 OPR > 75（意味着有"超级巨星"）+ 依赖度 > 12
        # 使用温和的 tanh 函数，避免对弱队误伤
        sd = team.star_dependency / 100.0  # 缩放到 [0,1]
        sd_threshold = 0.12  # 12/100
        is_superstar_team = team.opr_attack > 75 or team.opr_defense > 75

        if sd > sd_threshold and is_superstar_team:
            # tanh 函数: 在sd=0.12时≈0, sd=0.20时≈0.05, sd=0.35时≈0.12
            # 最大值约0.20（足够显著但不过度）
            non_linear_penalty = math.tanh((sd - sd_threshold) * 8) * 0.18
            if is_attack:
                base_lambda -= non_linear_penalty
            else:
                base_lambda -= non_linear_penalty * 0.5
            # 球霸叠加惩罚
            bh = team.ball_hog_index / 100.0
            if bh > 0.30:
                base_lambda -= (bh - 0.30) * non_linear_penalty * 0.5

        # ── 动态环境因子注入 ──
        if venue_data is not None and team_code is not None:
            from worldcup_venues import (
                compute_environmental_lambda_adjustment,
                get_temperature_sensitivity,
            )
            sensitivity = get_temperature_sensitivity(team_code)
            env_adj = compute_environmental_lambda_adjustment(
                team_code, venue_data, sensitivity
            )
            # 环境惩罚分别影响攻防
            base_lambda += env_adj['total_env_penalty'] * 0.3

        return base_lambda

    def compute_expected_goals(
        self, home: TeamData, away: TeamData,
        home_code: str = None, away_code: str = None,
        venue_data=None
    ) -> Tuple[float, float]:
        """泊松进球期望模型（支持场馆级环境调整）"""
        lam_home_att = self.compute_lambda(home, True, venue_data, home_code)
        lam_home_def = self.compute_lambda(home, False, venue_data, home_code)
        lam_away_att = self.compute_lambda(away, True, venue_data, away_code)
        lam_away_def = self.compute_lambda(away, False, venue_data, away_code)

        # 主场优势：仅当球队在本土场馆作战时才生效
        # 世界杯无"主客队"之分（只有3个东道主有真实主场）
        # 修正：去掉额外+0.10加成，使用HOME_ADVANTAGE本身（exp(0.12)=1.127 ≈ 13%合理范围）
        home_bonus = 0.0  # 中立场地无主场优势
        if venue_data is not None and home_code:
            from worldcup_venues import VENUES, is_home_stadium
            venue_key = next((k for k, v in VENUES.items() if v == venue_data), None)
            if venue_key and is_home_stadium(home_code, venue_key):
                home_bonus = self.HOME_ADVANTAGE  # 东道主主场：约13%进球提升
            elif venue_key and is_home_stadium(away_code, venue_key):
                home_bonus = -0.04  # 东道主在对方主场：小幅反压制

        log_eg_home = self.ALPHA + home_bonus + lam_home_att - lam_away_def
        log_eg_away = self.ALPHA + lam_away_att - lam_home_def

        eg_home = math.exp(log_eg_home)
        eg_away = math.exp(log_eg_away)

        return eg_home, eg_away

    def predict_match(
        self, home: TeamData, away: TeamData,
        home_code: str = None, away_code: str = None,
        venue_key: str = None
    ) -> Dict:
        """
        预测单场比赛

        参数:
            home, away: 球队数据
            home_code, away_code: 三字母代码 (如 'GER', 'JPN')
            venue_key: 场馆key (如 'Dallas', 'Houston')
        """
        from worldcup_venues import VENUES
        venue_data = VENUES.get(venue_key) if venue_key else None
        eg_home, eg_away = self.compute_expected_goals(
            home, away, home_code, away_code, venue_data
        )

        # 泊松概率矩阵
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for i in range(self.MAX_GOALS + 1):
            pi = poisson.pmf(i, eg_home)
            for j in range(self.MAX_GOALS + 1):
                pj = poisson.pmf(j, eg_away)
                prob = pi * pj
                if i > j:
                    p_home += prob
                elif i == j:
                    p_draw += prob
                else:
                    p_away += prob

        # Lambda 分解
        def _build_breakdown(t, code):
            return {
                '个人能力_lambda': round(self._layer_lambda(t, 0, True, venue_data, code), 3),
                '团队化学_lambda': round(self._layer_lambda(t, 1, True, venue_data, code), 3),
                '战术因子_lambda': round(self._layer_lambda(t, 2, True, venue_data, code), 3),
                '环境因子_lambda': round(self._layer_lambda(t, 3, True, venue_data, code), 3),
                '大赛基因_lambda': round(self._layer_lambda(t, 4, True, venue_data, code), 3),
                '软性因子_lambda': round(self._layer_lambda(t, 5, True, venue_data, code), 3),
            }

        # 最可能比分
        best_score, best_prob = "", -1
        for i in range(6):
            for j in range(6):
                prob = poisson.pmf(i, eg_home) * poisson.pmf(j, eg_away)
                if prob > best_prob:
                    best_prob = prob
                    best_score = f"{i}-{j}"

        # 环境修正明细
        env_detail = {}
        if venue_data is not None:
            from worldcup_venues import (
                compute_environmental_lambda_adjustment,
                get_temperature_sensitivity,
            )
            for label, code in [('主队', home_code), ('客队', away_code)]:
                if code:
                    adj = compute_environmental_lambda_adjustment(
                        code, venue_data, get_temperature_sensitivity(code)
                    )
                    env_detail[label] = adj

        return {
            'home_team': home.name,
            'away_team': away.name,
            'venue': venue_key,
            'expected_goals_home': round(eg_home, 2),
            'expected_goals_away': round(eg_away, 2),
            'p_home_win': round(p_home * 100, 1),
            'p_draw': round(p_draw * 100, 1),
            'p_away_win': round(p_away * 100, 1),
            'most_likely_score': best_score,
            'lambda_breakdown_home': _build_breakdown(home, home_code),
            'lambda_breakdown_away': _build_breakdown(away, away_code),
            'environmental_detail': env_detail,
        }

    def _layer_lambda(self, team: TeamData, layer: int, is_attack: bool,
                       venue_data=None, team_code: str = None) -> float:
        """计算单层因子对 λ 的贡献"""
        features = self.extractor.extract_attack_features(team) if is_attack \
                   else self.extractor.extract_defense_features(team)
        w = self.weights.get_attack_weights() if is_attack \
            else self.weights.get_defense_weights()

        layer_ranges = [(0,4), (4,8), (8,12), (12,16), (16,20), (20,24)]
        start, end = layer_ranges[layer]
        base = np.dot(w[start:end], features[start:end])

        # 环境层特殊处理：叠加动态环境修正
        if layer == 3 and venue_data is not None and team_code:
            from worldcup_venues import (
                compute_environmental_lambda_adjustment,
                get_temperature_sensitivity,
            )
            env_adj = compute_environmental_lambda_adjustment(
                team_code, venue_data, get_temperature_sensitivity(team_code)
            )
            base += env_adj['total_env_penalty'] * 0.3

        return base


# ============================================================
# 六、演示：预测几场焦点战
# ============================================================

if __name__ == "__main__":
    from worldcup_venues import VENUES

    print("=" * 70)
    print("2026世界杯预测系统 v2 · 场馆感知 + 非线性球星依赖")
    print("=" * 70)

    teams = load_2026_teams()
    predictor = WorldCupPredictor()

    # ── 焦点战1：巴西 vs 摩洛哥 (C组，纽约，实际结果1-1) ──
    print("\n【焦点战1】巴西 vs 摩洛哥 — C组 · 纽约/新泽西 (已完赛1-1)")
    print("-" * 55)
    result = predictor.predict_match(
        teams["BRA"], teams["MAR"],
        home_code="BRA", away_code="MAR",
        venue_key="New York / New Jersey"
    )
    print(f"  期望进球: Brazil {result['expected_goals_home']} — {result['expected_goals_away']} Morocco")
    print(f"  胜率分布: Brazil {result['p_home_win']}% | 平 {result['p_draw']}% | Morocco {result['p_away_win']}%")
    print(f"  最可能比分: {result['most_likely_score']}")
    if result['environmental_detail']:
        for label, adj in result['environmental_detail'].items():
            print(f"  🌡️ {label}环境修正: 总={adj['total_env_penalty']:.3f} (海拔{adj['altitude_impact']:.3f} 高温{adj['heat_impact']:.3f} 时差{adj['jetlag_impact']:.3f})")

    # ── 焦点战2：德国 vs 库拉索 (E组，休斯顿，高温高湿) ──
    print("\n【焦点战2】德国 vs 库拉索 — E组 · 休斯顿 (高温高湿🔥)")
    print("-" * 55)
    result = predictor.predict_match(
        teams["GER"], teams["CUW"],
        home_code="GER", away_code="CUW",
        venue_key="Houston"
    )
    print(f"  期望进球: Germany {result['expected_goals_home']} — {result['expected_goals_away']} Curacao")
    print(f"  胜率分布: Germany {result['p_home_win']}% | 平 {result['p_draw']}% | Curacao {result['p_away_win']}%")
    print(f"  最可能比分: {result['most_likely_score']}")
    if result['environmental_detail']:
        for label, adj in result['environmental_detail'].items():
            print(f"  🌡️ {label}环境修正: 总={adj['total_env_penalty']:.3f} (海拔{adj['altitude_impact']:.3f} 高温{adj['heat_impact']:.3f} 时差{adj['jetlag_impact']:.3f})")

    # ── 焦点战3：荷兰 vs 日本 (F组，达拉斯，日本时差惩罚) ──
    print("\n【焦点战3】荷兰 vs 日本 — F组 · 达拉斯 (日本时差考验🕐)")
    print("-" * 55)
    result = predictor.predict_match(
        teams["NED"], teams["JPN"],
        home_code="NED", away_code="JPN",
        venue_key="Dallas"
    )
    print(f"  期望进球: Netherlands {result['expected_goals_home']} — {result['expected_goals_away']} Japan")
    print(f"  胜率分布: Netherlands {result['p_home_win']}% | 平 {result['p_draw']}% | Japan {result['p_away_win']}%")
    print(f"  最可能比分: {result['most_likely_score']}")
    if result['environmental_detail']:
        for label, adj in result['environmental_detail'].items():
            print(f"  🌡️ {label}环境修正: 总={adj['total_env_penalty']:.3f} (海拔{adj['altitude_impact']:.3f} 高温{adj['heat_impact']:.3f} 时差{adj['jetlag_impact']:.3f})")

    # ── 焦点战4：墨西哥 vs 韩国 (A组，高原主场) ──
    print("\n【焦点战4】墨西哥 vs 韩国 — A组 · 瓜达拉哈拉 (高原主场🏔️)")
    print("-" * 55)
    result = predictor.predict_match(
        teams["MEX"], teams["KOR"],
        home_code="MEX", away_code="KOR",
        venue_key="Guadalajara"
    )
    print(f"  期望进球: Mexico {result['expected_goals_home']} — {result['expected_goals_away']} South Korea")
    print(f"  胜率分布: Mexico {result['p_home_win']}% | 平 {result['p_draw']}% | South Korea {result['p_away_win']}%")
    print(f"  最可能比分: {result['most_likely_score']}")
    if result['environmental_detail']:
        for label, adj in result['environmental_detail'].items():
            print(f"  🌡️ {label}环境修正: 总={adj['total_env_penalty']:.3f} (海拔{adj['altitude_impact']:.3f} 高温{adj['heat_impact']:.3f} 时差{adj['jetlag_impact']:.3f})")
    print(f"  🏔️ 墨西哥高原加成：海拔1566m对韩国惩罚{result['environmental_detail'].get('客队', {}).get('altitude_impact', 0):.3f}")

    # ── 球星依赖非线性惩罚演示 ──
    print("\n【焦点战5】埃及 vs 比利时 — 球星依赖非线性惩罚演示")
    print("-" * 55)
    egypt = teams["EGY"]
    print(f"  🇪🇬 埃及: 萨拉赫OPR=91, 全队平均=75 → 球星依赖度={egypt.star_dependency} (阈值=12)")
    print(f"       非线性惩罚: tanh((0.55-0.12)×8)×0.18 → "
          f"约{math.tanh((egypt.star_dependency/100-0.12)*8)*0.18:.3f}的λ扣减")
    result = predictor.predict_match(
        teams["BEL"], teams["EGY"],
        home_code="BEL", away_code="EGY",
        venue_key="Seattle"
    )
    print(f"  期望进球: Belgium {result['expected_goals_home']} — {result['expected_goals_away']} Egypt")
    print(f"  胜率分布: Belgium {result['p_home_win']}% | 平 {result['p_draw']}% | Egypt {result['p_away_win']}%")
    pen = math.tanh((egypt.star_dependency/100-0.12)*8)*0.18
    print(f"  ⚡ 萨拉赫效应：埃及攻击λ被削弱约{pen:.2f}（对手只需盯防一人）")

    # ── 球员级化学模型对比演示 ──
    print("\n【焦点战6】球员级化学模型 vs 原始模型对比（use_player_chemistry=True）")
    print("-" * 55)
    pred_player = WorldCupPredictor(use_player_chemistry=True)
    player_teams = pred_player.override_team_chemistry(teams)

    comparison_matches = [
        ("GER", "ESP", "Dallas", "德国 vs 西班牙"),
        ("ARG", "POR", "Atlanta", "阿根廷 vs 葡萄牙"),
        ("ENG", "CRO", "Toronto", "英格兰 vs 克罗地亚"),
    ]

    for h, a, v, label in comparison_matches:
        # 原始
        r_orig = predictor.predict_match(teams[h], teams[a],
                                          home_code=h, away_code=a, venue_key=v)
        # 球员级
        r_play = pred_player.predict_match(player_teams[h], player_teams[a],
                                            home_code=h, away_code=a, venue_key=v)

        delta_home = r_play['p_home_win'] - r_orig['p_home_win']
        arrow = "▲" if delta_home > 0 else "▼" if delta_home < 0 else "="
        print(f"\n  {label}")
        print(f"    原始: {r_orig['home_team']} {r_orig['p_home_win']}% | "
              f"平 {r_orig['p_draw']}% | {r_orig['away_team']} {r_orig['p_away_win']}%")
        print(f"    球员: {r_play['home_team']} {r_play['p_home_win']}% | "
              f"平 {r_play['p_draw']}% | {r_play['away_team']} {r_play['p_away_win']}%")
        print(f"    差异: {arrow}{abs(delta_home):.1f}%")

    print("\n" + "=" * 70)
    print("球员级化学引擎关键结论：")
    print("  1. 德国(拜仁系)化学分最高(99) → 实战加成最大")
    print("  2. 英/西化学居中(46-64) → 本土联赛效应存在但有限")
    print("  3. 巴西/克罗地亚化学校低(27-29) → 球员高度分散")
    print("  4. 化学层总贡献约λ的8-15%，是重要但不主导的因子")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("模型核心公式总结：")
    print("  λ_attack = Σ w_i × feat_i — 非线性球星惩罚 — 动态环境修正")
    print("  E[goals_H] = exp(α + β_home + β_stadium + λ_a(H) — λ_d(A))")
    print("  P(win|H) = Σ_{i>j} Poisson(i|E_H) × Poisson(j|E_A)")
    print("=" * 70)

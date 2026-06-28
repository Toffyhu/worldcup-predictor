"""
2026世界杯场馆数据库
===================
16座场馆的地理/气候完整数据 + 环境因子计算引擎

数据来源: TheDatabetics Stadium Guide + FIFA官方气候报告
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math

@dataclass
class VenueData:
    """单座场馆的完整环境数据"""
    name: str                    # 场馆名称
    city: str                    # 所在城市
    country: str                 # 所在国家
    altitude_m: float            # 海拔（米）
    avg_temp_june: float         # 6-7月均温（摄氏度）
    avg_humidity: float          # 平均湿度（%）
    climate_type: str            # 气候类型
    timezone_utc: int            # UTC时区偏移
    esi: float                   # 环境压力指数 (0-10)
    latitude: float              # 纬度
    longitude: float             # 经度
    has_roof: bool = False       # 是否有可闭合顶棚

# ═══════════════════════════════════════════════════════════════
# 16座场馆完整数据
# ═══════════════════════════════════════════════════════════════

VENUES: Dict[str, VenueData] = {
    "Mexico City": VenueData(
        name="Estadio Azteca", city="Mexico City", country="Mexico",
        altitude_m=2240, avg_temp_june=20, avg_humidity=55,
        climate_type="Subtropical Highland", timezone_utc=-6,
        esi=7.5, latitude=19.30, longitude=-99.15
    ),
    "Guadalajara": VenueData(
        name="Estadio Akron", city="Guadalajara", country="Mexico",
        altitude_m=1566, avg_temp_june=24, avg_humidity=60,
        climate_type="Humid Subtropical", timezone_utc=-6,
        esi=6.6, latitude=20.68, longitude=-103.46
    ),
    "Monterrey": VenueData(
        name="Estadio BBVA", city="Monterrey", country="Mexico",
        altitude_m=540, avg_temp_june=28, avg_humidity=68,
        climate_type="Semi-arid", timezone_utc=-6,
        esi=7.0, latitude=25.67, longitude=-100.27
    ),
    "Miami": VenueData(
        name="Hard Rock Stadium", city="Miami", country="USA",
        altitude_m=2, avg_temp_june=29, avg_humidity=75,
        climate_type="Tropical Monsoon", timezone_utc=-4,
        esi=9.1, latitude=25.96, longitude=-80.24, has_roof=True
    ),
    "Houston": VenueData(
        name="NRG Stadium", city="Houston", country="USA",
        altitude_m=13, avg_temp_june=28, avg_humidity=75,
        climate_type="Humid Subtropical", timezone_utc=-5,
        esi=8.7, latitude=29.68, longitude=-95.41, has_roof=True
    ),
    "Dallas": VenueData(
        name="AT&T Stadium", city="Dallas", country="USA",
        altitude_m=184, avg_temp_june=30, avg_humidity=65,
        climate_type="Humid Subtropical", timezone_utc=-5,
        esi=8.0, latitude=32.75, longitude=-97.09, has_roof=True
    ),
    "Atlanta": VenueData(
        name="Mercedes-Benz Stadium", city="Atlanta", country="USA",
        altitude_m=320, avg_temp_june=27, avg_humidity=70,
        climate_type="Humid Subtropical", timezone_utc=-4,
        esi=7.2, latitude=33.76, longitude=-84.40, has_roof=True
    ),
    "New York / New Jersey": VenueData(
        name="MetLife Stadium", city="New Jersey", country="USA",
        altitude_m=7, avg_temp_june=25, avg_humidity=65,
        climate_type="Humid Continental", timezone_utc=-4,
        esi=5.8, latitude=40.81, longitude=-74.07
    ),
    "Boston": VenueData(
        name="Gillette Stadium", city="Boston", country="USA",
        altitude_m=43, avg_temp_june=22, avg_humidity=70,
        climate_type="Humid Continental", timezone_utc=-4,
        esi=5.6, latitude=42.09, longitude=-71.26
    ),
    "Philadelphia": VenueData(
        name="Lincoln Financial Field", city="Philadelphia", country="USA",
        altitude_m=12, avg_temp_june=25, avg_humidity=68,
        climate_type="Humid Continental", timezone_utc=-4,
        esi=6.0, latitude=39.90, longitude=-75.17
    ),
    "Kansas City": VenueData(
        name="Arrowhead Stadium", city="Kansas City", country="USA",
        altitude_m=264, avg_temp_june=26, avg_humidity=70,
        climate_type="Humid Continental", timezone_utc=-5,
        esi=6.2, latitude=39.05, longitude=-94.48
    ),
    "Los Angeles": VenueData(
        name="SoFi Stadium", city="Los Angeles", country="USA",
        altitude_m=30, avg_temp_june=22, avg_humidity=65,
        climate_type="Mediterranean", timezone_utc=-7,
        esi=4.5, latitude=33.95, longitude=-118.34, has_roof=True
    ),
    "San Francisco Bay Area": VenueData(
        name="Levi's Stadium", city="San Francisco", country="USA",
        altitude_m=15, avg_temp_june=20, avg_humidity=65,
        climate_type="Mediterranean", timezone_utc=-7,
        esi=3.8, latitude=37.40, longitude=-121.97
    ),
    "Seattle": VenueData(
        name="Lumen Field", city="Seattle", country="USA",
        altitude_m=52, avg_temp_june=16, avg_humidity=70,
        climate_type="Marine West Coast", timezone_utc=-7,
        esi=3.4, latitude=47.60, longitude=-122.33
    ),
    "Vancouver": VenueData(
        name="BC Place", city="Vancouver", country="Canada",
        altitude_m=70, avg_temp_june=16, avg_humidity=75,
        climate_type="Marine West Coast", timezone_utc=-7,
        esi=3.2, latitude=49.28, longitude=-123.11, has_roof=True
    ),
    "Toronto": VenueData(
        name="BMO Field", city="Toronto", country="Canada",
        altitude_m=76, avg_temp_june=22, avg_humidity=70,
        climate_type="Humid Continental", timezone_utc=-4,
        esi=5.2, latitude=43.63, longitude=-79.42
    ),
}

# ═══════════════════════════════════════════════════════════════
# 球队时区偏移（用于计算时差惩罚）
# ═══════════════════════════════════════════════════════════════

TEAM_TIMEZONES: Dict[str, int] = {
    # 欧洲 (UTC+0~+3)
    "ENG": 1, "FRA": 2, "GER": 2, "ESP": 2, "ITA": 2, "NED": 2, "POR": 1,
    "BEL": 2, "CRO": 2, "SUI": 2, "SWE": 2, "NOR": 2, "AUT": 2, "CZE": 2,
    "BIH": 2, "TUR": 3, "SCO": 1,
    # 南美 (UTC-3~-5)
    "ARG": -3, "BRA": -3, "URU": -3, "COL": -5, "ECU": -5, "PAR": -4,
    # 北美
    "USA": -4, "MEX": -6, "CAN": -4,
    # 非洲 (UTC+0~+3)
    "MAR": 1, "SEN": 0, "EGY": 3, "TUN": 1, "CIV": 0, "ALG": 1,
    "GHA": 0, "RSA": 2, "COD": 1, "CPV": -1, "NGA": 1,
    # 亚洲/大洋洲
    "JPN": 9, "KOR": 9, "AUS": 10, "IRN": 3, "KSA": 3, "QAT": 3,
    "UZB": 5, "JOR": 3, "IRQ": 3, "NZL": 12,
    # 其他
    "HAI": -4, "CUW": -4, "PAN": -5,
}


# ═══════════════════════════════════════════════════════════════
# 环境压力 → 球队λ修正系数
# ═══════════════════════════════════════════════════════════════

def compute_environmental_lambda_adjustment(
    team_code: str,
    venue: VenueData,
    base_temp_sensitivity: float = 0.5  # 该队对温度的敏感度 (0=不怕热, 1=极怕热)
) -> Dict[str, float]:
    """
    计算一场比赛的环境因子修正量。

    返回 {
        'altitude_impact': float,   # 海拔对λ的修正 (负值=削弱)
        'heat_impact': float,       # 高温对λ的修正
        'jetlag_impact': float,     # 时差对λ的修正
        'total_env_penalty': float, # 总环境惩罚
    }

    所有值在 [-1, +1] 范围，用于直接加到λ上。
    """

    # 1. 海拔影响
    # 海拔每升高1000m，海平面球队VO2max下降约8%
    # 但本国高原球队免疫
    team_tz = TEAM_TIMEZONES.get(team_code, 0)
    is_high_altitude_native = team_code in ["MEX", "ECU", "COL", "BOL"]  # 高原国家
    if is_high_altitude_native:
        altitude_penalty = 0.0
    else:
        altitude_penalty = -0.06 * (venue.altitude_m / 1000)  # 每1000m -6%（杯赛有高原适应训练）

    # 2. 高温影响
    # 超过28°C开始有影响，到35°C影响最大
    if venue.avg_temp_june > 28:
        temp_excess = venue.avg_temp_june - 28
        heat_penalty = -0.015 * temp_excess * base_temp_sensitivity
    else:
        heat_penalty = 0.0

    # 湿度放大效应 (>70%时每1%额外0.2%惩罚)
    if venue.avg_humidity > 70:
        humidity_bonus = (venue.avg_humidity - 70) * 0.002
        heat_penalty *= (1.0 + humidity_bonus)

    # 3. 时差影响（杯赛版本：温和处理，球队预期并准备了旅行）
    time_diff = abs(venue.timezone_utc - team_tz)
    if time_diff <= 3:
        jetlag_penalty = 0.0
    elif time_diff <= 6:
        jetlag_penalty = -0.010 * (time_diff - 3)
    elif time_diff <= 9:
        jetlag_penalty = -0.017 * (time_diff - 6) + (-0.030)
    else:
        jetlag_penalty = -0.023 * (time_diff - 9) + (-0.081)

    total = altitude_penalty + heat_penalty + jetlag_penalty

    return {
        'altitude_impact': round(altitude_penalty, 4),
        'heat_impact': round(heat_penalty, 4),
        'jetlag_impact': round(jetlag_penalty, 4),
        'total_env_penalty': round(total, 4),
    }


def get_temperature_sensitivity(team_code: str) -> float:
    """根据球队所处气候带，返回温度敏感度（0-1，越高越怕热）"""
    # 北欧/加拿大球队对热敏感
    high_sensitivity = {"ENG", "SCO", "SWE", "NOR", "DEN", "CAN", "NED", "BEL", "GER", "SUI"}
    # 热带/亚热带球队适应
    low_sensitivity = {"BRA", "MAR", "SEN", "CIV", "GHA", "NGA", "EGY", "TUN", "ALG",
                       "MEX", "USA", "COL", "ECU", "PAR", "URU", "KSA", "QAT", "IRQ",
                       "JOR", "PAN", "HAI", "CUW", "RSA", "COD", "CPV"}
    if team_code in high_sensitivity:
        return 1.0
    elif team_code in low_sensitivity:
        return 0.3
    else:
        return 0.6  # 默认中等敏感


# ═══════════════════════════════════════════════════════════════
# 主场效应识别
# ═══════════════════════════════════════════════════════════════

def is_home_stadium(team_code: str, venue_key: str) -> bool:
    """判断球队是否在本土作战"""
    home_map = {
        "USA": ["Los Angeles", "Dallas", "Houston", "Atlanta", "Miami",
                "New York / New Jersey", "Boston", "Philadelphia",
                "Kansas City", "Seattle", "San Francisco Bay Area"],
        "MEX": ["Mexico City", "Guadalajara", "Monterrey"],
        "CAN": ["Toronto", "Vancouver"],
    }
    return venue_key in home_map.get(team_code, [])

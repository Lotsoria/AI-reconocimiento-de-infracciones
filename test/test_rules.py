# tests/test_rules.py
from core.rules.speed import SpeedRule

def test_speedrule_init():
    cfg = {"geometry":{"speed_lines":{"A":[[0,0],[10,0]],"B":[[0,10],[10,10]]}},
           "speed":{"pixel_distance":10,"k_calibration":0.1,"limit_kmh":10}}
    rule = SpeedRule(cfg)
    assert rule.limit == 10
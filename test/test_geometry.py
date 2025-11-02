# tests/test_geometry.py
from core.utils.geometry import crossed_line

def test_crossed_line_false_without_prev():
    t = {"bbox":[0,0,10,10]}  # sin prev_center -> False
    assert crossed_line(t, [0,5], [10,5]) == False

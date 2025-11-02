# core/types.py
# Tipos simples (pudimos usar dataclasses, pero dict funciona bien)
from typing import List, Dict, Any, Tuple

Det = Dict[str, Any]       # {"bbox":[x1,y1,x2,y2], "conf":float, "label":str}
Track = Dict[str, Any]     # {"id":int, "bbox":[...], "label":str, "history":[(x,y)]}
Event = Dict[str, Any]     # {"type":str, "ts":float, "track_id":int, "extra":dict}

from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortWrapper:
    def __init__(self, max_age=15):
        self.trk = DeepSort(max_age=max_age)

    def update(self, dets, frame):
        """
        dets: [{"bbox":[x1,y1,x2,y2], "conf":..., "label": str}, ...]
        retorna tracks: [{"id":int,"bbox":[...],"label":str,"prev_center":(x,y)}]
        """
        bbs = []
        for d in dets:
            x1,y1,x2,y2 = d["bbox"]
            w,h = x2-x1, y2-y1
            bbs.append(([x1,y1,w,h], d["conf"], d["label"]))
        tracks = self.trk.update_tracks(bbs, frame=frame)

        out = []
        for t in tracks:
            if not t.is_confirmed(): continue
            l,t_,r,b = t.to_ltrb()
            out.append({"id": t.track_id, "bbox":[l,t_,r,b], "label": t.get_det_class(), "prev_center": getattr(t, "_prev_center", None)})
            # Guarda centro previo para cruce de líneas
            c = ((l+r)/2.0, (t_+b)/2.0)
            t._prev_center = c
            out[-1]["prev_center"] = getattr(t, "_prev_center", None)
        return out
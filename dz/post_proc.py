from typing import Dict

import cv2
import numpy as np

from zone import SataticZoneSide, SataticZoneButtom
from frame_meta import FrameMeta


def draw_results(frame_meta: FrameMeta, zone_side: dict, zone_buttom: dict) -> np.ndarray:
    img = frame_meta.frame
    side_zone_status = False
    buttom_zone_status = False
    if not frame_meta.is_empty():
        img = img.copy()
        for detection in frame_meta.detections:
            yolo_score = str(np.round(detection.confidence, 2))
            p1, p2 = detection.crop.get_bbox()
            color = [255, 0, 0]
            label_id = "{}".format(yolo_score)
            t_size = cv2.getTextSize(label_id, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (p1.x, p1.y), (p2.x, p2.y), color, 1)
            cv2.putText(img, label_id, (p1.x, p1.y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            if not side_zone_status:
                side_zone_status = detection.side_zone_status
            if not buttom_zone_status:
                buttom_zone_status = detection.buttom_zone_status
        img = zone_side[frame_meta.zone_type].draw_zone(img, side_zone_status)
        img = zone_buttom[frame_meta.zone_type].draw_zone(img, buttom_zone_status)
    return img

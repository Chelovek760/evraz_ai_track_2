import cv2
import numpy as np

from zone import StaticZone
from frame_meta import FrameMeta


def draw_results(frame_meta: FrameMeta, zone: StaticZone = None) -> np.ndarray:
    img = frame_meta.frame
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
        # zone_frame_status = [alarm for alarm in frame_meta.alarms if alarm.type == "zone"] != []
        # img = zone.draw_zone(img, zone_frame_status)
    return img

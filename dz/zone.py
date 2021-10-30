import copy
from typing import List, Dict

import cv2
import numpy as np
from shapely.geometry import box
from shapely.geometry import Polygon

from frame_meta import FrameMeta


class StaticZoneBase:
    def __init__(self, zone: List, area_percent: float = 0.1):
        self.zone = Polygon(zone)
        self.area_percent = area_percent
        if zone is None:
            self.coords = []
        else:
            self.coords = [np.array(self.zone.exterior.coords).round().astype(np.int32)]

    def __call__(self, *args, **kwargs):
        pass

    def draw_zone(self, img, fill: bool = False, alpha: float = 0.75) -> np.ndarray:
        if fill:
            overlay = img.copy()
            cv2.fillPoly(img, self.coords, color=(255, 0, 0), lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            cv2.drawContours(img, self.coords, -1, color=(255, 0, 0))
        return img

    def zone_intersect(self, frame_meta: FrameMeta) -> FrameMeta:
        pass


class SataticZoneButtom(StaticZoneBase):
    def __call__(self, bbox: List) -> bool:
        """
        description: Check intersection with bbox
        param:
            intersect_status:
        return:
            intersect_status: bool
        """

        x1, y1, x2, y2 = bbox
        y1 += int((y2 - y1) * (1 - self.area_percent))  # Only bottom check
        box_obj = box(x1, y1, x2, y2)
        status = self.zone.intersects(box_obj)
        return status

    def zone_intersect(self, frame_meta: FrameMeta) -> FrameMeta:
        for detection in frame_meta.detections:
            (x1, y1), (x2, y2) = detection.get_bbox_xyxy()
            detection.buttom_zone_status = self([x1, y1, x2, y2])
        return frame_meta


class SataticZoneSide(StaticZoneBase):
    def __call__(self, bbox: List) -> bool:
        """
        description: Check intersection with bbox
        param:
            intersect_status:
        return:
            intersect_status: bool
        """
        x1, y1, x2, y2 = copy.deepcopy(bbox)
        y1 += int((y2 - y1) * (1 - self.area_percent))
        x1 += int((x2 - x1) * (1 - self.area_percent))
        box_obj = box(x1, y1, x2, y2)
        status_right = self.zone.intersects(box_obj)
        x1, y1, x2, y2 = copy.deepcopy(bbox)
        x2 = x1 + int((x2 - x1) * (self.area_percent))
        y1 += int((y2 - y1) * (1 - self.area_percent))
        box_obj = box(x1, y1, x2, y2)
        status_left = self.zone.intersects(box_obj)
        return status_left or status_right

    def zone_intersect(self, frame_meta: FrameMeta) -> FrameMeta:
        for detection in frame_meta.detections:
            (x1, y1), (x2, y2) = detection.get_bbox_xyxy()
            detection.side_zone_status = self([x1, y1, x2, y2])
        return frame_meta

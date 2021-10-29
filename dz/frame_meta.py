from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import numpy as np

from g_utils import xyxy_to_xywh
from config import ModelConfig


@dataclass
class FrameHistory:
    frame_ix: int
    frame_ix_global: int
    helmet_class: int
    helmet_proba: float
    zone_status: bool


@dataclass
class Alarm:
    identity: int
    type: str
    history: Optional[List[FrameHistory]]
    send: Optional[bool]


@dataclass
class Point:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))


BBoxXY = Tuple[Point, Point]
BBoxWH = Tuple[Point, int, int]


@dataclass
class DetectionCrop:
    x_1: int
    y_1: int
    x_2: int
    y_2: int

    detection: "Detection"
    frame_meta: "FrameMeta"

    def get_img(self) -> np.ndarray:
        return self.frame_meta.frame[self.y_1 : self.y_2, self.x_1 : self.x_2, :]

    def get_bbox(self) -> BBoxXY:
        return Point(self.x_1, self.y_1), Point(self.x_2, self.y_2)


@dataclass
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    cls: int

    frame_meta: "FrameMeta"
    model_config: ModelConfig
    crop: Optional["DetectionCrop"] = None

    zone_status: bool = None
    identity: int = -1

    def get_bbox(self) -> tuple:
        return self.x, self.y, self.width, self.height

    def get_bbox_xyxy(self) -> BBoxXY:
        x_1 = int(min(max(self.x - self.width / 2, 1), self.frame_meta.width - 1))
        x_2 = int(min(max(x_1 + self.width, 1), self.frame_meta.width - 1))
        y_1 = int(min(max(self.y - self.height / 2, 1), self.frame_meta.height - 1))
        y_2 = int(min(max(y_1 + self.height, 1), self.frame_meta.height - 1))
        return Point(x_1, y_1), Point(x_2, y_2)

    @property
    def min_side_size(self) -> int:
        return min(self.width, self.height)

    def make_crop(self) -> DetectionCrop:
        p1, p2 = self.get_bbox_xyxy()
        self.crop = DetectionCrop(
            x_1=p1.x,
            x_2=p2.x,
            y_1=p1.y,
            y_2=p2.y,
            detection=self,
            frame_meta=self.frame_meta,
        )
        return self.crop


class FrameMeta:
    detections: List[Detection]
    alarms: List[Alarm]
    frame_ix: int
    frame: np.ndarray
    width: int
    height: int

    def __init__(self, model_config: ModelConfig, frame: np.ndarray, frame_ix: int, raw_detections, file_name: str):
        self.detections = []
        self.alarms = []
        self.frame_ix = frame_ix
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        self.file_name = file_name

        for *xyxy, conf, cls in reversed(raw_detections):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            self.detections.append(
                Detection(
                    x=x_c,
                    y=y_c,
                    width=bbox_w,
                    height=bbox_h,
                    confidence=conf.item(),
                    cls=cls.item(),
                    frame_meta=self,
                    model_config=model_config,
                    zone_status=True,
                )
            )

    def is_empty(self) -> bool:
        return len(self.detections) == 0

    def to_bbox_tensor(self) -> torch.Tensor:
        return torch.Tensor([detection.get_bbox() for detection in self.detections])

    def to_conf_tensor(self) -> torch.Tensor:
        return torch.Tensor([[detection.confidence] for detection in self.detections])

    def set_identities(self, identities):
        for identity, detection in zip(identities, self.detections):
            detection.identity = int(identity)

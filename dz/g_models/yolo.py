import random
import threading
from typing import List

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, check_img_size, scale_coords


class YoloTransform(nn.Module):
    def __init__(self, size, pad, tensor_type=torch.half):
        super().__init__()
        self.transforms = nn.Sequential(
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.Pad(pad, padding_mode="constant", fill=0),
            T.ConvertImageDtype(tensor_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            return x


def get_new_shape_padding(model_frame_size, height, width):
    """
    description: calculate new shape and padding for letterbox

    param:
        model_frame_size: str - model size
        height: int - height of current img
        width: int -  width of current img
    return:
        new shape and padding: (new_height, new_width), (ty1, ty2, tx1, tx2)
    """
    r_w = model_frame_size / width
    r_h = model_frame_size / height
    if r_h > r_w:
        tw = model_frame_size
        th = int(r_w * height)
        tx1 = tx2 = 0
        ty1 = int((model_frame_size - th) / 2)
        ty2 = model_frame_size - th - ty1
    else:
        tw = int(r_h * width)
        th = model_frame_size
        tx1 = int((model_frame_size - tw) / 2)
        tx2 = model_frame_size - tw - tx1
        ty1 = ty2 = 0

    return (th, tw), (tx1, ty1, tx2, ty2)


class Yolo:
    metric_model_name = "yolo"

    def __init__(
        self,
        weight: List,
        model_frame_size: int,
        device: torch.device,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.4,
    ):
        self.device = device
        self.model_frame_size = model_frame_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = attempt_load(list(weight), map_location=device).eval().half()
        self.model.zero_grad(set_to_none=True)

        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(model_frame_size)

        self.camera_transforms = {}

    def __call__(self, frames: torch.Tensor) -> List:
        origin_shape = frames.shape[2:4]
        if origin_shape not in self.camera_transforms:
            height, width = origin_shape
            new_shape, padding = get_new_shape_padding(self.model_frame_size, height, width)

            if origin_shape not in self.camera_transforms:
                self.camera_transforms[origin_shape] = torch.jit.script(
                    YoloTransform(new_shape, padding).to(self.device)
                )

        with torch.no_grad():
            preprocessed_frames = self.camera_transforms[origin_shape](frames)
            preproccesed_shape = preprocessed_frames.shape[2:]
            predictions = self.model(preprocessed_frames, augment=False)[0]

            return self.post_processing(predictions, preproccesed_shape, origin_shape)

    def post_processing(self, prediction: torch.Tensor, preproccesed_shape: List, origin_shape: List) -> List:
        pred = non_max_suppression(
            prediction,
            self.confidence_threshold,
            self.iou_threshold,
            classes=[0],
            agnostic=True,
        )
        result = []
        for part in pred:
            part[:, :4] = scale_coords(preproccesed_shape, part[:, :4], origin_shape).round()
            result.append(part.cpu())
        return result  # [[x, y, x, y, conf, class], ... ]


def plot_one_box(bboxes, img0, labels, color=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:       a box likes [x1,y1,x2,y2]
        img:     a opencv image object
        labels:  List
        color:   color to draw rectangle, such as (0,255,0)
        line_thickness: int
    return:
        a opencv image object with bbox and label
    """
    if labels is None:
        labels = ["worker"]
    tl = line_thickness or round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    img = img0.copy()
    for bbox in bboxes:
        tf = max(tl - 1, 1)  # font thickness
        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        label = labels[int(bbox[4])]
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img

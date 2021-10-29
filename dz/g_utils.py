import json
from dataclasses import dataclass, asdict
import datetime
from typing import List, Sequence, Dict


@dataclass
class ImageCoco:
    id: int
    file_name: str
    width: int = 1920
    height: int = 1080
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: int = 0


@dataclass
class AnnotationCoco:
    id: int
    image_id: int
    area: float
    bbox: List
    category_id: int = 1
    segmentation = []
    iscrowd: int = 0
    atributes = {"occluded": "false"}


class CocoPresent:
    def __init__(self, save_path: str):

        self.INFO = {
            "description": "EVRAZ TASK2",
            "url": "",
            "version": "1",
            "year": 2021,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(" "),
        }
        self.LICENSES = [{"name": "", "id": 0, "url": ""}]

        self.CATEGORIES = [
            {
                "id": 1,
                "name": "person",
                "supercategory": "",
            }
        ]
        self.save_path = save_path
        self.annotations = []
        self.c_id_img = 1
        self.c_id_bbox = 1
        self.postfix = {}
        self.imgs_index = {}
        self.images = []

    def add_an(self, file_name, bbox):
        if file_name not in self.imgs_index:
            self.imgs_index[file_name] = self.c_id_img
            self.images.append(ImageCoco(id=self.c_id_img, file_name=file_name))
            self.c_id_img += 1
        # TODO area
        self.annotations.append(
            AnnotationCoco(id=self.c_id_bbox, image_id=self.imgs_index[file_name], area=0, bbox=bbox)
        )
        self.c_id_bbox += 1

    def create_json(self):
        for im in self.annotations:
            print(asdict(im)["bbox"])
        data = {
            "licenses": self.LICENSES,
            "info": self.INFO,
            "categories": self.CATEGORIES,
            "images": [asdict(im) for im in self.images],
            "annotations": [asdict(an) for an in self.annotations],
        }
        with open(self.save_path, "w") as outfile:
            json.dump(data, outfile)


def xyxy_to_xywh(*xyxy):
    """ " Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def chunks_generator(lst: Sequence, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

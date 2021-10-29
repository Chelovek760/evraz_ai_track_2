import datetime
from typing import List, Sequence


class CocoPresent:
    def __init__(self, save_path: str):

        self.INFO = {
        "description": "Example Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2018,
        "contributor": "waspinator",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
                                }
        self.LICENSES = [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ]

        self.CATEGORIES = [
            {
                'id': 1,
                'name': 'square',
                'supercategory': 'shape',
            }]
        self.save_path = save_path
        self.images_name = []
        self.bboxes = []
        self.ids = []
        self.c_id = 0
        self.postfix = {}

    def add_an(self, file_name, bbox):
        self.images_name.append(file_name)
        self.ids.append(self.c_id)
        self.c_id += 1
        self.bboxes.append(bbox)

    def create_json(self):
        pass


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

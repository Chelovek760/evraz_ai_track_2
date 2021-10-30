import datetime
import glob
import json
import multiprocessing
import time
import logging
from pathlib import Path
from typing import List

import cv2
import torch
import hydra

from post_proc import draw_results
from zone import SataticZoneButtom, SataticZoneSide
from config import GaleatiConfig
from g_models.yolo import Yolo
from g_utils import chunks_generator, CocoPresent
from frame_meta import FrameMeta
import numpy as np

logger = logging.getLogger(__name__)


class App:
    def __init__(self, config: GaleatiConfig):
        self.config = config
        self.model_config = config.model
        self.json_dest = self.config.json_dest + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")) + ".json"
        self.coco_maker = CocoPresent("data/submission_example.json", self.json_dest)
        self.coco_maker.gen_files_id()
        self.source = self.config.source + "*.jpg"
        self.dest = Path(self.config.dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda")
        with open(self.config.zones_meta, "r") as f:
            self.zones_types = json.load(f)
        with open(self.config.zones_poly, "r") as f:
            self.zones_poly = json.load(f)
        self.zones_side = {}
        self.zones_buttom = {}
        self.zones_type_id = set(self.zones_types.values())
        for zone_id in self.zones_type_id:
            self.zones_side[zone_id] = SataticZoneSide(self.zones_poly[str(zone_id)][1])
            self.zones_buttom[zone_id] = SataticZoneButtom(self.zones_poly[str(zone_id)][0])
        self.yolo = Yolo(
            weight=self.config.model.yolo_weights_path,
            model_frame_size=self.config.model.yolo_frame_size,
            device=self.device,
            confidence_threshold=self.config.model.confidence_threshold,
            iou_threshold=self.config.model.iou_threshold,
        )

    def start(self):
        logger.info("Files from source %s", self.source)
        files = glob.glob(self.source)
        logger.info("Files count %s", len(list(files)))
        batched = chunks_generator(files, 10)
        for batch in batched:
            start = time.perf_counter()
            batch_list_pic = []
            for pic in batch:
                batch_list_pic.append(cv2.imread(pic))
            frames_meta = []

            frames = torch.from_numpy(np.stack(batch_list_pic)).to(self.device)

            frames = frames[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
            predicts = self.yolo(frames)
            for i, detection in enumerate(predicts):
                frame_meta = FrameMeta(
                    model_config=self.config.model,
                    frame=batch_list_pic[i],
                    frame_ix=i,
                    raw_detections=detection,
                    file_name=batch[i],
                    zone_type=self.zones_types[Path(batch[i]).name],
                )
                frames_meta.append(frame_meta)

            for frame_meta in frames_meta:
                if frame_meta.is_empty():
                    continue
                for detection in frame_meta.detections:
                    detection.make_crop()
            for frame_meta in frames_meta:
                frames_meta = self.zones_side[frame_meta.zone_type].zone_intersect(frames_meta)
                frames_meta = self.zones_buttom[frame_meta.zone_type].zone_intersect(frames_meta)
                file_path_post = Path(frame_meta.file_name)
                for detection in frame_meta.detections:
                    self.coco_maker.add_an(file_path_post.name, detection.get_bbox())
                marked_frame = draw_results(frame_meta)
                self.write_img(str(self.dest / file_path_post.stem) + "_res.jpg", marked_frame)
            delta = time.perf_counter() - start
            logger.info("Batch time  %s ", str(delta))

        self.coco_maker.create_json()

    def write_img(self, files_name: str, img: np.ndarray):
        writer_thread = multiprocessing.Process(
            target=self._write_img_t,
            args=(files_name, img),
        )
        writer_thread.daemon = True
        writer_thread.start()

    def _write_img_t(self, files_name: str, img: np.ndarray):
        cv2.imwrite(files_name, img)


@hydra.main(config_path="../config", config_name="config")
def main(galeati_config: GaleatiConfig):
    app = App(galeati_config)

    try:
        app.start()
    finally:
        logger.info("Stop")


if __name__ == "__main__":
    main()

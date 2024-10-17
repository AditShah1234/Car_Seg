import torch
import sys
import time

sys.path.append("../")
from util.gaussian_blur import gaussian_blur
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm


class YOLO:
    def __init__(self, sub_model="yolov5n", conf_threshold=0.5):
        self.model = torch.hub.load("ultralytics/yolov5", sub_model, pretrained=True)
        self.model.classes = [2]  # For Car
        self.conf_threshold = conf_threshold

    def infer(self, frame_list):
        self.model.eval()
        frame = [image[..., ::-1] for image in frame_list]
        results = self.model(frame, size=640)

        return results

    def infer_batch(self, image_list, batch_size=16):
        self.model.eval()
        results = []
        start_time = time.time()
        no_images = len(image_list)
        for i in tqdm(range(0, no_images, batch_size)):
            if i + batch_size > no_images - 1:
                f = image_list[i:]
            else:
                f = image_list[i : i + batch_size]
            r = self.infer(f)
            results.append(r)
        end_time = time.time()
        self.results = results
        logger.info(
            "Avg time for prediction per frame done {}".format(
                (end_time - start_time) / no_images
            )
        )
        logger.info("Prediction done")
        return results

    def apply_gausian_blur(self, image_list, results):
        if results == None:
            results = self.results
        output = []

        for i in results:
            for j in range(len(i.ims)):
                frame, result = i.ims[j], i.xyxy[j]

                car_boxes = []
                for k in result:
                    if k[4] > self.conf_threshold:
                        car_boxes.append(k[:4])
                if car_boxes:
                    gb = gaussian_blur()
                    frame = gb.apply_gaussian_blur(frame, car_boxes)
                    output.append(frame)

        return output

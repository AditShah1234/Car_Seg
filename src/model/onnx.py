import torch
import sys
import time

sys.path.append("../")
from util.gaussian_blur import gaussian_blur
import logging
import onnxruntime as ort
from yolov5.utils.general import non_max_suppression

logger = logging.getLogger(__name__)
from tqdm import tqdm
import cv2
import numpy as np


class YOLO_onnx:
    def __init__(self, sub_model="yolov5n", conf_threshold=0.5):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.onnx")
        self.model = ort.InferenceSession("yolov5s.onnx")
        self.class_id = 2  # For Car
        self.conf_threshold = conf_threshold

    def infer(self, frame_list):

        frame = [
            cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
            .astype(np.float16)
            .transpose((2, 0, 1))
            for image in frame_list
        ]
        ort_output = self.model.get_outputs()[0].name
        ort_inputs = {self.model.get_inputs()[0].name: frame}
        results = self.model.run([ort_output], ort_inputs)
        output = torch.from_numpy(np.asarray(results))
        results = non_max_suppression(output, conf_thres=0.2, iou_thres=0.5)[0]
        return results

    def infer_batch(self, image_list, batch_size=16):

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
        for i in zip(image_list, results):
            frame, result = i[0], i[1]

            car_boxes = []
            for i, res in enumerate(result):
                if (
                    res[5] == self.class_id and res[4] > self.conf_threshold
                ):  # Allowing car class only andallowing only prediction above the threshold
                    car_boxes.append(res[:4])

            # Apply Gaussian blur on detected cars
            if car_boxes:
                gb = gaussian_blur()
                frame = gb.apply_gaussian_blur(frame, car_boxes)  # adding Gausian Blur
                output.append(frame)

        return output

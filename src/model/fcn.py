import torch
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


import matplotlib.pyplot as plt
import torchvision.transforms as T

from util.gaussian_blur import gaussian_blur


class FCN:
    def __init__(self, sub_model="fcn", conf_threshold=0.5):
        self.weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(weights=self.weights, progress=False)
        self.classes = "car"
        self.conf_threshold = conf_threshold
        self.transforms = self.weights.transforms(resize_size=None)

    def infer(self, frame):
        self.model.eval()
        results = self.model(frame, size=640)
        results = results.xyxy[0][results.xyxy[0][:, 4] > self.conf_threshold]
        return results

    def infer_batch(self, image_list):

        self.transform = T.Compose([T.ToTensor()])
        img = self.transform(image_list)

        frame = [
            image[..., ::-1] for image in image_list
        ]  # to convert OpenCV image (BGR to RGB)
        results = self.infer(frame)
        self.results = results
        return results

    def apply_gausian_blur(self, image_list, results):
        if results == None:
            results = self.results
        output = []
        for i, frame, results in enumerate(image_list, results):
            car_boxes = []
            for *box, conf, cls in results:
                if int(cls) == 2:
                    car_boxes.append(box)

            # Apply Gaussian blur on detected cars
            if car_boxes:
                gb = gaussian_blur
                frame = gb.apply_gaussian_blur(frame, car_boxes)
                output.append(frame)

        return output

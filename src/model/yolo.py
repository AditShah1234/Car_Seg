import torch
import sys
sys.path.append('../')
from util.gaussian_blur import gaussian_blur
class YOLO():
    def __init__(self,sub_model = "yolov5n",conf_threshold=.5):
        self.model = torch.hub.load('ultralytics/yolov5',sub_model, pretrained=True)
        self.model.classes = [2]# For Car
        self.conf_threshold =  conf_threshold
    
    def infer(self,frame):
        self.model.eval()
        results = self.model(frame, size=640)
        results = results.xyxy[0][results.xyxy[0][:, 4] > self.conf_threshold]
        return results
    
    def infer_batch(self, image_list):
        self.model.eval()
        frame = [image[..., ::-1] for image in image_list]# to convert OpenCV image (BGR to RGB)
        results = self.infer(frame)
        self.results =results
        return results
    
    def apply_gausian_blur(self,image_list,results):
        if results ==None:
            results = self.results
        output =[]
        for i, frame , results in enumerate(image_list,results):
            car_boxes = []
            for *box, conf, cls in results:
                if int(cls) == 2: 
                    car_boxes.append(box)

            # Apply Gaussian blur on detected cars
            if car_boxes:
                frame = gaussian_blur.apply_gaussian_blur(frame, car_boxes)
                output.append(frame)
            
        return output
        
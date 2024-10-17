# Car Detection Project
[![Video Title](https://github.com/AditShah1234/Car_Seg/blob/main/asset/output.png)](https://github.com/AditShah1234/Car_Seg/blob/main/asset/output_fastRCNN.mp4)
This project allows for object detection using different models such as YOLOv5, Fast R-CNN, and SSD. The main scripts can be used to compare model performances, run object detection on video or image files, or perform real-time object detection.

## Main Scripts

### 1. `compare.py`
   - This script compares the performance of different models on a given dataset or input file. It can evaluate models using various metrics such as accuracy and inference speed.

### 2. `demo.py`
   - This script demonstrates object detection on a given video or image file using a specified model. It processes the input file and saves the output with the detected objects.

### 3. `real_time.py`
   - This script performs real-time object detection using a specified model. It can be used for live video processing, making it suitable for applications like surveillance or monitoring.

## How to Use

Each script accepts command-line arguments for configuration, managed through `argparse`. Below is a description of the common arguments available:

### Arguments

- **`--model_name`**: 
   - Type: `str`
   - Choices: `["YOLOv5", "fastRCNN", "Fast_rcnn_half", "SSD"]`
   - Default: `"YOLOv5"`
   - Description: Specifies the model to be used for object detection.

- **`--input_file`**:
   - Type: `str`
   - Default: `"data/hanoi-traffic-clip.mov"`
   - Description: Path to the input video or image file.

- **`--output_file`**:
   - Type: `str`
   - Default: `"output_yolo.mp4"`
   - Description: Path to save the output file with detections (default: `output.mp4`).

- **`--device`**:
   - Type: `str`
   - Choices: `["cpu", "cuda"]`
   - Default: `"cpu"`
   - Description: Specifies the device to run the model on (`cpu` or `cuda` for GPU).

- **`--batch_size`**:
   - Type: `int`
   - Default: `16`
   - Description: The batch size for processing (should be more than 0).

- **`--skip_frame`**:
   - Type: `int`
   - Default: `1`
   - Description: Skips frames to speed up processing.

- **`--confidence`**:
   - Type: `float`
   - Default: `0.4`
   - Description: Sets the confidence threshold for detection (range: 0-1).

## Results

- Video Clip FPS: 60fps
- Achieved FPS using Yolov5n : 7fps

## Citation


```text
@software{yolov5,
  title = {YOLOv5 by Ultralytics},
  author = {Glenn Jocher},
  year = {2020},
  version = {7.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/yolov5},
  doi = {10.5281/zenodo.3908559},
  orcid = {0000-0001-5950-6979}
}

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

import torchvision.transforms as T
import torchvision
import argparse
from util.utils import video_utils
from model.yolo import YOLO

def infer(args):
    print(str(args.input_file))
    if str(args.input_file).endswith(("mp4","mkv","mov")):
        video_util = video_utils()
    if str(args.model_name) == "YOLOv5":
        model = YOLO()
    
    
    frames= video_util.load_video(args.input_file)
    result = model.infer_batch(frames)
    output = model.apply_gausian_blur(frames,result)
    video_util.save_video(output,args.output_file)
    
    
        
        
        
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for running a model on a given input video or image.")
    parser.add_argument(
        '--model_name',
        type=str,
        default='YOLOv5',
        help="Name of the model"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to the input video or image file."
    )

    # Additional optional arguments
    parser.add_argument(
        '--output_file',
        type=str,
        default="output.mp4",
        help="Path to save the output file (default: output.mp4)."
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help="Device to run the model on (default: cpu)."
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Model Name: {args.model_name}")
    print(f"Input File: {args.input_file}")
    print(f"Output File: {args.output_file}")
    print(f"Device: {args.device}")
    infer(args)


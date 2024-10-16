import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import os
class video_utils():
    def __init__(self,frame_width=None,frame_height=None,fps=None,frame_skip=1):
        
        self.frame_width=frame_width
        self.frame_height=frame_height
        self.fps=fps
        self.frame_skip = frame_skip
    
    def load_video(self,input_video_path):
        video = cv2.VideoCapture(input_video_path)
        output = []
        if not self.frame_width:
            self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not self.frame_height:
            self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.fps:
            self.fps = video.get(cv2.CAP_PROP_FPS)

        frame_idx = 0
        while(video.isOpened()):
            ret, frame = video.read()
            if frame_idx % self.frame_skip == 0:
                output.append(frame)
        
        video.release()
        cv2.destroyAllWindows()
        return output


    def save_video(self,list_frame,output_video_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        for frame in list_frame:
            out.write(frame)
        out.release()
        
        
class images_utils():
    def __init__(self,frame_width=640,frame_height=640):
        self.frame_width=frame_width
        self.frame_height=frame_height
    def load_images(self, image_folder,frame_width =None,frame_height=None ):
        if not frame_width:
            frame_width = self.frame_width
        else:
            self.frame_width = frame_width
        if not frame_height:
            frame_height = self.frame_height
        else:
            self.frame_height=frame_height
        image_paths = list(Path(image_folder).glob('*.[jp][pn]*'))
        
        images = [Image.open(path).convert('RGB') for path in image_paths]
        # Resize images and convert to tensor format
        images = [cv2.resize(np.array(image), (frame_width, frame_height)) for image in images]

        return images
    
    def save_images(self, list_frame, output_image_folder):
        for i,frame in enumerate(list_frame):
            filename = os.path.join(output_image_folder,i+".jpg")
            cv2.imwrite(filename,frame)
    
    
        


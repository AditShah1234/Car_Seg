import matplotlib.pyplot as plt
import numpy as np
import cv2


def frame_show(list_image):
    for indx, image in enumerate(list_image):
        cv2.imshow("Image-{}".format(indx), image)
        cv2.waitKey(600)

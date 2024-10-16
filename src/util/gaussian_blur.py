import cv2

class gaussian_blur():
    def __init__(self) -> None:
        pass
    def apply_gaussian_blur(self,image, boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            car_roi = image[y1:y2, x1:x2]  # Extract region of interest (car)
            blurred_car = cv2.GaussianBlur(car_roi, (51, 51), 0)  # Apply Gaussian blur
            image[y1:y2, x1:x2] = blurred_car  # Replace car area with blurred version
        return image
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


class FindAndCropSpheroid:
    def __init__(self, margin=10, debug=False):
        self.margin = margin
        self.debug = debug

    def __call__(self, img_set):
        bf, dead, live = img_set

        # Join the images into a three-channel image
        image_np = np.concatenate((bf, dead, live), axis=2)

        # Image processing to find contours and crop
        blurred = cv2.GaussianBlur(image_np, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            #raise ValueError("No contours found in the image.")
            return image_np



        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        side_length = max(w, h) + self.margin  # Adding margin
        center_x, center_y = x + w // 2, y + h // 2
        x = max(center_x - side_length // 2, 0)
        y = max(center_y - side_length // 2, 0)
        x_end = min(x + side_length, image_np.shape[1])
        y_end = min(y + side_length, image_np.shape[0])

        if self.debug:
            # Draw rectangle on image
            cv2.rectangle(image_np, (x, y), (x_end, y_end), (0, 255, 0), 2)
            # Convert back to PIL Image and save
            return Image.fromarray(image_np)

        cropped_image_np = image_np[y:y_end, x:x_end]

        # Convert to PIL Image and return
        return Image.fromarray(cropped_image_np)

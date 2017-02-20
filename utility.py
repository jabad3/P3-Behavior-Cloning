import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

# import image given a path
def load_image_values(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = preprocess(image)
    return image

def preprocess(image):
    image = crop_resize(image)
    image = image.astype(np.float32)
    image = image/255.0-0.5 # normalize image
    return image

def crop_resize(image):
    cropped = image[55:135, :, :] # convert image from 160x320x3 to 64x64x3
    resized = resize_image(cropped)
    return resized

def resize_image(image):
    NEW_SIZE = (64, 64)
    return cv2.resize(image, NEW_SIZE)
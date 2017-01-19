import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def preprocess_image(image):
    #Crop and resize
    image = crop_and_resize(image)
    image = image.astype(np.float32)

    #Normalize image
    image = image/255.0 - 0.5
    return image

def resize_to_target_size(image):
    TARGET_SIZE = (64, 64)
    return cv2.resize(image, TARGET_SIZE)

def crop_and_resize(image):
    '''
    :param image: The input image of dimensions 160x320x3
    :return: Output image of size 64x64x3
    '''
    cropped_image = image[55:135, :, :]
    processed_image = resize_to_target_size(cropped_image)
    return processed_image

def load_image_values(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = preprocess_image(image)
    return image
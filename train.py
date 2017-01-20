import numpy as np
import pandas as pd
import cv2
from model import get_model
from utility import load_image_values


# load data file
data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])

# shuffle the data
data_frame = data_frame.sample(frac=1).reset_index(drop=True)

x_images = []
y_labels = []
for index, row in data_frame.iterrows():
    LEFT_RIGHT_CORRECTION_CONSTANT = 0.30
    
    left_image_path = row['left']
    center_image_path = row['center']
    right_image_path = row['right']
    center_steering_value = row['steering']
    
    left_image = load_image_values("data/" + left_image_path.strip())
    center_image = load_image_values("data/" + center_image_path.strip())
    right_image = load_image_values("data/" + right_image_path.strip())
    
    # add images
    x_images.append(left_image)
    y_labels.append(center_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_image)
    y_labels.append(center_steering_value)
    
    x_images.append(right_image)
    y_labels.append(center_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)

    # mini augment
    left_flipped_image = cv2.flip(left_image, 1)
    center_flipped_image = cv2.flip(center_image, 1)
    right_flipped_image = cv2.flip(right_image, 1)
    center_flipped_steering_value = center_steering_value*-1
    
    x_images.append(left_flipped_image)
    y_labels.append(center_flipped_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_flipped_image)
    y_labels.append(center_flipped_steering_value)
    
    x_images.append(right_flipped_image)
    y_labels.append(center_flipped_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)

    
# release the main data_frame from memory
data_frame = None







####
# load curves data
data_frame = pd.read_csv('curves/driving_log.csv', usecols=[0, 1, 2, 3])

# shuffle the data
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
print(data_frame)

x_images = []
y_labels = []
for index, row in data_frame.iterrows():
    #LEFT_RIGHT_CORRECTION_CONSTANT = 0.20
    
    #left_image_path = row['left']
    center_image_path = row['center']
    #right_image_path = row['right']
    center_steering_value = row['steering']
    
    #left_image = load_image_values("curves/" + left_image_path.strip())
    center_image = load_image_values("curves/" + center_image_path.strip())
    #right_image = load_image_values("curves/" + right_image_path.strip())
    
    # add images
    #x_images.append(left_image)
    #y_labels.append(center_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_image)
    y_labels.append(center_steering_value)
    
    #x_images.append(right_image)
    #y_labels.append(center_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)

    # mini augment
    #left_flipped_image = cv2.flip(left_image, 1)
    center_flipped_image = cv2.flip(center_image, 1)
    #right_flipped_image = cv2.flip(right_image, 1)
    center_flipped_steering_value = center_steering_value*-1
    
    #x_images.append(left_flipped_image)
    #y_labels.append(center_flipped_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_flipped_image)
    y_labels.append(center_flipped_steering_value)
    
    #x_images.append(right_flipped_image)
    #y_labels.append(center_flipped_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)

    
# release the main data_frame from memory
data_frame = None
####








####
# load error data
data_frame = pd.read_csv('edge_recovery/driving_log.csv', usecols=[0, 1, 2, 3])

# shuffle the data
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
print(data_frame)

x_images = []
y_labels = []
for index, row in data_frame.iterrows():
    #LEFT_RIGHT_CORRECTION_CONSTANT = 0.20
    
    #left_image_path = row['left']
    center_image_path = row['center']
    #right_image_path = row['right']
    center_steering_value = row['steering']
    
    #left_image = load_image_values("curves/" + left_image_path.strip())
    center_image = load_image_values("edge_recovery/" + center_image_path.strip())
    #right_image = load_image_values("curves/" + right_image_path.strip())
    
    # add images
    #x_images.append(left_image)
    #y_labels.append(center_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_image)
    y_labels.append(center_steering_value)
    
    #x_images.append(right_image)
    #y_labels.append(center_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)

    # mini augment
    #left_flipped_image = cv2.flip(left_image, 1)
    center_flipped_image = cv2.flip(center_image, 1)
    #right_flipped_image = cv2.flip(right_image, 1)
    center_flipped_steering_value = center_steering_value*-1
    
    #x_images.append(left_flipped_image)
    #y_labels.append(center_flipped_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_flipped_image)
    y_labels.append(center_flipped_steering_value)
    
    #x_images.append(right_flipped_image)
    #y_labels.append(center_flipped_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)

    
# release the main data_frame from memory
data_frame = None
####















x_images = np.asarray(x_images)
y_labels = np.asarray(y_labels)

model = get_model()
#model.fit(training_generator, validation_data=validation_data_generator, samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)
#model.fit(x_train_data, Y_train_data, batch_size=128, nb_epoch=2, validation_split=0.2)
model.fit(x_images, y_labels, batch_size=128, nb_epoch=100, validation_split=0.2)

print("Saving model.")
model.save_weights('model.h5')
with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())
import pandas as pd
import numpy as np
import cv2
from model import create_model
from utility import load_image_values

# constant for left and right images
LEFT_RIGHT_CORRECTION_CONSTANT = 0.15

# load data file
data = pd.read_csv('supreme/driving_log.csv', usecols=[0, 1, 2, 3]) #center,left,right,steering cols
data = data.sample(frac=1).reset_index(drop=True) # shuffle the data
    
training_split = 0.80
num_rows_training = int(data.shape[0]*training_split)
training_data = data.loc[0:num_rows_training-1]
validation_data = data.loc[num_rows_training:]
data = None

def get_data_row(row):
    x_images = []
    y_labels = []
    
    left_image_path = row['left']
    center_image_path = row['center']
    right_image_path = row['right']
    center_steering_value = row['steering']
    
    left_image = load_image_values("supreme/" + left_image_path.strip())
    center_image = load_image_values("supreme/" + center_image_path.strip())
    right_image = load_image_values("supreme/" + right_image_path.strip())
    
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
    y_labels.append(center_flipped_steering_value-LEFT_RIGHT_CORRECTION_CONSTANT)
    
    x_images.append(center_flipped_image)
    y_labels.append(center_flipped_steering_value)
    
    x_images.append(right_flipped_image)
    y_labels.append(center_flipped_steering_value+LEFT_RIGHT_CORRECTION_CONSTANT)
    
    return x_images, y_labels

def data_generator(data, batch_size=32):
    N = data.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0
        for index, row in data.loc[start:end].iterrows():
            x_images, y_labels = get_data_row(row)
            for i in range(6):
                X_batch[j], y_batch[j] = x_images[i], y_labels[i];
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            i = 0
        yield (X_batch, y_batch)


t_generator = data_generator(training_data)
v_generator = data_generator(validation_data)

model = create_model()
model.fit_generator(t_generator, 
                    validation_data=v_generator,
                    samples_per_epoch=40000,
                    nb_val_samples=6000,
                    nb_epoch=5)

print("Saving model!")
model.save_weights('model.h5')
with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())
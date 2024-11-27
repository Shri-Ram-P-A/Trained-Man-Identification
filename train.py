# %% Import the requried Libraries
import os
import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

#%% To store the image on the specfied folder

main_folder = r"D:\Projects\Trained Man Identification\images"
s = input("Enter folder name: ").strip().lower()
nested_folder = os.path.join(main_folder, s)
os.makedirs(nested_folder, exist_ok=True)

#%%

def data_augmentation(img,nested_folder):

    img_format = ".jpg"
    formatted_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")
    img_filename = f"frame_{formatted_datetime}{img_format}"
    path = os.path.join(nested_folder, img_filename)
    cv2.imwrite(path, img)

    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_filename1 = "grayscaled_" + img_filename
    path_gray = os.path.join(nested_folder, img_filename1)
    cv2.imwrite(path_gray, grayscaled)


    saturated = tf.image.adjust_saturation(img, 3)
    saturated = saturated.numpy() 
    img_filename2 = "saturated_" + img_filename
    path_saturated = os.path.join(nested_folder, img_filename2)
    cv2.imwrite(path_saturated, saturated)


    bright = tf.image.adjust_brightness(img, 0.5).numpy() 
    img_filename1 = "bright_" + img_filename
    path_bright = os.path.join(nested_folder, img_filename1)
    cv2.imwrite(path_bright, bright)

    dim = tf.image.adjust_brightness(img, -0.5).numpy() 
    img_filename2 = "dim_" + img_filename
    path_dim = os.path.join(nested_folder, img_filename2)
    cv2.imwrite(path_dim, dim)

#%% Live image upload

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    data_augmentation(frame,nested_folder)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cap.release()

# %% load the image

img = cv2.imread(r"C:\Users\shrir\OneDrive\Pictures\Camera Roll\WIN_20231129_16_52_48_Pro.jpg")
if img is None:
    print("Error: Image not found or path is incorrect.")
else:
    data_augmentation(img, nested_folder)

# %%

labels = []
images = []

for nested_folder_path in os.listdir(main_folder):
    for img_path in os.listdir(os.path.join(main_folder,nested_folder_path)):
        image = cv2.imread(os.path.join(main_folder,nested_folder_path,img_path))
        if image is not None:
                image_resized = cv2.resize(image, (224, 224))
                images.append(image_resized)
                labels.append(nested_folder_path)


images = np.array(images)
labels = np.array(labels)

#%%

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

#%%

size_of_folder = len(os.listdir(main_folder))

model = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%

model.fit(images,labels_categorical,epochs = 5)

#%%

# Load the image for prediction
image_path = r"C:\Users\shrir\OneDrive\Pictures\Camera Roll\WIN_20231214_18_58_02_Pro.jpg"
img = cv2.imread(image_path)

if img is not None:
    # Resize and preprocess the image
    img = cv2.resize(img, (224, 224))
    img_array = img.astype(np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index and convert it to the label
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]  # Get original label

    print(f"Predicted Label: {predicted_label}")
else:
    print("Error: Image not found or path is incorrect.")


# %%

loss,accuracy = model.evaluate(images,labels_categorical)
print(loss,accuracy)

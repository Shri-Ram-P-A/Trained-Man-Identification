import os
import cv2
import streamlit as st
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

st.title("Real-Time Image Identification")

class ImageIdentification:
    def __init__(self, folder_name="demo") -> None:
        self.main_folder = "images"
        self.temp_folder = "demo"
        self.nested_folder = os.path.join(self.main_folder, folder_name)
        os.makedirs(self.nested_folder, exist_ok=True)


def data_augmentation(img, nested_folder):
    img_format = ".jpg"
    formatted_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")
    img_filename = f"frame_{formatted_datetime}{img_format}"
    path = os.path.join(nested_folder, img_filename)
    cv2.imwrite(path, img)

    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(nested_folder, f"grayscaled_{img_filename}"), grayscaled)

    saturated = tf.image.adjust_saturation(img, 3).numpy()
    cv2.imwrite(os.path.join(nested_folder, f"saturated_{img_filename}"), saturated)

    bright = tf.image.adjust_brightness(img, 0.5).numpy()
    cv2.imwrite(os.path.join(nested_folder, f"bright_{img_filename}"), bright)

    dim = tf.image.adjust_brightness(img, -0.5).numpy()
    cv2.imwrite(os.path.join(nested_folder, f"dim_{img_filename}"), dim)


def capture_image(nested_folder=None):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        st.image(frame, caption="Captured Image", channels="BGR")
        if nested_folder:
            data_augmentation(frame, nested_folder)

    cap.release()
    return frame


def data(main_folder):
    labels = []
    images = []

    for nested_folder_path in os.listdir(main_folder):
        for img_path in os.listdir(os.path.join(main_folder, nested_folder_path)):
            image = cv2.imread(os.path.join(main_folder, nested_folder_path, img_path))
            if image is not None:
                image_resized = cv2.resize(image, (224, 224))
                images.append(image_resized)
                labels.append(nested_folder_path)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def apply_cnn(images, labels, image):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    model = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels_categorical, epochs=5)

    img = cv2.resize(image, (224, 224))
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label


# Sidebar
st.sidebar.header("Sidebar Menu")
user_input = st.sidebar.text_input("Enter your Image Name:")
folder = ImageIdentification(user_input)
nested_folder = folder.nested_folder

if user_input:
    st.sidebar.subheader("Live Photo")
    if st.sidebar.button("Capture Image"):
        capture_image(nested_folder)

    st.sidebar.subheader("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(img, caption="Uploaded Image", channels="BGR")
        data_augmentation(img, nested_folder)

# Page
st.header("Import the Image")
folder_page = ImageIdentification()

if len(os.listdir(folder_page.main_folder)):
    st.subheader("Test Live Photo")
    images, labels = data(folder_page.main_folder)

    if st.button("Test Capture Image"):
        image = capture_image()
        if image is not None:
            answer = apply_cnn(images, labels, image)
            st.write(answer)

    st.subheader("Test Upload Image")
    uploaded_file = st.file_uploader("Choose a testing image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Testing Image", channels="BGR")
        if image is not None:
            answer = apply_cnn(images, labels, image)
            st.write(answer)
else:
    st.write("Upload the image to start testing.")

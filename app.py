import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json  # <-- Explicitly import Sequential

# Load the model architecture
with open("emotion_model1.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load the weights
model.load_weights("emotion_model1.h5")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load architecture
with open("emotion_model1.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load weights
model.load_weights("emotion_model1.h5")

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Emotion labels (based on your final Dense layer with 5 units)
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Adjust if different

# Load pre-trained emotion detection model
model = load_model("emotion_model1.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Real-Time Emotion Detection")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not accessible")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)

            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

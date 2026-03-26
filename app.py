import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import class_names
import streamlit as st
from predict import predict
from decision import decision
model = load_model("model.h5")
st.title("Vehicle Classification System 🚗")
option = st.radio("Select Input Method:", ["Upload Image", "Live Camera"])
if option == "Upload Image":
    file = st.file_uploader("Upload an image")
    if file:
        pred, conf = predict(file)
        result = decision(conf)

        st.image(file, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {pred}")
        st.write(f"Confidence: {conf:.2f}")
        st.write(f"Decision: {result}")
elif option == "Live Camera":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to access camera")
            break

        # Convert frame for prediction
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        class_id = np.argmax(pred)
        confidence = float(np.max(pred))

        label = class_names[class_id]
        result = decision(confidence)

        # Show text on frame
        cv2.putText(frame, f"{label} {confidence:.2f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()
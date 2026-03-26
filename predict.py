import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess, class_names

model = load_model("model.h5")

def predict(image):
    img = preprocess(image)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)

    class_id = np.argmax(pred)
    confidence = float(np.max(pred))

    return class_names[class_id], confidence
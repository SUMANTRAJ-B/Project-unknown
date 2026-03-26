import cv2
import numpy as np

IMG_SIZE = 224

class_names = ['car','bike','bus','truck','ambulance']

def preprocess(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img
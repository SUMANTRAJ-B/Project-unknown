#  Vehicle Type Classification System (Image-Based)

This project presents an intelligent image-based vehicle classification system built using deep learning. The system is capable of identifying vehicle types from input images and classifying them into five categories:

* Car
* Bike
* Bus
* Truck
* Ambulance

In addition to classification, the system includes a **confidence-based decision layer** that evaluates the reliability of predictions.

---

##  Objective

The main objective of this project is to:

* Build a robust image classification model using CNN (Convolutional Neural Network)
* Accurately classify vehicles into predefined categories
* Introduce an intelligent decision system based on prediction confidence
* Provide special emphasis on **ambulance detection** for real-world applications

---

##  Model Architecture

The model is based on **Transfer Learning** using MobileNetV2:

* MobileNetV2 (Pretrained CNN on ImageNet)
* Global Average Pooling Layer
* Dense Layer (ReLU activation)
* Dropout Layer (0.5) for regularization
* Output Layer (Softmax activation – 5 classes)

This architecture ensures:

* Faster training
* Better generalization
* High accuracy with limited data

---

##  Preprocessing

To ensure consistency and improve model performance, the following preprocessing steps are applied:

* Image resizing to **224 × 224 pixels**
* Pixel normalization (values scaled between 0 and 1)
* Data augmentation techniques:

  * Horizontal flipping
  * Rotation (±20 degrees)
  * Zoom transformation

The dataset is structured into:

* Training set
* Validation set
* Test set

---

##  Training & Evaluation

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Evaluation Metric:** Accuracy
* **Epochs:** 8–10

###  Performance

* Training Accuracy: ~85% – 95%
* Validation Accuracy: ~80% – 90%
* Test Accuracy: ~80%+ (depending on dataset quality)

---

##  Decision Layer

A rule-based intelligent decision layer is implemented to evaluate prediction confidence:

| Confidence Score | Decision          |
| ---------------- | ----------------- |
| ≥ 0.85           |  High Confidence |
| 0.65 – 0.85      |  Needs Review    |
| < 0.65           |  Uncertain      |

This enhances reliability and helps identify ambiguous predictions.

---

##  Special Feature: Ambulance Detection

* Ambulance is treated as a **separate critical class**
* Dataset includes additional ambulance images to improve detection
* Model learns distinctive features such as:

  * Red cross markings
  * Emergency lighting patterns
  * Unique vehicle structure

This feature makes the system suitable for **smart traffic and emergency response systems**

---

##  Output

For each input image, the system provides:

* Predicted Vehicle Class
* Confidence Score (Probability)
* Decision Label (High / Review / Uncertain)

---

##  Demo

The application is built using **Streamlit** and supports:

*  Image Upload for classification
*  Live Camera Input (optional)
*  Real-time display of:

  * Prediction
  * Confidence score
  * Decision output

---

##  Project Structure

```text
project/
 ├── app.py              # Streamlit application (UI)
 ├── train.py            # Model training script
 ├── predict.py          # Prediction logic
 ├── decision.py         # Decision layer implementation
 ├── utils.py            # Preprocessing functions
 ├── model.h5            # Trained model file
 ├── requirements.txt    # Dependencies
 └── README.md           # Project documentation
```

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vehicle-classification.git
cd vehicle-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional)

```bash
python train.py
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## Constraints

* Model performance depends on dataset quality and balance
* Similar vehicle types may cause misclassification
* Training is slower without GPU support
* Real-time performance may vary based on system capability

---

## Future Enhancements

* Integration with real-time traffic signal systems
* Object detection (bounding boxes using YOLO/SSD)
* Deployment on edge devices (Raspberry Pi)
* Cloud-based deployment for scalability

---

## Conclusion

This project successfully demonstrates the use of CNN-based transfer learning combined with a confidence-based decision system to build a reliable and practical vehicle classification solution. The system is especially effective in detecting emergency vehicles like ambulances and can be extended for real-world intelligent traffic applications.

---

## Author

**
Kawin S S
Pooja S
Sumantraj B
Vijaya Sri M S
**

---

## Acknowledgments

* TensorFlow & Keras
* OpenCV
* Streamlit
* Kaggle (for datasets)

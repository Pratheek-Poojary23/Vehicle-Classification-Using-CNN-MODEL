
# 🚗 Vehicle Classification using CNN Model

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to automatically classify images of vehicles into three categories: **Cars, Trains, and Planes**. The system can help in **traffic management, surveillance, and autonomous transportation systems** by identifying vehicle types from images.

---

## 📂 Dataset
- **Number of Classes:** 3 (Car, Train, Plane)
- **Total Images:** 1200
- **Images per Class:** 400
- **Conditions:** Different angles, lighting, and backgrounds for better generalization.

---

## 🎯 Problem Statement
To build and train a **CNN model** capable of classifying an image into one of the three vehicle categories: **Car, Train, or Plane**.

---

## 🛠️ Data Preprocessing
### Key Steps:
- **Resized all images to 128x128 pixels.**
- **Assigned numeric labels to each class (Car, Train, Plane).**
- **Normalized pixel values (0-255 scaled to 0-1).**
- **Split data into training and testing sets (80-20).**

```python
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
```

---

## 🏗️ CNN Model Architecture
The CNN model consists of:

| Layer | Type | Details |
|---|---|---|
| 1 | **Conv2D** | 32 filters, 3x3 kernel, ReLU activation |
| 2 | **MaxPooling2D** | 2x2 pool size |
| 3 | **Conv2D** | 64 filters, 3x3 kernel, ReLU activation |
| 4 | **MaxPooling2D** | 2x2 pool size |
| 5 | **Flatten** | Converts 2D feature maps to 1D |
| 6 | **Dense** | 128 neurons, ReLU activation |
| 7 | **Output Layer** | 3 neurons (Car, Train, Plane), Softmax activation |

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])
```

---

## ⚙️ Model Compilation
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metric:** Accuracy

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 🏋️ Model Training
- **Batch Size:** 32
- **Epochs:** 25
- **Train/Validation Split:** 80/20

```python
history = model.fit(train_generator, epochs=25, validation_data=validation_generator)
```

---

## 🧪 Model Evaluation
- **Metrics:** Accuracy, Loss, Classification Report, Confusion Matrix

```python
from sklearn.metrics import classification_report, confusion_matrix

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=['Car', 'Train', 'Plane']))
```

---

## 📊 Results
| Metric | Value |
|---|---|
| **Test Accuracy** | 89.16% |
| **Test Loss** | 0.27 |
| **Best Classified Class** | Car |

---

## 📈 Accuracy & Loss Curves
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training & Validation Loss')
plt.show()
```

---

## 🔍 Example Prediction
```python
import cv2
import numpy as np

image = cv2.imread('sample_image.jpg')
image = cv2.resize(image, (128, 128))
image = image / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
class_index = np.argmax(prediction)
class_labels = ['Car', 'Train', 'Plane']

print(f"Predicted Class: {class_labels[class_index]}")
```

---

## 📌 Conclusion
- The **CNN model achieved 89.16% accuracy**.
- The model handled variations in **lighting, angles, and backgrounds**.
- Possible improvements:
    - Use of **data augmentation**.
    - **Transfer learning** (using pre-trained models like VGG16 or ResNet).

---

## 📂 Folder Structure
```
.
├── data/
│   ├── train/           # Training images (Car, Train, Plane folders)
│   ├── test/            # Test images
├── src/
│   ├── train_model.py    # Training script
│   ├── evaluate_model.py # Evaluation script
│   ├── predict.py        # Prediction script
├── README.md
├── requirements.txt
└── cnn_vehicle_classification.ipynb
```
---

# 🩺 Medical Image Classification for Disease Detection

## 📘 Overview
This project uses **Deep Learning (Convolutional Neural Networks)** to automatically detect **Pneumonia** from **Chest X-ray images**.
It leverages the **Kaggle Chest X-Ray Images (Pneumonia)** dataset and builds a CNN model using **TensorFlow/Keras** for disease classification.

---

## 🧠 Objectives
- Classify chest X-ray images as **Normal** or **Pneumonia**.
- Implement an end-to-end deep learning pipeline for image preprocessing, model training, evaluation, and prediction.
- Improve model generalization using **data augmentation** and visualize performance using metrics and plots.

---

## 🧩 Dataset
**Source:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Structure after extraction:**
```
chest_xray/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## ⚙️ Requirements
Create a clean virtual environment before installing dependencies.

```bash
python -m venv tf_env
tf_env\Scripts\activate     # Windows
# or
source tf_env/bin/activate    # macOS/Linux
```

Then install:

```bash
pip install tensorflow-cpu==2.17.0 numpy==1.26.4 matplotlib seaborn scikit-learn opencv-python pillow
```

---

## 🏗️ Project Workflow

### 1️⃣ Data Preprocessing
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
```
- All images resized to **150×150**
- Normalized pixel values (0–1 range)
- Augmentation improves robustness

---

### 2️⃣ CNN Model Architecture
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

---

### 3️⃣ Model Compilation and Training
```python
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=15)
```

---

### 4️⃣ Evaluation and Visualization
- **Accuracy, Precision, Recall**
- **Confusion Matrix** using Seaborn
- **ROC Curve** using Scikit-learn

---

### 5️⃣ Saving and Loading Model
```python
model.save("medical_xray_pneumonia_model.h5")
loaded_model = load_model("medical_xray_pneumonia_model.h5")
```

---

### 6️⃣ Single Image Prediction
```python
def predict_new_image(img_path):
    img_path = os.path.join(os.getcwd(), img_path)
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = loaded_model.predict(img_array)[0][0]
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    print(f"🩺 Prediction: {label} (Confidence: {prediction:.2f})")

predict_new_image("chest_xray/chest_xray/test/PNEUMONIA/person78_bacteria_378.jpeg")
```

---

## 📊 Outputs
- **Training accuracy and loss curves**
- **Test accuracy**
- **Confusion matrix and ROC curve**
- **Visual samples of misclassifications**
- **Prediction on single image**

---

## 🧮 Example Results
| Metric | Value (Approx.) |
|--------|-----------------|
| Training Accuracy | 95%+ |
| Validation Accuracy | 90–93% |
| Test Accuracy | 88–92% |

*(Results vary based on epochs, augmentation, and dataset balance.)*

---

## 🖥️ Example Prediction Output
```
🩺 Prediction: Pneumonia (Confidence: 0.93)
```

---

## 📦 Folder Structure
```
📂 Medical-Image-Classification/
│
├── chest_xray/
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
│
├── medical_xray_pneumonia_model.h5
├── main_model_training.py
├── predict_image.py
└── README.md
```

---

## 💡 Future Enhancements
- Use **transfer learning** (e.g., VGG16, ResNet50, MobileNet).
- Deploy the model as a **web app** using Streamlit or Flask.
- Implement **Grad-CAM** visualization for explainable AI insights.

---

## 👨‍💻 Author
**Ankan Dutta**  
Deep Learning & Distributed Systems Enthusiast

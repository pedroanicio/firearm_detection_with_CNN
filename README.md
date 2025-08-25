# Firearm Detection using Convolutional Neural Networks (CNN)

This project implements a **machine learning system** for detecting firearms in images using **Convolutional Neural Networks (CNNs)**.  
The model is trained to both **classify whether a firearm is present** and to **localize it using bounding boxes**.

⚠️ **Note:** The dataset used in this project was **manually created and annotated** (not from public pretrained datasets), in compliance with academic requirements.

---

## 📌 Project Overview
- **Objective:** Detect and localize firearms in the hands of people using image processing and deep learning.  
- **Approach:**
  - Custom dataset (images collected from the internet and manually labeled).
  - Preprocessing: resizing, normalization, and data augmentation.
  - Model: CNN with dual outputs (binary classification + bounding box regression).
  - Evaluation: accuracy for classification and MSE for bounding box coordinates.

---

## 🧠 Techniques Applied
- **Convolutional Neural Networks (CNNs):** for image feature extraction.  
- **Bounding Box Regression:** predicts the position of the firearm in the image.  
- **Data Augmentation:** flipping, rotation, zoom, brightness/contrast adjustments.  
- **Image Normalization:** scales pixel values to `[0,1]`.  

---

## 🛠️ Technologies Used
- [Python 3.10](https://www.python.org/)  
- [TensorFlow / Keras](https://www.tensorflow.org/)  
- [OpenCV (cv2)](https://opencv.org/)  
- [NumPy](https://numpy.org/)  
- [LabelImg](https://github.com/heartexlabs/labelImg) (for bounding box annotation)  

---

## 📂 Project Structure

```├── treinamento.py # Training script (model definition, data augmentation, training loop)
├── teste_final.py # Final testing script (loads trained model, runs detection, draws bounding boxes)
├── imagens/ # Dataset images (raw and preprocessed)
├── labels/ # Bounding box annotations (XML / TXT / NPY)
├── modelo_final.h5 # Trained model (saved after training)
└── README.md # Project documentation
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/firearm-detection.git
cd TRABALHO2IA
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train the model

```
python treinamento.py
```

### 4. Test with new images

```
python teste_final.py
```
## 📚 References

```
- https://www.tensorflow.org/tutorials/images/cnn?hl=pt-br
- https://www.tensorflow.org/tutorials/images/classification?hl=pt-br
- https://keras.io/api/
- https://www.tensorflow.org/tutorials/images/data_augmentation?hl=pt-br
- https://www.tensorflow.org/api_docs
- https://github.com/CretivityArt/Custom-Object-Detection-Using-Python-OpenCV--Training-Database-using-TeachableMachine
- https://github.com/GurpreetKukkar/Weapon-Detection-using-Artificial-Intelligence
- https://www.youtube.com/watch?v=VGLAe-PINfE
- https://www.youtube.com/watch?v=4XhopE2fEf0
- https://www.youtube.com/watch?v=WvoLTXIjBYU
```


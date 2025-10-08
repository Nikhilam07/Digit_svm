# Digit_svm
# Handwritten Digit Recognition using SVM

## 📘 Project Overview

This project implements a **Support Vector Machine (SVM)** model using the **MNIST Digits Dataset** from `sklearn.datasets`. The dataset consists of images of handwritten digits (0–9), each represented as an 8x8 matrix of pixel intensities. The goal is to classify these digits accurately using a linear SVM classifier.

---

## 💻 Installation

To run this project, install the required Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 📊 Dataset Overview

* **Dataset Source:** `sklearn.datasets.load_digits()`
* **Number of Samples:** 1,797 images
* **Image Size:** 8x8 pixels (each pixel = intensity value from 0–16)
* **Features:** Flattened 64-pixel intensity values per image
* **Target Variable:** Actual digit label (0–9)

---

## 🧹 Data Preprocessing

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
digits = datasets.load_digits()

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Normalize pixel values to range [0, 1]
x_train = x_train / 16.0
x_test = x_test / 16.0
```

**Explanation:**
Normalization ensures that all pixel values are on a similar scale, which helps algorithms like SVM converge faster and perform better.

---

## 🤖 Model Training

```python
# Initialize SVM model
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the model
svm_classifier.fit(x_train, y_train)
```

**Notes:**

* **Kernel:** Linear — finds a linear hyperplane separating digit classes.
* **C (Regularization):** Controls the trade-off between achieving a low training error and a low testing error.

---

## 📈 Model Evaluation

```python
# Predict on test data
y_pred = svm_classifier.predict(x_test)

# Calculate metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
```

**Typical Results:**

* Accuracy: ~0.97–0.99
* Precision, Recall, F1: ~0.97+ (depending on train/test split)

---

## 🖼️ Visualization

```python
# Predict first 5 digits
new_digit_predictions = svm_classifier.predict(x_test[:5])

# Display images with predictions
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Predicted: {new_digit_predictions[i]}')
    plt.axis('off')

plt.show()
```

This visualization shows five handwritten digits from the test set along with their predicted labels.

---

## 📁 Project Structure

```
📦 svm-handwritten-digit-recognition
├── svm_digits.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

---

## 🧠 Insights

* Linear SVM performs extremely well for digit recognition due to the clear separability of MNIST features.
* Normalization significantly improves accuracy.
* For further improvement, try **non-linear kernels (RBF, polynomial)** or **SVM with PCA** for dimensionality reduction.

---



**End of README**


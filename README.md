# Spam Detection using Neural Networks (PyTorch)

## 📌 Project Overview

This project implements a spam detection system for SMS messages using a Multi-Layer Perceptron (MLP) built with PyTorch. The goal is to classify messages as **spam** or **ham (non-spam)** using numerical representations of text.

The project focuses on comparing different neural network architectures and analyzing their impact on performance, especially under class imbalance.

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Dataset: SMS Spam Collection
* Text normalization (lowercasing)
* Label encoding:

  * ham → 0
  * spam → 1

### 2. Feature Engineering

* TF-IDF (Term Frequency–Inverse Document Frequency)
* Limited to top features for efficiency
* Converts text into numerical vectors suitable for neural networks

### 3. Model Architecture (PyTorch)

Implemented using **PyTorch (torch.nn)**:

* Model 1: 128 → 64
* Model 2: 256 → 128 → 64
* Model 3: 512 → 256 → 128 → 64

Activation: ReLU
Loss: BCEWithLogitsLoss (with class weighting)
Optimizer: Adam

---

## ⚖️ Handling Class Imbalance

The dataset is imbalanced (~85% ham, ~15% spam).

To address this:

* Used **weighted binary cross-entropy loss**
* Increased importance of the minority class (spam)
* Avoided data-level methods like SMOTE due to TF-IDF sparsity

---

## 📊 Evaluation Metrics

Accuracy alone is not reliable due to class imbalance.

We evaluate using:

* Precision (spam)
* Recall (spam) ⭐
* F1-score (spam)

---

## 🧪 Results Summary

| Model   | Precision | Recall    | F1-score |
| ------- | --------- | --------- | -------- |
| Model 1 | 0.970     | 0.873     | 0.919    |
| Model 2 | 0.925     | **0.907** | 0.916    |
| Model 3 | 0.899     | 0.893     | 0.896    |

### ✅ Final Model: Model 2

* Best recall for spam detection
* Strong balance between precision and recall
* Additional depth beyond this led to diminishing returns

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install torch scikit-learn pandas
```

2. Run the notebook:

```bash
jupyter notebook
```

---

## 📁 Project Structure

* Notebook: preprocessing, training, evaluation
* Models implemented using PyTorch
* TF-IDF via scikit-learn

---

## 👥 Team

* Hazem Mohamed El-Kanawati
* Youssef Hany Sayed

---

## 📌 Key Takeaways

* Moderate-depth networks performed best
* Increasing model complexity does not always improve performance
* Recall is the most critical metric in spam detection
* Proper handling of class imbalance is essential

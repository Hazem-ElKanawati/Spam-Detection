# Spam Detection using Neural Networks (MLP)

## 📌 Project Description

This project implements a spam detection system using a Multi-Layer Perceptron (MLP) neural network. The model classifies SMS messages as either spam or ham (not spam) using TF-IDF features.

---

## ⚙️ Approach

* Text preprocessing (lowercasing)
* Feature extraction using TF-IDF
* Training neural network models with different architectures
* Handling class imbalance using weighted loss
* Evaluation using precision, recall, and F1-score

---

## 🧪 Experiments

Three models were tested:

* Model 1: 128 → 64
* Model 2: 256 → 128 → 64
* Model 3: 512 → 256 → 128 → 64

Model 2 achieved the best balance between precision and recall.

---

## 📊 Evaluation Metrics

Due to class imbalance, the following metrics were used:

* Precision
* Recall (most important)
* F1-score

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

## 📁 Dataset

SMS Spam Collection dataset

---

## 👥 Team

* Hazem Mohamed El-Kanawati
* Youssef Hany Sayed

---

## 📌 Notes

* Deeper models did not always improve performance
* Model complexity must match dataset size

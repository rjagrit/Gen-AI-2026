# 🤖 Machine Learning Fundamentals — Complete Guide for Gen AI

> **Why ML?** Gen AI is built on top of ML principles. Understanding how models learn, what loss functions are, and how optimization works is non-negotiable for understanding LLMs.

---

## 1. What is Machine Learning?

Machine Learning is a subfield of AI where systems **learn patterns from data** rather than being explicitly programmed.

```
Traditional Programming:  Rules + Data  → Output
Machine Learning:         Data + Output → Rules (learned)
```

### Three Types of ML

| Type | How It Learns | Examples |
|---|---|---|
| **Supervised** | From labeled data (X → Y) | Spam detection, image classification |
| **Unsupervised** | From unlabeled data | Clustering, compression |
| **Reinforcement** | From rewards/penalties | Game playing, RLHF in LLMs |
| **Self-Supervised** | Labels derived from data itself | GPT (predict next word), BERT (predict masked word) |

> 🔑 **Self-supervised learning** is what powers modern LLMs — the model learns from raw text without human-labeled data.

---

## 2. Supervised Learning

The model maps inputs **X** to outputs **Y** using labeled training data.

### Regression — Predicting Continuous Values

Predict a number, like temperature or house price.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])   # features
y = np.array([2, 4, 6, 8])           # labels

model = LinearRegression()
model.fit(X, y)
model.predict([[5]])   # → [10.]
```

**How Linear Regression works:**
```
y = w₁x₁ + w₂x₂ + ... + b
```
- `w` = weights (learned)
- `b` = bias (learned)
- Goal: find weights that minimize prediction error

### Classification — Predicting Categories

Predict a class label (spam/not spam, cat/dog).

```python
from sklearn.linear_model import LogisticRegression

# Logistic Regression for binary classification
# Uses sigmoid function to output probability between 0 and 1
# sigmoid(z) = 1 / (1 + e^(-z))

model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.predict_proba(X_test)   # probability for each class
```

### Key Supervised Algorithms

#### Decision Trees
```
                 [Age > 30?]
                /           \
           Yes /             \ No
        [Income > 50k?]    [Student?]
         /        \          /     \
       Loan      No       Loan     No
```
- Splits data on feature values
- Easy to interpret
- Prone to overfitting on complex data

#### Random Forest
- Builds **many** decision trees on random subsets of data
- Aggregates their votes → more robust than a single tree
- Reduces overfitting significantly

#### Support Vector Machine (SVM)
- Finds the **maximum margin hyperplane** between classes
- Very effective on small datasets
- Kernel trick for non-linear boundaries

---

## 3. Unsupervised Learning

No labels. The model finds hidden structure in data.

### K-Means Clustering
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans.labels_       # cluster assignments per sample
kmeans.cluster_centers_   # centroid of each cluster
```

**Algorithm:**
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Recompute centroids
4. Repeat until convergence

### PCA — Principal Component Analysis

Reduces high-dimensional data to fewer dimensions while preserving maximum variance.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)
print(pca.explained_variance_ratio_)   # % variance kept per component
```

> Used in NLP to visualize word embeddings in 2D/3D.

---

## 4. The Math Behind ML

### Linear Algebra Essentials

```python
import numpy as np

# Vectors
v = np.array([1, 2, 3])

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication — the most important operation in ML/DL
C = A @ B           # np.dot(A, B)

# Transpose
A.T                 # swap rows and columns

# Inverse
np.linalg.inv(A)

# Dot product (similarity)
np.dot(v, v)        # sum of squares = magnitude²
```

**Why this matters:** Every forward pass in a neural network is **matrix multiplication** (linear algebra).

### Calculus — Gradients

A **gradient** tells us the direction and rate of change of a function.

```
f(x) = x²
f'(x) = 2x     ← derivative

For multivariable functions:
∇f = [∂f/∂w₁, ∂f/∂w₂, ...]   ← gradient vector
```

We use gradients to know **which direction to move weights** to reduce loss.

### Probability Basics

```python
# Probability distributions
import numpy as np

# Bernoulli: P(X=1) = p, P(X=0) = 1-p
# Normal: bell curve — N(mean, std²)
samples = np.random.normal(loc=0, scale=1, size=1000)

# Bayes Theorem
# P(A|B) = P(B|A) * P(A) / P(B)
# "Posterior = Likelihood × Prior / Evidence"
```

---

## 5. The ML Training Pipeline

```
Raw Data
   ↓
Data Preprocessing (clean, encode, scale)
   ↓
Train / Val / Test Split
   ↓
Choose Model Architecture
   ↓
Define Loss Function
   ↓
Optimization Loop:
   Forward Pass → Compute Loss → Backprop → Update Weights
   ↓
Evaluate on Validation Set
   ↓
Final Evaluation on Test Set
```

### Train / Validation / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Typical ratios: 70% train, 15% val, 15% test
```

| Split | Purpose |
|---|---|
| **Train** | Model learns from this |
| **Validation** | Tune hyperparameters, monitor training |
| **Test** | Final unbiased evaluation — touch ONCE |

---

## 6. Loss Functions

A loss function measures **how wrong** the model's predictions are. Lower = better.

### For Regression

```python
# Mean Squared Error (MSE)
# L = (1/n) * Σ(y_pred - y_true)²
# Penalizes large errors heavily

# Mean Absolute Error (MAE)
# L = (1/n) * Σ|y_pred - y_true|
# More robust to outliers
```

### For Classification

```python
# Cross-Entropy Loss (most common for classification)
# L = -Σ y_true * log(y_pred)

# Binary Cross-Entropy (2 classes)
# L = -(y*log(p) + (1-y)*log(1-p))

import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCELoss()        # binary
loss_fn = nn.MSELoss()        # regression
```

> 🔑 LLMs use **cross-entropy loss** — they predict the probability distribution over the vocabulary, and the loss penalizes wrong predictions.

---

## 7. Optimization — How Models Learn

### Gradient Descent

```
w_new = w_old - learning_rate * gradient
```

We move weights in the **opposite direction** of the gradient to reduce loss.

```python
learning_rate = 0.01

# Gradient Descent variants:

# 1. Batch GD — use ALL training data per update (slow, stable)
# 2. Stochastic GD — use ONE sample per update (fast, noisy)
# 3. Mini-Batch GD — use a BATCH of samples (best of both worlds)
```

### Adam Optimizer (Most Common in Gen AI)

Adam = Adaptive Moment Estimation. Automatically adjusts learning rate per parameter.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
# AdamW — Adam with weight decay, standard for LLM training
```

### Learning Rate

```
Too high LR → loss oscillates, model diverges
Too low LR  → learning is very slow
Just right  → smooth convergence
```

**Learning Rate Scheduling:**
```python
# Warmup then decay — standard for LLMs
# Start with low LR → gradually increase → then decrease
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

---

## 8. Overfitting & Underfitting

```
Underfitting: Model is too simple — high bias
              Training loss high, Val loss high

Overfitting:  Model memorizes training data — high variance
              Training loss low, Val loss HIGH

Just right:   Both training and val loss are low
```

### How to Fix Overfitting

| Technique | Description |
|---|---|
| More training data | Biggest fix |
| Regularization (L1/L2) | Penalize large weights |
| Dropout | Randomly zero neurons |
| Early stopping | Stop when val loss increases |
| Data augmentation | Artificially expand dataset |
| Simpler model | Reduce capacity |

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)   # 5-fold CV
print(scores.mean(), scores.std())
```

Splits data into 5 folds, trains on 4, tests on 1, rotates — gives robust performance estimate.

---

## 9. Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Accuracy   = Correct predictions / Total predictions
# Precision  = TP / (TP + FP)  → "of positives predicted, how many are right?"
# Recall     = TP / (TP + FN)  → "of actual positives, how many did we catch?"
# F1-Score   = 2 * (P * R) / (P + R)  → harmonic mean of precision & recall
```

**Confusion Matrix:**
```
                 Predicted
                Pos    Neg
Actual  Pos  [ TP  |  FN ]
        Neg  [ FP  |  TN ]
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)   # 1.0 = perfect fit
```

---

## 10. Feature Engineering & Preprocessing

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
)

# Standardize: (x - mean) / std  → mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Min-Max: (x - min) / (max - min) → [0, 1] range
minmax = MinMaxScaler()
X_norm = minmax.fit_transform(X)

# Encode categorical variables
le = LabelEncoder()
y_encoded = le.fit_transform(["cat", "dog", "cat"])   # [0, 1, 0]
```

---

## 11. The Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias     → Underfitting (model too simple, misses patterns)
High Variance → Overfitting  (model too complex, memorizes noise)
```

The goal is to find a model complexity where **both** are reasonably low.

---

## ✅ ML Concepts Checklist for Gen AI

| Concept | Why It Matters for Gen AI |
|---|---|
| Supervised / Self-supervised learning | LLMs use self-supervised pretraining |
| Cross-entropy loss | Training objective for all LLMs |
| Gradient descent & Adam | How LLM weights are updated |
| Overfitting / regularization | Preventing LLM memorization |
| Train/val/test splits | Evaluating model generalization |
| Evaluation metrics | Perplexity = LM version of cross-entropy |
| Feature scaling | Input normalization in transformers |
| Probability & softmax | Token probability distributions in LLMs |

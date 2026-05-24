# 🧠 Deep Learning Concepts — Complete Guide for Gen AI

> **Why Deep Learning?** Every Gen AI model — GPT, Claude, Stable Diffusion — is a deep neural network. Understanding how neural networks are built, trained, and regularized is essential.

---

## 1. What is a Neural Network?

A neural network is a system of **interconnected layers of neurons** that learn to map inputs to outputs through training.

```
Input Layer → [Hidden Layer 1] → [Hidden Layer 2] → Output Layer
```

### The Neuron (Perceptron)

```
Inputs: x₁, x₂, x₃
Weights: w₁, w₂, w₃
Bias: b

z = w₁x₁ + w₂x₂ + w₃x₃ + b   ← weighted sum
output = activation(z)           ← pass through activation function
```

In matrix form: `z = Wx + b` where W is the weight matrix and x is input.

```python
import torch
import torch.nn as nn

# A single linear (dense) layer
layer = nn.Linear(in_features=784, out_features=256)
# Applies z = Wx + b automatically
```

---

## 2. Activation Functions

Activation functions introduce **non-linearity** — without them, stacking layers is pointless (they collapse to one linear transformation).

### ReLU — Rectified Linear Unit (Most Common)
```
f(x) = max(0, x)

Negative values → 0
Positive values → unchanged

Pros: Simple, fast, no vanishing gradient for positive values
Cons: "Dying ReLU" — neurons that output 0 can stop learning
```

### Sigmoid
```
f(x) = 1 / (1 + e^(-x))     → Range: (0, 1)

Used for: Binary classification output layer
Cons: Vanishing gradient for very large/small x
```

### Softmax
```
f(xᵢ) = e^xᵢ / Σ e^xⱼ      → Range: (0, 1), sums to 1

Converts raw scores to probability distribution
Used for: Multi-class classification, LLM token prediction
```

### GELU — Gaussian Error Linear Unit
```
f(x) ≈ x * Φ(x)   where Φ is the Gaussian CDF

Smooth version of ReLU
Used in: GPT, BERT, and most modern Transformers
```

### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))   → Range: (-1, 1)

Zero-centered (unlike sigmoid)
Used in: RNNs
```

```python
import torch.nn.functional as F

F.relu(x)
F.sigmoid(x)
F.softmax(x, dim=-1)
F.gelu(x)
F.tanh(x)
```

---

## 3. Feedforward Neural Networks (FNN)

A stack of linear layers with activations between them.

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

model = FeedForward(784, 256, 10)   # e.g., MNIST digit classifier
```

### Forward Pass (Inference)
```
x → Layer1 → ReLU → Layer2 → ReLU → Layer3 → Output
```

---

## 4. Backpropagation — How Models Learn

Backprop uses the **chain rule of calculus** to compute how much each weight contributed to the loss.

```
Forward Pass:
  x → model → ŷ (prediction)

Compute Loss:
  L = loss(ŷ, y_true)

Backward Pass (backprop):
  ∂L/∂w  =  ∂L/∂ŷ  ×  ∂ŷ/∂z  ×  ∂z/∂w     ← chain rule
             (loss)   (activation)  (linear)

Weight Update:
  w = w - lr × ∂L/∂w
```

```python
# PyTorch automates this with autograd
optimizer.zero_grad()       # reset gradients from last step
loss = criterion(output, target)
loss.backward()             # compute all gradients (backprop)
optimizer.step()            # update weights using gradients
```

---

## 5. PyTorch Training Loop (Standard Template)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. Define model
model = MyModel().to(device)

# 2. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 3. Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(batch_x)       # forward pass
        loss = criterion(outputs, batch_y)  # compute loss
        loss.backward()                # backward pass
        optimizer.step()               # update weights

    # Validation
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_out = model(val_x.to(device))
            val_loss = criterion(val_out, val_y.to(device))

    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
```

---

## 6. Convolutional Neural Networks (CNNs)

CNNs are specialized for **spatial data** (images, sometimes text).

### Convolution Operation
```
A filter (kernel) slides across the input, computing dot products.

Input:                Filter (3×3):       Output (Feature Map):
[1, 2, 3, 4]         [1, 0, -1]
[5, 6, 7, 8]    →    [1, 0, -1]    →    [edge detections]
[9, 0, 1, 2]         [1, 0, -1]
```

### Key CNN Components

```python
# Convolution layer
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

# Pooling — downsampling
nn.MaxPool2d(kernel_size=2, stride=2)   # takes max in 2×2 window
nn.AvgPool2d(kernel_size=2)             # takes average

# Full CNN example
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 28×28 → 14×14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 14×14 → 7×7
        )
        self.classifier = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)     # flatten spatial dims
        return self.classifier(x)
```

### Famous CNN Architectures
| Model | Year | Contribution |
|---|---|---|
| LeNet | 1998 | First practical CNN |
| AlexNet | 2012 | Deep CNN won ImageNet |
| VGG | 2014 | Very deep, simple 3×3 conv |
| ResNet | 2015 | Residual connections — enabled 100+ layers |
| EfficientNet | 2019 | Scaled width/depth/resolution together |

---

## 7. Recurrent Neural Networks (RNNs)

RNNs process **sequences** by maintaining a hidden state across time steps.

```
Input: "The cat sat"
        ↓    ↓    ↓
       h₁ → h₂ → h₃ → output
        ↑    ↑    ↑
       x₁   x₂   x₃

hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
```

### The Vanishing Gradient Problem
```
Gradient at time step 1 = product of many Jacobians
If each < 1, after 100 steps: 0.9^100 ≈ 0.000027 → vanishes
Gradient essentially becomes 0 — network can't learn long dependencies
```

### LSTM — Long Short-Term Memory

Solves vanishing gradients with **gates** that control information flow.

```
Forget Gate:  f = σ(Wf·[hₜ₋₁, xₜ] + bf)    → what to forget
Input Gate:   i = σ(Wi·[hₜ₋₁, xₜ] + bi)    → what new info to add
Cell Update:  C̃ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
Cell State:   Cₜ = f⊙Cₜ₋₁ + i⊙C̃           → updated memory
Output Gate:  o = σ(Wo·[hₜ₋₁, xₜ] + bo)
Hidden State: hₜ = o⊙tanh(Cₜ)
```

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
gru  = nn.GRU(input_size=128, hidden_size=256, batch_first=True)  # simpler LSTM
```

> **Note:** Transformers replaced RNNs for NLP because they process sequences in parallel and handle long-range dependencies better.

---

## 8. Regularization Techniques

### Dropout
Randomly **zeroes out neurons** during training — prevents co-adaptation.

```python
nn.Dropout(p=0.5)   # 50% of neurons zeroed randomly per forward pass
# During inference: no dropout applied (model.eval())
```

```
Training: [a, b, c, d] → [a, 0, c, 0]  (random mask)
Testing:  [a, b, c, d]  (full network, but scaled by 1-p)
```

### Batch Normalization
Normalizes activations within each mini-batch — speeds training, allows higher LR.

```python
nn.BatchNorm2d(num_features=32)    # for CNN feature maps
nn.BatchNorm1d(num_features=256)   # for dense layers

# Normalizes: x̂ = (x - μ_batch) / σ_batch
# Then scales: y = γx̂ + β   (γ, β are learnable)
```

### Layer Normalization
Normalizes across features (not batch) — standard in **Transformers**.

```python
nn.LayerNorm(normalized_shape=512)
# Normalizes across feature dimension — works for any batch size
# Used in: BERT, GPT, T5, Claude — all transformer-based models
```

### L1 / L2 Regularization (Weight Decay)

```python
# L2 via weight_decay in optimizer (most common)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Adds penalty to loss:
# L_total = L_task + λ * Σ w²   (L2)
# L_total = L_task + λ * Σ |w|  (L1)
```

---

## 9. Residual Connections (Skip Connections)

Invented in **ResNet**, now used in every Transformer.

```
Without residual:     x → Layer → y
With residual:        x → Layer → y + x   ← add input back

Why it helps:
- Gradient flows directly through skip connection
- Enables training of very deep networks (100+ layers)
- "At minimum, the layer can learn identity"
```

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.layer(self.norm(x))   # x + F(x)
```

---

## 10. GPU Training with PyTorch

```python
# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model = model.to(device)

# Move tensors to GPU
x = x.to(device)
y = y.to(device)

# Mixed Precision Training (faster, less memory)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 11. PyTorch Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

dataset = TextDataset(texts, labels)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,        # shuffle for training
    num_workers=4        # parallel data loading
)
```

---

## 12. Key Deep Learning Hyperparameters

| Hyperparameter | Description | Typical Range |
|---|---|---|
| Learning Rate | Step size for weight updates | 1e-5 to 1e-2 |
| Batch Size | Samples per gradient update | 8 to 512 |
| Epochs | Full passes over training data | 3 to 100 |
| Hidden Dim | Width of hidden layers | 128 to 4096 |
| Num Layers | Depth of network | 2 to 96 |
| Dropout | Regularization probability | 0.1 to 0.5 |
| Weight Decay | L2 regularization strength | 0.01 to 0.1 |

---

## 13. Deep Learning Frameworks Comparison

| Feature | PyTorch | TensorFlow/Keras |
|---|---|---|
| Popularity in research | ⭐⭐⭐ Dominant | ⭐⭐ |
| Ease of use | High (Pythonic) | High (Keras API) |
| Dynamic graphs | ✅ Default | ✅ TF2 Eager |
| Production deployment | ✅ TorchScript/ONNX | ✅ TF Serving |
| Gen AI ecosystem | ✅ PyTorch-native | ✅ JAX/Flax |

> **Recommendation:** Learn PyTorch. HuggingFace Transformers, LLaMA, and nearly all cutting-edge Gen AI research uses PyTorch.

---

## ✅ Deep Learning Checklist for Gen AI

| Concept | Why It Matters for Gen AI |
|---|---|
| Neural network layers | All LLMs are stacked layers |
| Activation functions (GELU, Softmax) | Used in Transformer blocks & output |
| Backpropagation & gradients | Core training mechanism of LLMs |
| PyTorch training loop | Need this to fine-tune or train models |
| Layer Normalization | Every Transformer uses this |
| Residual connections | Every Transformer uses this |
| Dropout | Regularization during LLM training |
| DataLoader | Loading tokenized text batches |
| GPU / Mixed precision | Required to train LLMs efficiently |

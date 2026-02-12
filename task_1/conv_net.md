# Convolutional Neural Network (CNN)

## Introduction

A Convolutional Neural Network (CNN) is a deep learning architecture primarily used for processing structured grid-like data such as images. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input data. Unlike traditional neural networks, CNNs reduce the number of parameters by using shared weights and local connectivity.

CNNs are widely used in computer vision, but they are also applied in cybersecurity tasks such as malware detection, intrusion detection, and anomaly detection.

---

## CNN Architecture

A typical CNN consists of the following layers:

### 1. Convolutional Layer

This layer applies filters (kernels) to the input image. Each filter extracts specific features such as edges, textures, or patterns.

Mathematically:

Output = (Input * Kernel) + Bias

The filters slide across the image and generate feature maps.

### 2. Activation Function

Usually ReLU (Rectified Linear Unit):

f(x) = max(0, x)

It introduces non-linearity into the model.

### 3. Pooling Layer

Pooling reduces spatial dimensions and helps control overfitting.

Common type:
- Max Pooling

### 4. Fully Connected Layer

After convolution and pooling, the feature maps are flattened and passed into fully connected layers for classification.

---

## CNN Workflow

Input Image → Convolution → ReLU → Pooling → Flatten → Fully Connected → Output

---

## CNN Architecture Visualization

Below is a simplified representation of a CNN structure:

[Input Image]
       |
[Convolution Layer]
       |
[ReLU Activation]
       |
[Pooling Layer]
       |
[Flatten]
       |
[Fully Connected Layer]
       |
[Output Class]

This diagram illustrates the flow of data through the CNN architecture.



---

## Practical Example: Malware Image Classification

In cybersecurity, CNNs can be used to detect malware by converting binary files into grayscale images. Each byte of a file is interpreted as a pixel value (0–255). Malware files often have structural patterns that CNNs can learn to recognize.

In this example, we simulate malware and benign samples using randomly generated image data.

---

## Dataset Description

We generate:
- 100 benign samples
- 100 malware samples
- Image size: 32x32 grayscale

Label:
- 0 = Benign
- 1 = Malware

---

## Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)

benign = np.random.rand(100, 1, 32, 32)
malware = np.random.rand(100, 1, 32, 32) + 0.5  # shift distribution

X = np.vstack((benign, malware))
y = np.array([0]*100 + [1]*100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*15*15, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8*15*15)
        x = self.fc1(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
losses = []

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()







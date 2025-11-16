# Path: main.py

import numpy as np
import matplotlib.pyplot as plt
from models.ann_numpy import ANN
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load data
X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")

# Model
model = ANN(layer_dims=[12, 32, 16, 1], learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean() * 100
print(f"Test Accuracy: {acc:.2f}%")

# Plot loss
plt.figure()
plt.plot(range(0, len(model.losses) * 100, 100), model.losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_plot.png")
print("Saved: outputs/loss_plot.png")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import os

# Load MNIST dataset from OpenML
mnist = fetch_openml("mnist_784", version=1)
X = mnist["data"].values / 255.0  # Normalize the images (0-1 range)
y = mnist["target"].astype(int).values

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Split into testing set
X_test, y_test = X, y_one_hot  # We are only testing in this script

# Load the pre-trained model weights
model_file = 'trained_models/model_weights_improved_v7.npz'
if os.path.exists(model_file):
    print("Loading pre-trained model weights...")
    model = np.load(model_file)
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']
    W4, b4 = model['W4'], model['b4']
else:
    raise ValueError("Model weights file not found. Please ensure the model is trained first.")

# Activation function and its derivative (ReLU and softmax for output)
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)  # Leaky ReLU

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Testing the model
hidden_input_1_test = np.dot(X_test, W1) + b1
hidden_output_1_test = leaky_relu(hidden_input_1_test)

hidden_input_2_test = np.dot(hidden_output_1_test, W2) + b2
hidden_output_2_test = leaky_relu(hidden_input_2_test)

hidden_input_3_test = np.dot(hidden_output_2_test, W3) + b3
hidden_output_3_test = leaky_relu(hidden_input_3_test)

final_input_test = np.dot(hidden_output_3_test, W4) + b4
y_pred_test = softmax(final_input_test)

# Convert predictions to class labels
y_pred_class = np.argmax(y_pred_test, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred_class == y_test_class)
print(f"Test Accuracy: {accuracy * 100:.4f}%")

# Visualizing some predictions
fig, axes = plt.subplots(1, 5, figsize=(12, 5))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred_class[i]} / True: {y_test_class[i]}")
    ax.axis("off")

plt.show()

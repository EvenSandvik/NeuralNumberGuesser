import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

# Load MNIST dataset from OpenML
mnist = fetch_openml("mnist_784", version=1)
X = mnist["data"].values / 255.0  # Normalize the images (0-1 range)
y = mnist["target"].astype(int).values

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Initialize weights and biases using Xavier initialization
input_size = 784  # 28x28 pixels
hidden_size_1 = 128
hidden_size_2 = 64
hidden_size_3 = 32  # Additional hidden layer for increased depth
output_size = 10  # 10 digits (0-9)

# Xavier initialization (Glorot initialization)
def xavier_initialization(size_in, size_out):
    return np.random.randn(size_in, size_out) * np.sqrt(2. / (size_in + size_out))

# Initialize weights and biases
W1 = xavier_initialization(input_size, hidden_size_1)
b1 = np.zeros(hidden_size_1)
W2 = xavier_initialization(hidden_size_1, hidden_size_2)
b2 = np.zeros(hidden_size_2)
W3 = xavier_initialization(hidden_size_2, hidden_size_3)
b3 = np.zeros(hidden_size_3)
W4 = xavier_initialization(hidden_size_3, output_size)
b4 = np.zeros(output_size)

# Activation function and its derivative (ReLU and softmax for output)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function (Cross-Entropy)
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Adding small epsilon for numerical stability

# Training parameters
learning_rate = 0.01  # Lower learning rate for better stability
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input_1 = np.dot(X_train, W1) + b1
    hidden_output_1 = relu(hidden_input_1)

    hidden_input_2 = np.dot(hidden_output_1, W2) + b2
    hidden_output_2 = relu(hidden_input_2)

    hidden_input_3 = np.dot(hidden_output_2, W3) + b3
    hidden_output_3 = relu(hidden_input_3)

    final_input = np.dot(hidden_output_3, W4) + b4
    y_pred = softmax(final_input)

    # Loss calculation (Cross-Entropy)
    loss = cross_entropy_loss(y_pred, y_train)

    # Backpropagation
    output_error = y_pred - y_train
    output_delta = output_error  # Since softmax + cross-entropy combined simplifies the gradient

    hidden_error_3 = np.dot(output_delta, W4.T)
    hidden_delta_3 = hidden_error_3 * relu_derivative(hidden_output_3)

    hidden_error_2 = np.dot(hidden_delta_3, W3.T)
    hidden_delta_2 = hidden_error_2 * relu_derivative(hidden_output_2)

    hidden_error_1 = np.dot(hidden_delta_2, W2.T)
    hidden_delta_1 = hidden_error_1 * relu_derivative(hidden_output_1)

    # Update weights and biases using gradient descent
    W4 -= learning_rate * np.dot(hidden_output_3.T, output_delta)
    b4 -= learning_rate * np.sum(output_delta, axis=0)

    W3 -= learning_rate * np.dot(hidden_output_2.T, hidden_delta_3)
    b3 -= learning_rate * np.sum(hidden_delta_3, axis=0)

    W2 -= learning_rate * np.dot(hidden_output_1.T, hidden_delta_2)
    b2 -= learning_rate * np.sum(hidden_delta_2, axis=0)

    W1 -= learning_rate * np.dot(X_train.T, hidden_delta_1)
    b1 -= learning_rate * np.sum(hidden_delta_1, axis=0)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Save the trained model weights
np.savez('model_weights_improved_v2.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)

# Testing the model
hidden_input_1_test = np.dot(X_test, W1) + b1
hidden_output_1_test = relu(hidden_input_1_test)

hidden_input_2_test = np.dot(hidden_output_1_test, W2) + b2
hidden_output_2_test = relu(hidden_input_2_test)

hidden_input_3_test = np.dot(hidden_output_2_test, W3) + b3
hidden_output_3_test = relu(hidden_input_3_test)

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

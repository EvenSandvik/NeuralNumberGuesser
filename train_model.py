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
reg_lambda = 0.01

# Xavier initialization (Glorot initialization)
def xavier_initialization(size_in, size_out):
    return np.random.randn(size_in, size_out) * np.sqrt(2. / (size_in + size_out))

# Check if model weights file exists
model_file = 'trained_models/model_weights_improved_v4.npz'
if os.path.exists(model_file):
    print("Loading pre-trained model weights...")
    # Load the saved model weights
    model = np.load(model_file)
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']
    W4, b4 = model['W4'], model['b4']
else:
    # If no model, initialize weights randomly
    print("Training new model...")
    W1 = xavier_initialization(input_size, hidden_size_1)
    b1 = np.zeros(hidden_size_1)
    W2 = xavier_initialization(hidden_size_1, hidden_size_2)
    b2 = np.zeros(hidden_size_2)
    W3 = xavier_initialization(hidden_size_2, hidden_size_3)
    b3 = np.zeros(hidden_size_3)
    W4 = xavier_initialization(hidden_size_3, output_size)
    b4 = np.zeros(output_size)

# Activation function and its derivative (ReLU and softmax for output)
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)  # Leaky ReLU

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)  # Derivative of Leaky ReLU

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function (Cross-Entropy with L2 Regularization)
def cross_entropy_loss(y_pred, y_true, reg_lambda=0.01):
    m = y_true.shape[0]
    log_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Adding small epsilon for numerical stability
    # L2 regularization on weights
    l2_loss = (reg_lambda / 2) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    return log_loss + l2_loss

# Mini-batch Gradient Descent parameters
batch_size = 64
learning_rate = 0.00025  # Lower learning rate for better stability
epochs = 10000
gradient_clip_value = 5  # Gradient clipping threshold

# To store the loss for visualization
losses = []

# Training loop with mini-batch gradient descent

for epoch in range(epochs):
    # Shuffle data at the start of each epoch
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Mini-batch gradient descent
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        # Forward pass
        hidden_input_1 = np.dot(X_batch, W1) + b1
        hidden_output_1 = leaky_relu(hidden_input_1)

        hidden_input_2 = np.dot(hidden_output_1, W2) + b2
        hidden_output_2 = leaky_relu(hidden_input_2)

        hidden_input_3 = np.dot(hidden_output_2, W3) + b3
        hidden_output_3 = leaky_relu(hidden_input_3)

        final_input = np.dot(hidden_output_3, W4) + b4
        y_pred = softmax(final_input)

        # Loss calculation (Cross-Entropy + L2 Regularization)
        loss = cross_entropy_loss(y_pred, y_batch)
        losses.append(loss)

        # Backpropagation
        output_error = y_pred - y_batch
        output_delta = output_error  # Since softmax + cross-entropy combined simplifies the gradient

        hidden_error_3 = np.dot(output_delta, W4.T)
        hidden_delta_3 = hidden_error_3 * leaky_relu_derivative(hidden_output_3)

        hidden_error_2 = np.dot(hidden_delta_3, W3.T)
        hidden_delta_2 = hidden_error_2 * leaky_relu_derivative(hidden_output_2)

        hidden_error_1 = np.dot(hidden_delta_2, W2.T)
        hidden_delta_1 = hidden_error_1 * leaky_relu_derivative(hidden_output_1)

        # Gradient clipping
        for weight in [W4, W3, W2, W1]:
            np.clip(weight, -gradient_clip_value, gradient_clip_value, out=weight)

        # Update weights and biases using gradient descent with L2 regularization
        W4 -= learning_rate * (np.dot(hidden_output_3.T, output_delta) + reg_lambda * W4)
        b4 -= learning_rate * np.sum(output_delta, axis=0)

        W3 -= learning_rate * (np.dot(hidden_output_2.T, hidden_delta_3) + reg_lambda * W3)
        b3 -= learning_rate * np.sum(hidden_delta_3, axis=0)

        W2 -= learning_rate * (np.dot(hidden_output_1.T, hidden_delta_2) + reg_lambda * W2)
        b2 -= learning_rate * np.sum(hidden_delta_2, axis=0)

        W1 -= learning_rate * (np.dot(X_batch.T, hidden_delta_1) + reg_lambda * W1)
        b1 -= learning_rate * np.sum(hidden_delta_1, axis=0)

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Save the trained model weights
np.savez('trained_models/model_weights_improved_v4.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)

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

# Plot the loss curve to visualize training progress
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

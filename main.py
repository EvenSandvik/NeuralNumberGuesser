import numpy as np

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple rule: sum > 1 -> class 1

# Initialize weights and biases
input_size = 2
hidden_size = 3
output_size = 1

W1 = np.random.rand(input_size, hidden_size)  # Weights for input to hidden
b1 = np.random.rand(hidden_size)             # Bias for hidden layer
W2 = np.random.rand(hidden_size, output_size)  # Weights for hidden to output
b2 = np.random.rand(output_size)             # Bias for output layer

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training parameters
learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1  # Input to hidden layer
    hidden_output = sigmoid(hidden_input)  # Activation of hidden layer
    
    final_input = np.dot(hidden_output, W2) + b2  # Input to output layer
    y_pred = sigmoid(final_input)  # Activation of output layer

    # Loss calculation (Mean Squared Error)
    loss = np.mean((y_pred - y.reshape(-1, 1)) ** 2)

    # Backpropagation
    output_error = y_pred - y.reshape(-1, 1)
    output_delta = output_error * sigmoid_derivative(y_pred)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 -= learning_rate * np.dot(hidden_output.T, output_delta)
    b2 -= learning_rate * np.sum(output_delta, axis=0)

    W1 -= learning_rate * np.dot(X.T, hidden_delta)
    b1 -= learning_rate * np.sum(hidden_delta, axis=0)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Predictions
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = np.mean(y_pred_class.flatten() == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

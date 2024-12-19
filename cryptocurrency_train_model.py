import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Load cryptocurrency historical data (price, volume, etc.)
# For simplicity, we will use a sample CSV file. You should replace it with real data.
# The data should include columns like: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Feature engineering
def add_technical_indicators(data):
    # Simple Moving Averages (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Exponential Moving Averages (EMA)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with NaN values (e.g., first few rows due to rolling windows)
    data.dropna(inplace=True)

    return data

# Generate target (labels)
def generate_target(data):
    # Predict the price movement: 1 for up, 0 for down
    data['Price_Movement'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    return data

# Load and prepare data
file_path = 'cryptocurrency_data.csv'  # Replace with your dataset path
data = load_data(file_path)

# Add technical indicators and target
data = add_technical_indicators(data)
data = generate_target(data)

# Features (inputs) and target (output)
features = ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal']
X = data[features].values
y = data['Price_Movement'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.4f}%")

# Save the trained model
model_file = 'crypto_trading_model.pkl'
if not os.path.exists(model_file):
    import joblib
    joblib.dump(model, model_file)
    print(f"Model saved as {model_file}")

# Visualize the results (example: show predicted vs actual movement)
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Price Movement', color='blue')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Price Movement', color='red', linestyle='--')
plt.legend()
plt.title("Actual vs Predicted Price Movement")
plt.xlabel("Date")
plt.ylabel("Price Movement")
plt.show()


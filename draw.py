import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Load the trained model weights
model = np.load('trained_models/model_weights_improved_v7.npz')
W1, b1 = model['W1'], model['b1']
W2, b2 = model['W2'], model['b2']
W3, b3 = model['W3'], model['b3']
W4, b4 = model['W4'], model['b4']

# Activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Prediction function
def predict(image_array):
    image_flattened = image_array.flatten().reshape(1, -1) / 255.0
    h1 = leaky_relu(np.dot(image_flattened, W1) + b1)
    h2 = leaky_relu(np.dot(h1, W2) + b2)
    h3 = leaky_relu(np.dot(h2, W3) + b3)
    output = softmax(np.dot(h3, W4) + b4)
    print(f"h1: {h1}\nh2: {h2}\nh3: {h3}\noutput: {output}")
    return np.argmax(output, axis=1)[0]

# Drawing interface
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas_size = 480  # Canvas size in pixels
        self.cell_size = self.canvas_size // 28  # Size of each cell
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left")

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side="right")

        # Initialize the image object and a drawing context
        self.image = Image.new("L", (28, 28), color=0)
        self.draw_context = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        brush_size = 3  # Adjust the brush size
        for dx in range(-brush_size, brush_size + 1):
            for dy in range(-brush_size, brush_size + 1):
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if distance <= brush_size:
                    intensity = int(255 * (1 - distance / brush_size))
                    px, py = (x + dx) // self.cell_size, (y + dy) // self.cell_size
                    if 0 <= px < 28 and 0 <= py < 28:
                        existing = self.image.getpixel((px, py))
                        self.image.putpixel((px, py), min(255, existing + intensity))

        # Update the canvas visually
        x0 = x // self.cell_size * self.cell_size
        y0 = y // self.cell_size * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        self.canvas.create_oval(
            x - brush_size, y - brush_size, x + brush_size, y + brush_size,
            fill="black", outline="black"
        )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw_context = ImageDraw.Draw(self.image)

    def predict_digit(self):
        # Normalize the image and invert colors
        image_array = np.array(self.image)
        print(f"Input Image Array (flattened): {image_array.flatten()[:100]}")  # Show first 100 values
        prediction = predict(image_array)
        print(f"Predicted Digit: {prediction}")
        messagebox.showinfo("Prediction", f"The predicted digit is: {prediction}")

# Run the application
root = tk.Tk()
root.title("Digit Recognition")
app = DrawingApp(root)
root.mainloop()

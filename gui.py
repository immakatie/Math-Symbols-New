import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Загрузка обученной модели
model = tf.keras.models.load_model('model.keras')

# Список классов
class_labels = ["-", "!", "( )", ",", "[ ]", "{ }", "+", "=", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "alpha", "ascii_124", "b", "beta", "C", "cos", "d", "delta", "div", "e", "exists", "f", "forall", "forward_slash", "G", "gamma", "geq", "gt", "H", "i", "in", "infinity", "int", "j", "k", "l", "lambda", "ldots", "leq", "lim", "log", "lt", "M", "mu", "N", "neq", "o", "p", "phi", "pm", "prime", "q", "R", "rightarrow", "S", "sigma", "sin", "sqrt", "sum", "T", "tan", "theta", "times", "u", "v", "w", "X", "y", "z"]

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")
        self.root.geometry("300x300")

        self.canvas = tk.Canvas(self.root, bg='white', width=280, height=280)
        self.canvas.pack()

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.clear_on_right_click)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.predicted_class_label = tk.Label(self.root, text="")
        self.predicted_class_label.pack()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.predicted_class_label.config(text="")

    def clear_on_right_click(self, event):
        self.clear_canvas()

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def predict(self):
        # Преобразование изображения
        self.image = self.image.resize((28, 28))
        self.image = ImageOps.invert(self.image)
        image_array = np.array(self.image).astype('float32') / 255
        image_array = image_array.reshape(1, 28, 28, 1)

        # Предсказание
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Показ результата
        self.predicted_class_label.config(text=f"Predicted: {class_labels[predicted_class]}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

input("Нажмите Enter для выхода...")
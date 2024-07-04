





'''
import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import json

class DrawingApp:
    def __init__(self, root, model, label_map):
        self.root = root
        self.model = model
        self.label_map = {v: k for k, v in label_map.items()}  # Обратный словарь для отображения меток
        self.canvas = Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_predict = Button(root, text='Predict', command=self.predict)
        self.button_predict.pack()
        self.button_clear = Button(root, text='Clear', command=self.clear)
        self.button_clear.pack()
        self.image1 = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.line([x1, y1, x2, y2], fill='black', width=5)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 280, 280), fill="white")

    def predict(self):
        # Преобразование изображения для предсказания
        img = self.image1.copy()
        img = img.resize((28, 28), Image.LANCZOS)  # Использование Image.LANCZOS вместо ANTIALIAS
        img = ImageOps.invert(img)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = self.label_map[predicted_class]
        print(f"Predicted class: {predicted_label}")

# Загрузка модели и меток классов
model = tf.keras.models.load_model('model5.keras')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

root = Tk()
app = DrawingApp(root, model, label_map)
root.mainloop()
'''

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Загрузка обученной модели (обновите путь к вашей модели)
model = tf.keras.models.load_model('model3.keras')

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")
        self.root.geometry("400x400")

        self.canvas = tk.Canvas(self.root, bg='white', width=280, height=280)
        self.canvas.pack()

        self.button_clear = tk.Button(self.root, text="Очистить", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_predict = tk.Button(self.root, text="Предсказать", command=self.predict)
        self.button_predict.pack()

        self.result_label = tk.Label(self.root, text="Предсказание: ")
        self.result_label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.clear_on_right_click)  # При нажатии правой кнопки стирается

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.result_label.config(text="Предсказать: ")

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
        self.result_label.config(text=f"Предсказание: {predicted_class}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()



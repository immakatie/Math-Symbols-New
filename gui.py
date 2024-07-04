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
model = tf.keras.models.load_model('model3.keras')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

root = Tk()
app = DrawingApp(root, model, label_map)
root.mainloop()

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

        self.button_predict = Button(root, text='Предсказать', command=self.predict)
        self.button_predict.pack()

        self.button_clear = Button(root, text='Очистить', command=self.clear)
        self.button_clear.pack()

        self.prediction_label = Label(root, text='Предсказание: ')
        self.prediction_label.pack()

        self.image1 = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.line([x1, y1, x2, y2], fill='black', width=5)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 280, 280), fill="white")
        self.prediction_label.config(text='Предсказание: ')

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
        
        self.prediction_label.config(text=f'Предсказание: {predicted_label}')
        print(f"Предсказание: {predicted_label}")

# Загрузка модели и меток классов
model = tf.keras.models.load_model('model.keras')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

root = Tk()
app = DrawingApp(root, model, label_map)
root.mainloop()
'''
import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import json
import io

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

        #self.button_show_model = Button(root, text='Show Model Architecture', command=self.show_model_architecture)
        #self.button_show_model.pack()

        self.prediction_label = Label(root, text='Prediction: ')
        self.prediction_label.pack()

        self.model_text = Text(root, height=10, width=50)
        self.model_text.pack()

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
        self.prediction_label.config(text='Prediction: ')

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

        self.prediction_label.config(text=f'Prediction: {predicted_label}')
        print(f"Predicted class: {predicted_label}")

    def show_model_architecture(self):
        # Вывод архитектуры модели в текстовое поле
        with io.StringIO() as stream:
            self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
            model_summary = stream.getvalue()
        self.model_text.delete(1.0, tk.END)
        self.model_text.insert(tk.END, model_summary)

# Загрузка модели и меток классов
model = tf.keras.models.load_model('model.keras')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

root = Tk()
app = DrawingApp(root, model, label_map)
root.mainloop()
'''

input("Нажмите Enter для выхода...")
import os
import shutil
import random

# Пути к исходным данным и целевым папкам
dataset_dir = 'C:\\Users\\user\\Downloads\\data\\extracted_images'  # Замените на путь к вашему исходному датасету
train_dir = 'C:\\Users\\user\\Downloads\\data\\train'
validation_dir = 'C:\\Users\\user\\Downloads\\data\\validation'

# Создание директорий для данных
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Процент данных, которые будут использованы для проверки
validation_split = 0.2

# Функция для разделения данных
def split_data(class_name):
    class_dir = os.path.join(dataset_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    validation_class_dir = os.path.join(validation_dir, class_name)
    
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(validation_class_dir, exist_ok=True)
    
    files = os.listdir(class_dir)
    random.shuffle(files)
    
    validation_size = int(len(files) * validation_split)
    validation_files = files[:validation_size]
    train_files = files[validation_size:]
    
    for file in train_files:
        src = os.path.join(class_dir, file)
        dst = os.path.join(train_class_dir, file)
        shutil.copyfile(src, dst)
    
    for file in validation_files:
        src = os.path.join(class_dir, file)
        dst = os.path.join(validation_class_dir, file)
        shutil.copyfile(src, dst)

# Получение списка классов
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# Разделение данных для каждого класса
for class_name in classes:
    split_data(class_name)

input("Нажмите Enter для выхода...")
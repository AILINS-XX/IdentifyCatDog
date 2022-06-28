import sns
from easygui import *
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import image
from matplotlib import pyplot as plt
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import easygui as g

keys =""

def load_file():


    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    keys = file_path
    return  keys
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])
model = create_model()



if msgbox("欢迎使用猫狗识别功能，请选择下方按钮选择识别的图片!",  title='猫狗识别',ok_button="选择图片!",image="face.jpg"):

    keys = load_file()
    model.load_weights('./save_weights/my_save_weights')
    strlist = keys[keys.find("cats_and_dogs_filtered/"):]
    #print(strlist) 图片路径
    img = image.load_img(keys, target_size=(150, 150))
    x = image.img_to_array(img)
    # 在第0维添加维度变为1x150x150x3，模型的输入数据一样
    x = np.expand_dims(x, axis=0)
    # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    #print(classes[0])

    if classes[0] > 0:
        #print(keys + " is a dog")
        g.msgbox("识别成功这是一只 狗",  title='dog')

    else:
        #print(keys + " is a cat")
        g.msgbox("识别成功这是一只 猫 ",title='cat');



# === 混淆矩阵：真实值与预测值的对比 ===
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#con_mat = confusion_matrix(y_test, y_pred)

#con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
#con_mat_norm = np.around(con_mat_norm, decimals=2)

# === plot ===
#figure = plt.figure(figsize=(8, 8))
#sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

#plt.ylim(0, 10)
#plt.xlabel('Predicted labels')
#plt.ylabel('True labels')
#plt.show()


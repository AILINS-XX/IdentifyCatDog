import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import image


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

"""

数据准备
    将图像格式化成经过适当预处理的浮点张量，然后输入网络:
    - 从磁盘读取图像。
    - 解码这些图像的内容，并根据它们的RGB内容将其转换成适当的网格格式。
    - 把它们转换成浮点张量。
    - 将张量从0到255之间的值重新缩放到0到1之间的值，因为神经网络更喜欢处理小的输入值。
    幸运的是，所有这些任务都可以用tf.keras提供的ImageDataGenerator类来完成。
    它可以从磁盘读取图像，并将它们预处理成适当的张量。它还将设置发生器，将这些图像转换成一批张量——这对训练网络很有帮助。
"""
model.load_weights('./save_weights/my_save_weights')

base_dir = 'cats_and_dogs_filtered/'

filePath = base_dir + '/tmp/content/'
keys = os.listdir(filePath)
# 获取所有文件名
# for i,j,k in os.walk(filePath):
#     keys = k
for fn in keys:
    # 对图片进行预测
    # 读取图片
    path = base_dir + '/tmp/content/' + fn
    print(path)
    img = image.load_img(path, target_size=(150, 150))
    # 在第0维添加维度变为1x150x150x3，和我们模型的输入数据一样
    x = np.expand_dims(x, axis=0)
    # np.vstack:按垂直方向    x = image.img_to_array(img)（行顺序）堆叠数组构成一个新的数组，我们一次只有一个数据所以不这样也可以
    images = np.vstack([x])
    # batch_size批量大小，程序会分批次地预测测试数据，这样比每次预测一个样本会快。因为我们也只有一个测试所以不用也可以
    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0:
        print(fn + " is a dog")

    else:
        print(fn + " is a cat")


from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
# src=Image.open('C:/Users/ASUS/PycharmProjects/demo01/cad/cats_and_dogs_filtered/tmp/content/9bf80786bb8c5837bec6519c86f79906.jpg')

im = np.array(Image.open('./cats_and_dogs_filtered/tmp/content/9bf80786bb8c5837bec6519c86f79906.jpg').convert('L'))

hist(im.flatten(),128)

show()

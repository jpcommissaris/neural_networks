import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist  # a given dataset that comes with keras
(train_images, train_labels), (test_images, test_labels) = data.load_data()  # loads data into variables we need

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0



# testing
print(train_labels[6])
print(train_images[7])  # 28X28 numpy array that stores rgb grayscale numbers

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()



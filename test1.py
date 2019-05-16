import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt

# --- grab the sample dataset from keras ---
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()  # loads data into variables we need
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0

# --- create layers ---
#  input 784 different neurons, 1 per pixel
layer1 = keras.layers.Flatten(input_shape=(28,28))  # flatten put all our data into a single array
# hidden lay, 128 neurons,  (picks up on patterns) (usually pick 1 at 15-20% data size)
layer2 = keras.layers.Dense(128, activation="relu") # dense layer aka each neron connected to every other neuron in previous layer
                                                    # relu --> fast and good activiation function (can pick others)
# final layer, 10 different choices, highest percent light-up chosen
layer3 = keras.layers.Dense(10, activation="softmax") # softmax does probablity of each neuron being correct


# --- create model ---
model = keras.Sequential([layer1, layer2, layer3])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# different loss funcs and optimizers: arbitrary and situational
model.fit(train_images, train_labels, epochs=5) # trains model, epochs= num times data is reseen

# --- make predictions ---
prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[0])])
print()

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()




'''
# --- test: test the model ---
#test on new unseen testdata
test_loss, test_acc = model.evaluate(test_images, test_labels) # compares real answer to answer model gives
print("Test for acc: ", test_acc) 
'''

'''
# --- test: show an image---
print(train_labels[6])
print(train_images[7])  # 28X28 numpy array that stores rgb grayscale numbers

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()
'''



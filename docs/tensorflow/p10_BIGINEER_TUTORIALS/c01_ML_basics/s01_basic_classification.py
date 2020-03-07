# #!

# # Train your first neural network: basic classification
# # (https://www.tensorflow.org/alpha/tutorials/keras/basic_classification)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.__version__

# ## Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# ## Explore the data
(train_images.shape, train_labels.shape)

# -
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# ## Build the model
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# ## Train the model
model.fit(train_images, train_labels, epochs=5)

# ## Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc


# ## Make predictions
predictions = model.predict(test_images)
predictions[0]
# -
(np.argmax(predictions[0]), test_labels[0])
# -
img = test_images[0]
img.shape
# -
img = np.expand_dims(img, 0)
model.predict(img)

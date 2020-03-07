# # Save and restore models
# # (https://www.tensorflow.org/alpha/tutorials/keras/save_and_restore_models)

# ## Setup
import os
import time

import tensorflow as tf
from tensorflow import keras

from ivory.utils.path import cache_dir

train, test = tf.keras.datasets.mnist.load_data()
train_images, train_labels = train
test_images, test_labels = test
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# ## Define a model
# !Returns a short sequence model
def create_model():
    model = tf.keras.models.Sequential(
        [
            keras.layers.Dense(512, activation="relu", input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# -
# !Create a basic model instance
model = create_model()
model.summary()

# ## Save checkpoints during training
# ### Checkpoint callback usage
checkpoint_dir = cache_dir("tensorflow/ml_basics/training_1")
checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")

# !Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_weights_only=True, verbose=1
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback],
)

# -
os.listdir(checkpoint_dir)

# -
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# -
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# ## Checkpoint callback options
# !include the epoch in the file name. (uses `str.format`)
checkpoint_dir = cache_dir("tensorflow/ml_basics/training_2")
checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5,
)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(
    train_images,
    train_labels,
    epochs=50,
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0,
)

# -
os.listdir(checkpoint_dir)
# -
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# -
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# ## Save the entire model
# ### As an HDF5 file
model = create_model()
model.fit(train_images, train_labels, epochs=5)
# !Save entire model to a HDF5 file
model_dir = cache_dir("tensorflow/ml_basics/model")
path = os.path.join(model_dir, "my_model.h5")
model.save(path)
# !Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model(path)
new_model.summary()
# !Check its accuracy:
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# ### As a saved_model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# !Create a saved_model, and place it in a time-stamped directory:
saved_model_path = cache_dir(
    "tensorflow/ml_basics/saved_models/{}".format(int(time.time()))
)

tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path
# -
os.listdir(saved_model_path)

# -
# !Reload a fresh keras model from the saved model.
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()
# -
model.predict(test_images).shape
# -
new_model.compile(
    optimizer=model.optimizer,  # keep the optimizer that was loaded
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# !Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

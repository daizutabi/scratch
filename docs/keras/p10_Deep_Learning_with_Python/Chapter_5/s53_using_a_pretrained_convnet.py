# # 5.3 Using a pre-trained convnet
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/5.3-using-a-pretrained-convnet.ipynb)

import os

import altair as alt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16

from ivory.utils.path import cache_dir
from ivory.utils.keras.history import history_to_dataframe

# ### Feature extraction
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

# -
base = "keras/ch5/cats_and_dogs_small"
dirs = {}
for dataset in ["train", "validation", "test"]:
    dirs[dataset] = cache_dir(base, dataset)
datagen = ImageDataGenerator(rescale=1.0 / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory, target_size=(150, 150), batch_size=batch_size, class_mode="binary"
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(dirs["train"], 2000)
validation_features, validation_labels = extract_features(dirs["validation"], 1000)
test_features, test_labels = extract_features(dirs["test"], 1000)

# -
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# -
model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5), loss="binary_crossentropy", metrics=["acc"]
)

history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels),
)


# -
df = history_to_dataframe(history, ["loss", "acc"])
df.head(3)

chart = (
    alt.Chart(df)
    .mark_point()
    .encode(x="epoch", shape="type", color="type")
    .properties(width=250, height=200)
)
alt.hconcat(chart.encode(y="acc"), chart.encode(y="loss"))

# -
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
# -
print(
    "This is the number of trainable weights " "before freezing the conv base:",
    len(model.trainable_weights),
)
# -
conv_base.trainable = False
print(
    "This is the number of trainable weights " "after freezing the conv base:",
    len(model.trainable_weights),
)

# -
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# !Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    dirs["train"],
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary",
)

validation_generator = test_datagen.flow_from_directory(
    dirs["validation"], target_size=(150, 150), batch_size=20, class_mode="binary"
)

model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=2e-5), metrics=["acc"]
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2,
)

# -
model.save(os.path.join(cache_dir("keras/ch5"), "cats_and_dogs_small_3.h5"))

# -
df = history_to_dataframe(history, ["loss", "acc"])
df.head(3)

chart = (
    alt.Chart(df)
    .mark_point()
    .encode(x="epoch", shape="type", color="type")
    .properties(width=250, height=200)
)
alt.hconcat(chart.encode(y="acc"), chart.encode(y="loss"))


# ### Fine-tuning
conv_base.summary()

# -
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# -
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-5), metrics=["acc"]
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
)

# -
model.save(os.path.join(cache_dir("keras/ch5"), "cats_and_dogs_small_4.h5"))

# -
df = history_to_dataframe(history, ["loss", "acc"])
df.head(3)

chart = (
    alt.Chart(df)
    .mark_point()
    .encode(x="epoch", shape="type", color="type")
    .properties(width=250, height=200)
)
alt.hconcat(chart.encode(y="acc"), chart.encode(y="loss"))

# -
test_generator = test_datagen.flow_from_directory(
    dirs['test'], target_size=(150, 150), batch_size=20, class_mode="binary"
)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print("test acc:", test_acc)

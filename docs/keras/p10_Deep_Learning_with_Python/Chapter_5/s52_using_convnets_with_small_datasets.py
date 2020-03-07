# # 5.2 Using convnets with small datasets
# # (https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/
# # blob/master/5.2-using-convnets-with-small-datasets.ipynb)
import os
import shutil

import altair as alt
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ivory.utils.path import cache_dir
from ivory.utils.keras.history import history_to_dataframe

# ### Downloading the data

# [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

# !The path to the directory where the original dataset was uncompressed
base = "keras/ch5"
original_dataset_dir = cache_dir(base, "kaggle_original_data")

# !The directory where we will store our smaller dataset
name = "cats_and_dogs_small"
cache_dir(base, name, rmtree=True)

# !Directory with our (training/validation/test, cat/dog)-pictures
dirs = {}
for dataset in ["train", "validation", "test"]:
    for kind in ["cat", "dog"]:
        dirs[(dataset, kind)] = cache_dir(base, name, dataset, kind)
# !Copy first 1000 cat/dog images to train_XXX_dir
for kind in ["cat", "dog"]:
    fnames = [f"{kind}.{i}.jpg" for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(dirs[("train", kind)], fname)
        shutil.copyfile(src, dst)
# !Copy next 500 cat/dog images to validation_XXX_dir
for kind in ["cat", "dog"]:
    fnames = [f"{kind}.{i}.jpg" for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(dirs[("validation", kind)], fname)
        shutil.copyfile(src, dst)
# !Copy next 500 cat/dog images to test_XXX_dir
for kind in ["cat", "dog"]:
    fnames = [f"{kind}.{i}.jpg" for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(dirs[("test", kind)], fname)
        shutil.copyfile(src, dst)
# -
for dataset in ["train", "validation", "test"]:
    for kind in ["cat", "dog"]:
        print(f"total {dataset} {kind} images:", len(os.listdir(dirs[(dataset, kind)])))
# ### Building our network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"]
)
model.summary()

# ### Data preprocessing

# !All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    os.path.dirname(dirs[("train", "cat")]),
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary",
)

validation_generator = test_datagen.flow_from_directory(
    os.path.dirname(dirs[("validation", "cat")]),
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary",
)

# -
for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break
# -
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
)

# -
model.save(os.path.join(cache_dir(base), "cats_and_dogs_small_1.h5"))

# -
df = history_to_dataframe(history, ["acc", "loss"])
chart = (
    alt.Chart(df)
    .mark_point()
    .encode(x="epoch", color="type", shape="type")
    .properties(width=250, height=200)
)
alt.hconcat(chart.encode(y="acc"), chart.encode(y="loss"))

# ### Using data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# -
dir_ = dirs[("train", "cat")]
fnames = [os.path.join(dir_, fname) for fname in os.listdir(dir_)]
# !We pick one image to "augment"
img_path = fnames[3]
# !Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))
# !Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)
# !Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
# !The .flow() command below generates batches of randomly transformed images.
# !It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
# -
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"]
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
)

# !Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    os.path.dirname(dirs[("train", "cat")]),
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=32,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary",
)

validation_generator = test_datagen.flow_from_directory(
    os.path.dirname(dirs[("validation", "cat")]),
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
)


# -
model.save(os.path.join(cache_dir(base), "cats_and_dogs_small_2.h5"))

# -
df = history_to_dataframe(history, ["acc", "loss"])
chart = (
    alt.Chart(df)
    .mark_point()
    .encode(x="epoch", color="type", shape="type")
    .properties(width=250, height=200)
)
alt.hconcat(chart.encode(y="acc"), chart.encode(y="loss"))

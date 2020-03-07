from tensorflow.keras.datasets import mnist

from ivory.common.dataset import Dataset


def load_data(flatten=True):
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    if flatten:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)
    else:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

    return (x_train, t_train), (x_test, t_test)


def load_dataset(flatten=True, train_only=False):
    train, test = load_data(flatten=flatten)
    data = Dataset(train, name="mnist_train")
    if train_only:
        return data
    else:
        return data, Dataset(test, name="mnist_test")

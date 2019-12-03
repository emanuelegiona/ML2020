"""
Model training and parameter tuning.
"""

from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from src.models import FeatureExtractor, EmaNet, TransferNet
from src.preprocess import load_dataset


def get_time():
    """
    Gets the current time
    :return: Current time, as String
    """

    return str(datetime.now())


def log_message(file_handle, message, to_stdout=True, with_time=True):
    """
    Log utility function
    :param file_handle: Open file handle to the log file
    :param message: Log message
    :param to_stdout: True: also prints the message to the standard output (default); False: only writes to file
    :param with_time: True: appends time at the end of the line (default); False: only prints the given message
    :return: None
    """

    if with_time:
        message = "%s [%s]" % (message, get_time())

    if file_handle is not None:
        file_handle.write("%s\n" % message)
        file_handle.flush()

    if to_stdout:
        print("%s" % message)


def kfold_cross_validation(dataset_inputs: np.ndarray, dataset_labels: np.ndarray,
                           model_id: str, model: Model,
                           batch_size: int, epochs: int = 100, k_folds: int = 5) -> (float, float, float, float):
    """
    Performs K-fold cross validation of the given model, returning minimum, average (+ std dev), and maximum accuracy achieved.

    :param dataset_inputs: NumPy array representing input samples of the whole dataset
    :param dataset_labels: NumPy array representing labels of the whole dataset
    :param model_id: str identifying the current model being evaluated
    :param model: Keras model instance (ATTENTION: must have freshly initialized weights - straight from 'compile()')
    :param batch_size: Size of the batch to use
    :param epochs: Number of epochs to train the model
    :param k_folds: Number of folds to use for CV (default: 5)

    :return: (min accuracy, average accuracy, accuracy standard deviation, max accuracy)
    """

    history = []
    model.save_weights("../misc/cv/{model_id}.h5".format(model_id=model_id))
    for _ in range(k_folds):
        # start with fresh weights
        model.load_weights("../misc/cv/{model_id}.h5".format(model_id=model_id))

        # 80% training, 20% validation sets
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(dataset_inputs,
                                                                                dataset_labels,
                                                                                test_size=0.2)

        # train the model
        model.fit(x=train_inputs, y=train_labels, batch_size=batch_size,
                  validation_data=(test_inputs, test_labels), epochs=epochs,
                  callbacks=[EarlyStopping(monitor="val_acc", patience=10)],
                  verbose=2)

        # evaluate it
        _, val_acc = model.evaluate(x=test_inputs, y=test_labels, verbose=1)
        history.append(val_acc)

    history = np.asarray(history)
    return np.min(history), np.mean(history), np.std(history), np.max(history)


def grid_search(dataset_inputs: np.ndarray, dataset_labels: np.ndarray,
                batch_size: int, epochs: int = 100, k_folds: int = 5) -> None:

    # TODO
    pass


if __name__ == "__main__":
    train = "../data/MWI-Dataset-1.1.1_400/"
    batch_size = 64

    train_datagen = ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(directory=train,
                                                  target_size=(48, 64),
                                                  color_mode="rgb",
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=False)

    # whole dataset
    inputs, labels = load_dataset(train_generator=train_gen)

    # model
    my_model = EmaNet(input_shape=train_gen.image_shape,
                      num_classes=train_gen.num_classes)

    # k-fold CV
    min_acc, mean_acc, std_acc, max_acc = kfold_cross_validation(dataset_inputs=inputs,
                                                                 dataset_labels=labels,
                                                                 model_id="EmaNet_base",
                                                                 model=my_model,
                                                                 batch_size=batch_size,
                                                                 epochs=50)

    print("Min/Avg +- std/Max accuracy: {v1:.3f} / {v2:.3f} +- {v3:.3f} / {v4:.3f}".format(v1=min_acc,
                                                                                           v2=mean_acc,
                                                                                           v3=std_acc,
                                                                                           v4=max_acc))

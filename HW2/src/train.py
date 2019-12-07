"""
Model training and parameter tuning.
"""

from math import ceil
from datetime import datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from typing import Tuple, Any

from src.models import FeatureExtractor, EmaNet, TransferNet
from src.preprocess import load_dataset


class CustomEarlyStopping(EarlyStopping):
    """
    Extends the EarlyStopping callback in order to consider a minimum number of training epochs before starting to
    monitor the selected metric.
    """

    def __init__(self, monitor: str = "val_loss", min_delta: int = 0, patience: int = 0, min_epochs: int = 10,
                 verbose: int = 0, mode: str = "auto", baseline: Any = None, restore_best_weights: bool = False):

        self.__min_epochs = min_epochs
        super(CustomEarlyStopping, self).__init__(monitor=monitor,
                                                  min_delta=min_delta,
                                                  patience=patience,
                                                  verbose=verbose,
                                                  mode=mode,
                                                  baseline=baseline,
                                                  restore_best_weights=restore_best_weights)

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.__min_epochs:
            super(CustomEarlyStopping, self).on_epoch_end(epoch=epoch, logs=logs)


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


def kfold_cross_validation(model_id: str, model: Model,
                           dataset_inputs: np.ndarray = None, dataset_labels: np.ndarray = None,
                           train_generator: DirectoryIterator = None, test_generator: DirectoryIterator = None,
                           batch_size: int = 32, epochs: int = 100, k_folds: int = 5) -> (float, float, float, float):
    """
    Performs K-fold cross validation of the given model, returning minimum, average (+ std dev), and maximum accuracy achieved.
    Supports generator-based training providing train_generator instead of dataset_inputs and dataset_labels.
    In this case, test_generator has to be provided as well.
    In case dataset_inputs, dataset_labels, and test_generator are provided: the validation set will be extracted from
    dataset_inputs and dataset_labels, but test_generator will be used when reporting the accuracy score.

    :param dataset_inputs: NumPy array representing input samples of the whole training set
    :param dataset_labels: NumPy array representing labels of the whole training set
    :param train_generator: DirectoryIterator instance for iterating over the training set
    :param test_generator: DirectoryIterator instance for iterating over the test set
    :param model_id: str identifying the current model being evaluated
    :param model: Keras model instance (ATTENTION: must have freshly initialized weights - straight from 'compile()')
    :param batch_size: Size of the batch to use (default: 32)
    :param epochs: Number of epochs to train the model (default: 100)
    :param k_folds: Number of folds to use for CV (default: 5)

    :return: (min accuracy, average accuracy, accuracy standard deviation, max accuracy)
    """

    assert not ((dataset_inputs is None or dataset_labels is None) and train_generator is None), "dataset_inputs and dataset_labels must be provided if train_generator is not in use"
    assert not (train_generator is not None and test_generator is None), "test_generator must be provided if train_generator is in use"

    history = []
    print("Saving weights...")
    model.save_weights("../misc/cv/{model_id}.h5".format(model_id=model_id))

    for i in range(1, k_folds+1):
        print("{m} | Iteration {i}/{k}".format(m=model_id, i=i, k=k_folds))

        # start with fresh weights
        print("Loading weights...")
        model.load_weights("../misc/cv/{model_id}.h5".format(model_id=model_id))

        # early stopping in case validation set accuracy does not increase
        early_stopping_callback = CustomEarlyStopping(monitor="val_acc", patience=5, min_epochs=int(epochs/4))
        val_acc = 0.

        # train the model
        print("Training model...")
        # dataset-based training
        if dataset_inputs is not None and dataset_labels is not None:
            # use 80% as training set, remaining 20% as validation set
            train_inputs, val_inputs, train_labels, val_labels = train_test_split(dataset_inputs,
                                                                                  dataset_labels,
                                                                                  test_size=0.2)

            h = model.fit(x=train_inputs, y=train_labels,
                          validation_data=(val_inputs, val_labels),
                          batch_size=batch_size, epochs=epochs,
                          verbose=0, callbacks=[early_stopping_callback])

            # model evaluation
            if test_generator is not None:
                val_acc = model.evaluate_generator(generator=test_generator, steps=ceil(test_generator.n/batch_size))
                val_acc = val_acc[1]
            else:
                val_acc = h.history["val_acc"][-1]

        # generator-based training
        elif train_generator is not None and test_generator is not None:
            h = model.fit_generator(generator=train_generator, steps_per_epoch=ceil(train_generator.n/batch_size),
                                    validation_data=test_generator, validation_steps=ceil(test_generator.n/batch_size),
                                    callbacks=[early_stopping_callback], epochs=epochs, verbose=0)

            # model evaluation
            val_acc = h.history["val_acc"][-1]

        history.append(val_acc)
        print("Training ended/stopped at epoch {e} with accuracy {a:.3f}.".format(e=early_stopping_callback.stopped_epoch,
                                                                                  a=val_acc))

    history = np.asarray(history)
    return np.min(history), np.mean(history), np.std(history), np.max(history)


def grid_search(dataset_inputs: np.ndarray, dataset_labels: np.ndarray, test_generator: DirectoryIterator,
                input_shape: Tuple[int, int], num_classes: int, log_path: str,
                batch_size: int = 32, epochs: int = 100, k_folds: int = 5) -> None:

    # tuning parameters for an EmaNet model
    emanet_params = {"convolutions": [2, 3],
                     "dense_layers": [3, 5],
                     "filters": [32],
                     "dropout": [0.2, 0.3]}

    # tuning parameters for a TransferNet model
    transfernet_params = {"feature_extractor": ["Dense201", "NASNetMobile"],
                          "dense_layers": [3, 5],
                          "dropout": [0.2, 0.3]}

    # holds scores of each model
    scores = {}

    with open(log_path, "w"):
        pass

    with open(log_path, "a") as log:
        log_message(file_handle=log,
                    message="Starting grid search...")

        # EmaNet search
        for conv in emanet_params["convolutions"]:
            for dense in emanet_params["dense_layers"]:
                for filters in emanet_params["filters"]:
                    for prob in emanet_params["dropout"]:
                        model_id = "EmaNet_c{c}_d{d}_f{f}_p{p}".format(c=conv,
                                                                       d=dense,
                                                                       f=filters,
                                                                       p=prob)
                        log_message(file_handle=log,
                                    message="Testing model: {m}".format(m=model_id))

                        model = EmaNet(input_shape=input_shape,
                                       num_classes=num_classes,
                                       convolutions=conv,
                                       dense_layers=dense,
                                       filters=filters,
                                       dropout_rate=prob)

                        min_acc, mean_acc, std_acc, max_acc = kfold_cross_validation(dataset_inputs=dataset_inputs,
                                                                                     dataset_labels=dataset_labels,
                                                                                     test_generator=test_generator,
                                                                                     model_id=model_id,
                                                                                     model=model,
                                                                                     batch_size=batch_size,
                                                                                     epochs=epochs,
                                                                                     k_folds=k_folds)

                        scores[model_id] = mean_acc
                        res = "Min/Avg (std)/Max accuracy: {v1:.3f} / {v2:.3f} ({v3:.3f}) / {v4:.3f}".format(v1=min_acc,
                                                                                                             v2=mean_acc,
                                                                                                             v3=std_acc,
                                                                                                             v4=max_acc)

                        log_message(file_handle=log, message=res)

        # TransferNet search
        for fe in transfernet_params["feature_extractor"]:
            for dense in transfernet_params["dense_layers"]:
                for prob in transfernet_params["dropout"]:
                    model_id = "TransferNet_{fe}_d{d}_p{p}".format(fe=fe,
                                                                   d=dense,
                                                                   p=prob)
                    log_message(file_handle=log,
                                message="Testing model: {m}".format(m=model_id))

                    model = TransferNet(input_shape=input_shape,
                                        num_classes=num_classes,
                                        feature_extractor=FeatureExtractor.get(name=fe),
                                        dense_layers=dense,
                                        dropout_rate=prob)

                    min_acc, mean_acc, std_acc, max_acc = kfold_cross_validation(dataset_inputs=dataset_inputs,
                                                                                 dataset_labels=dataset_labels,
                                                                                 test_generator=test_generator,
                                                                                 model_id=model_id,
                                                                                 model=model,
                                                                                 batch_size=batch_size,
                                                                                 epochs=epochs,
                                                                                 k_folds=k_folds)

                    scores[model_id] = mean_acc
                    res = "Min/Avg (std)/Max accuracy: {v1:.3f} / {v2:.3f} ({v3:.3f}) / {v4:.3f}".format(v1=min_acc,
                                                                                                         v2=mean_acc,
                                                                                                         v3=std_acc,
                                                                                                         v4=max_acc)

                    log_message(file_handle=log, message=res)

        log_message(file_handle=log,
                    message="Grid search ended.\n\nResults:\nModel\t\t\tMean accuracy")

        for k in sorted(scores, key=scores.get, reverse=True):
            log_message(file_handle=log,
                        message="{k}\t{v}".format(k=k, v=scores[k]),
                        with_time=False)


# --- best model ---
# EmaNet achieving 76.5% in 5-fold CV on MWI-400 (no data augmentation)
# TODO: to be confirmed after new grid search
CONVOLUTIONS = 1
DENSE_LAYERS = 3
FILTERS = 32
DROPOUT_PROB = 0.4
# --- --- ---


def train_best_model(training_set_dir: str, validation_set_dir: str, batch_size: int,
                     data_augmentation: bool = True, early_stopping: bool = True) -> Model:

    # TODO: continue development after new grid search

    # perform data augmentation if needed
    training_datagenerator = ImageDataGenerator(rescale=(1./255),
                                                zoom_range=0.1,
                                                rotation_range=10,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                horizontal_flip=True,
                                                vertical_flip=False) if data_augmentation else ImageDataGenerator()

    pass


if __name__ == "__main__":
    train = "../data/MWI-Dataset-1.1.1_400/"
    test = "../data/TestSet_Weather/Weather_Dataset/"
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=(1./255))
    train_gen = train_datagen.flow_from_directory(directory=train,
                                                  target_size=(48, 48),
                                                  color_mode="rgb",
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=False)

    test_datagen = ImageDataGenerator(rescale=(1. / 255))
    test_gen = test_datagen.flow_from_directory(directory=test,
                                                target_size=(48, 48),
                                                color_mode="rgb",
                                                batch_size=batch_size,
                                                class_mode="categorical",
                                                shuffle=False)

    # whole dataset
    inputs, labels = load_dataset(train_generator=train_gen)

    grid_search(dataset_inputs=inputs,
                dataset_labels=labels,
                test_generator=test_gen,
                input_shape=train_gen.image_shape,
                num_classes=train_gen.num_classes,
                batch_size=batch_size,
                log_path="../misc/grid_search_full3.txt")

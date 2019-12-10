"""
Model training and parameter tuning.
"""

from math import ceil
from datetime import datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras import Model
from keras.callbacks import EarlyStopping, CSVLogger, History
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


class CustomCSVLogger(CSVLogger):
    """
    Extends the CSVLogger callback in order to print out epoch stats during training.
    """

    def __init__(self, filename: str, separator: str = ",", append: bool = False):
        super(CustomCSVLogger, self).__init__(filename=filename, separator=separator, append=append)

    def on_epoch_end(self, epoch, logs=None):
        super(CustomCSVLogger, self).on_epoch_end(epoch=epoch, logs=logs)
        log_message(file_handle=None,
                    message="Epoch {e}\n\ttrain loss: {v1:.3f}, train acc: {v2:.3f}\n\tval loss: {v3:.3f}, val acc: {v4:.3f}\n".format(e=epoch,
                                                                                                                                       v1=logs["loss"],
                                                                                                                                       v2=logs["acc"],
                                                                                                                                       v3=logs["val_loss"],
                                                                                                                                       v4=logs["val_acc"]))


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
    Supports generator-based training providing generator instead of dataset_inputs and dataset_labels.
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

    assert not ((dataset_inputs is None or dataset_labels is None) and train_generator is None), "dataset_inputs and dataset_labels must be provided if generator is not in use"
    assert not (train_generator is not None and test_generator is None), "test_generator must be provided if generator is in use"

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
    """
    Performs grid parameter search with integrated k-fold cross validation, printing results to a log file.
    Models are trained on k random splits of the training set, using an 80-20 partition scheme for training and
    validation sets, then evaluated on the given test set (test_generator).

    :param dataset_inputs: NumPy array representing input samples of the whole training set
    :param dataset_labels: NumPy array representing labels of the whole training set
    :param test_generator: DirectoryIterator instance for iterating over the test set
    :param input_shape: Shape of the input tensor as a 2-dimensional int tuple
    :param num_classes: Number of classes for the final FC layer
    :param log_path: Path to the log file
    :param batch_size: Size of the batch to use (default: 32)
    :param epochs: Number of epochs to train the model (default: 100)
    :param k_folds: Number of folds to use for CV (default: 5)

    :return: None
    """

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


def full_training(model: Model, log_path: str,
                  training_set_dir: str, validation_set_dir: str,
                  image_size: Tuple[int, int] = (48, 48), batch_size: int = 32, data_augmentation: bool = True,
                  epochs: int = 150, early_stopping: bool = True) -> (Model, History):
    """
    Performs full training of the given model, logging metrics at the end of each epoch.
    Supports techniques such as data augmentation and early stopping, the latter being enforced only after 1/3 of the
    original epochs have already been performed and after 10 epochs with no improvement.
    In case of early stopping, the best performing weights are restored.

    :param model: Keras Model instance
    :param log_path: Path to the log file (CSV format)
    :param training_set_dir: Path to the directory containing the training set
    :param validation_set_dir: Path to the directory containing the validation set
    :param image_size: Target size of images (default: 48 x 48)
    :param batch_size: Size of the batch to use (default: 32)
    :param data_augmentation: if True, performs data augmentation (default: True)
    :param epochs: Number of epochs to train the model (default: 150)
    :param early_stopping: if True, applies early stopping (default: True)

    :return: (best-performing Keras Model, History object)
    """

    training_datagenerator = ImageDataGenerator(rescale=(1. / 255))
    validation_datagenerator = ImageDataGenerator(rescale=(1. / 255))
    if data_augmentation:
        training_datagenerator = ImageDataGenerator(rescale=(1. / 255),
                                                    zoom_range=0.1,
                                                    rotation_range=10,
                                                    width_shift_range=0.1,
                                                    height_shift_range=0.1,
                                                    horizontal_flip=True,
                                                    vertical_flip=False)

    train_generator = training_datagenerator.flow_from_directory(directory=training_set_dir,
                                                                 target_size=image_size,
                                                                 color_mode="rgb",
                                                                 batch_size=batch_size,
                                                                 class_mode="categorical",
                                                                 shuffle=True)

    validation_generator = validation_datagenerator.flow_from_directory(directory=validation_set_dir,
                                                                        target_size=image_size,
                                                                        color_mode="rgb",
                                                                        batch_size=batch_size,
                                                                        class_mode="categorical",
                                                                        shuffle=False)

    callbacks = [CustomCSVLogger(filename=log_path)]
    if early_stopping:
        early_stopping_callback = CustomEarlyStopping(monitor="val_loss",
                                                      patience=10,
                                                      min_epochs=int(epochs/3),
                                                      restore_best_weights=True)
        callbacks.append(early_stopping_callback)

    h = model.fit_generator(generator=train_generator,
                            steps_per_epoch=ceil(train_generator.n/batch_size),
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=ceil(validation_generator.n/batch_size),
                            verbose=0)

    return model, h


# --- best EmaNet model ---
# 37.5% accuracy on Smart-I in 5-fold CV (training on MWI-400 - no data augmentation)
EN_CONVOLUTIONS = 2
EN_DENSE_LAYERS = 3
EN_FILTERS = 32
EN_DROPOUT_RATE = 0.2
# --- --- ---


def train_best_EmaNet(training_set_dir: str, validation_set_dir: str,
                      input_shape: Tuple[int, int], num_classes: int,
                      batch_size: int = 32, epochs: int = 150,
                      data_augmentation: bool = True, early_stopping: bool = True) -> (Model, History):
    """
    Trains an EmaNet model with the best-performing parameters from grid search, then exports it to a H5 file.

    :param training_set_dir: Path to the directory containing the training set
    :param validation_set_dir: Path to the directory containing the validation set
    :param input_shape: Shape of the input tensor as a 2-dimensional int tuple
    :param num_classes: Number of classes for the final FC layer
    :param batch_size: Size of the batch to use (default: 32)
    :param data_augmentation: if True, performs data augmentation (default: True)
    :param epochs: Number of epochs to train the model (default: 150)
    :param early_stopping: if True, applies early stopping (default: True)

    :return: (best-performing Keras Model, History object)
    """

    model_id = "EmaNet_c{c}_d{d}_f{f}_p{p}".format(c=EN_CONVOLUTIONS,
                                                   d=EN_DENSE_LAYERS,
                                                   f=EN_FILTERS,
                                                   p=EN_DROPOUT_RATE)

    print("Model: {m}".format(m=model_id))
    model = EmaNet(input_shape=input_shape,
                   num_classes=num_classes,
                   convolutions=EN_CONVOLUTIONS,
                   dense_layers=EN_DENSE_LAYERS,
                   filters=EN_FILTERS,
                   dropout_rate=EN_DROPOUT_RATE)

    print("Training...")
    model, history = full_training(model=model,
                                   log_path="../misc/{id}.csv".format(id=model_id),
                                   training_set_dir=training_set_dir,
                                   validation_set_dir=validation_set_dir,
                                   batch_size=batch_size,
                                   data_augmentation=data_augmentation,
                                   epochs=epochs,
                                   early_stopping=early_stopping)

    print("Training ended.\nSaving model...")
    model.save(filepath="../misc/EmaNet.h5")
    print("Done.")

    return model, history


# --- best TransferNet model ---
# 33% accuracy on Smart-I in 5-fold CV (training on MWI-400 - no data augmentation)
TN_FEAT_EXTRACTOR = "NASNetMobile"
TN_DENSE_LAYERS = 5
TN_DROPOUT_RATE = 0.3
# --- --- ---


def train_best_TransferNet(training_set_dir: str, validation_set_dir: str,
                           input_shape: Tuple[int, int], num_classes: int,
                           batch_size: int = 32, epochs: int = 150,
                           data_augmentation: bool = True, early_stopping: bool = True) -> (Model, History):
    """
    Trains a TransferNet model with the best-performing parameters from grid search, then exports it to a H5 file.

    :param training_set_dir: Path to the directory containing the training set
    :param validation_set_dir: Path to the directory containing the validation set
    :param input_shape: Shape of the input tensor as a 2-dimensional int tuple
    :param num_classes: Number of classes for the final FC layer
    :param batch_size: Size of the batch to use (default: 32)
    :param data_augmentation: if True, performs data augmentation (default: True)
    :param epochs: Number of epochs to train the model (default: 150)
    :param early_stopping: if True, applies early stopping (default: True)

    :return: (best-performing Keras Model, History object)
    """

    model_id = "TransferNet_{fe}_d{d}_p{p}".format(fe=TN_FEAT_EXTRACTOR,
                                                   d=TN_DENSE_LAYERS,
                                                   p=TN_DROPOUT_RATE)

    print("Model: {m}".format(m=model_id))
    model = TransferNet(input_shape=input_shape,
                        num_classes=num_classes,
                        feature_extractor=FeatureExtractor.get(TN_FEAT_EXTRACTOR),
                        dense_layers=TN_DENSE_LAYERS,
                        dropout_rate=TN_DROPOUT_RATE)

    print("Training...")
    model, history = full_training(model=model,
                                   log_path="../misc/{id}.csv".format(id=model_id),
                                   training_set_dir=training_set_dir,
                                   validation_set_dir=validation_set_dir,
                                   batch_size=batch_size,
                                   data_augmentation=data_augmentation,
                                   epochs=epochs,
                                   early_stopping=early_stopping)

    print("Training ended.\nSaving model...")
    model.save(filepath="../misc/TransferNet.h5")
    print("Done.")

    return model, history


if __name__ == "__main__":
    train = "../data/MWI-Dataset-full/"
    test = "../data/TestSet_Weather/Weather_Dataset/"
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=(1./255))
    train_gen = train_datagen.flow_from_directory(directory=train,
                                                  target_size=(48, 48),
                                                  color_mode="rgb",
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=False)

    #test_datagen = ImageDataGenerator(rescale=(1. / 255))
    #test_gen = test_datagen.flow_from_directory(directory=test,
    #                                            target_size=(48, 48),
    #                                            color_mode="rgb",
    #                                            batch_size=batch_size,
    #                                            class_mode="categorical",
    #                                            shuffle=False)

    # whole dataset
    #inputs, labels = load_dataset(generator=train_gen)

    #grid_search(dataset_inputs=inputs,
    #            dataset_labels=labels,
    #            test_generator=test_gen,
    #            input_shape=train_gen.image_shape,
    #            num_classes=train_gen.num_classes,
    #            batch_size=batch_size,
    #            log_path="../misc/grid_search_full3.txt")

    # EmaNet model training
    train_best_EmaNet(training_set_dir=train,
                      validation_set_dir=test,
                      input_shape=train_gen.image_shape,
                      num_classes=train_gen.num_classes)

    # TransferNet model training
    train_best_TransferNet(training_set_dir=train,
                           validation_set_dir=test,
                           input_shape=train_gen.image_shape,
                           num_classes=train_gen.num_classes)

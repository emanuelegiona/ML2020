"""
Model training and parameter tuning.
"""

from time import sleep
from math import ceil
from datetime import datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from typing import Tuple

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
    for i in range(1, k_folds+1):
        print("{m} | Iteration {i}/{k}".format(m=model_id, i=i, k=k_folds))

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
                  verbose=0)

        # evaluate it
        _, val_acc = model.evaluate(x=test_inputs, y=test_labels, verbose=1)
        history.append(val_acc)

    history = np.asarray(history)
    return np.min(history), np.mean(history), np.std(history), np.max(history)


def grid_search(dataset_inputs: np.ndarray, dataset_labels: np.ndarray,
                input_shape: Tuple[int, int], num_classes: int, batch_size: int, log_path: str,
                epochs: int = 100, k_folds: int = 5) -> None:

    # tuning parameters for an EmaNet model
    emanet_params = {"convolutions": [], #[1, 2, 3],
                     "dense_layers": [], #[3, 5, 7],
                     "starting_filters": [], #[32],
                     "dropout": []} #[0.3, 0.4]}

    # tuning parameters for a TransferNet model
    transfernet_params = {"feature_extractor": ["NASNetLarge"], #["Dense121", "Dense169", "Dense201", "NASNetMobile"],
                          "dense_layers": [3, 5],
                          "dropout": [0.3, 0.4]}

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
                for filters in emanet_params["starting_filters"]:
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
                                       starting_filters=filters,
                                       dropout_rate=prob)

                        min_acc, mean_acc, std_acc, max_acc = kfold_cross_validation(dataset_inputs=dataset_inputs,
                                                                                     dataset_labels=dataset_labels,
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
                        #sleep(60)

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
                    #sleep(60)

        log_message(file_handle=log,
                    message="Grid search ended.\n\nResults:\nModel\t\tMean accuracy\n")

        for k in sorted(scores, key=scores.get, reverse=True):
            log_message(file_handle=log,
                        message="{k}\t{v}".format(k=k, v=scores[k]),
                        with_time=False)


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

    grid_search(dataset_inputs=inputs,
                dataset_labels=labels,
                input_shape=train_gen.image_shape,
                num_classes=train_gen.num_classes,
                batch_size=batch_size,
                log_path="../misc/grid_search2.txt",
                epochs=50)

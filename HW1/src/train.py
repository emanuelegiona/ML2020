"""
Model training and parameter tuning.
"""

from datetime import datetime
from src.preprocess import load_dataset
from src.models import get_SVM_model, get_KNN_model
from sklearn.model_selection import GridSearchCV
import numpy as np
from joblib import dump
from typing import Any, Dict, List


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


def cross_validation(model_id: str, model: Any, param_grid: Dict[str, list],
                     dataset_inputs: np.ndarray, dataset_labels: np.ndarray,
                     k_folds: int = 5) -> (float, Dict):
    """
    Performs a grid parameter search for the given model, applying K-fold cross-validation to estimate the performance.
    :param model_id: string to identify the model being tested
    :param model: any SciKit-Learn model object
    :param param_grid: dictionary of parameters to perform grid search on, for each of them a list of values to be used
    :param dataset_inputs: np.ndarray representing the input data
    :param dataset_labels: np.ndarray representing output labels
    :param k_folds: number of folds to be used for cross-validation
    :return: Tuple (best score, best parameters)
    """

    print("Performing {k}-fold CV on {model_id} with params:\n{p}".format(k=k_folds,
                                                                          model_id=model_id,
                                                                          p=param_grid))
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=k_folds,
                               verbose=2)
    grid_search.fit(dataset_inputs, dataset_labels)
    score = grid_search.best_score_
    params = grid_search.best_params_
    print("Best score: {s:.3f} achieved with params:\n{p}".format(s=score,
                                                                  p=params))

    return score, params


def find_best_model(train_path: str, input_dict_paths: List[str], ngrams_sizes: List[int],
                    compiler_dict_path: str, optimization_dict_path: str, log_path: str) -> None:
    """
    Performs grid parameter search with integrated cross-validation over multiple models and feature engineering
    techniques, writing results to a log file for further inspection.
    :param train_path: path to the training set file (JSONL-formatted)
    :param input_dict_paths: paths to the input dictionary file (as List[str]) - WARNING: must match len(ngrams_sizes)
    :param ngrams_sizes: sliding window size to be used when building ngrams (if 1, ngrams are NOT used) (as List[int]) - WARNING: must match len(input_dict_paths)
    :param compiler_dict_path: path to the compilers input/output dictionary file
    :param optimization_dict_path: path to the optimization levels input/output dictionary file
    :param log_path: path to the log file to be written
    :return: None
    """

    assert len(input_dict_paths) == len(ngrams_sizes), "Input dictionary paths and ngrams sizes lists size mismatch."

    with open(log_path, "w"):
        pass

    with open(log_path, "a") as log:
        for i, input_dict_path in enumerate(input_dict_paths):
            log_message(file_handle=log,
                        message="Dataset: {path} using ngrams size: {size}".format(path=input_dict_path,
                                                                                   size=ngrams_sizes[i]))

            inputs, labels = load_dataset(path=train_path,
                                          input_dict_path=input_dict_path,
                                          compiler_dict_path=compiler_dict_path,
                                          optimization_dict_path=optimization_dict_path,
                                          ngrams_size=ngrams_sizes[i],
                                          train=True)

            # perform CV on a reduced dataset
            inputs = np.vstack([inputs[:2_500],
                                inputs[10_000:12_500],
                                inputs[20_000:22_500]])

            labels = np.vstack([labels[:2_500],
                                labels[10_000:12_500],
                                labels[20_000:22_500]])

            task = "compiler"
            model = "LinearSVC"
            model_id = "{model} - {task}".format(model=model, task=task)

            log_message(file_handle=log,
                        message="Task: {task}\nModel\tScore\tParams".format(task=task))
            score, params = cross_validation(model_id=model_id,
                                             model=get_SVM_model(),
                                             param_grid={"C": [1.0, 1.5, 2.0, 3.0]},
                                             dataset_inputs=inputs,
                                             dataset_labels=labels[:, 0])
            log_message(file_handle=log,
                        message="{model}\t{score}\t{params}".format(model=model,
                                                                    score=score,
                                                                    params=params))

            model = "KNN"
            model_id = "{model} - {task}".format(model=model, task=task)
            score, params = cross_validation(model_id=model_id,
                                             model=get_KNN_model(),
                                             param_grid={"n_neighbors": [3, 5],
                                                         "weights": ["uniform", "distance"]},
                                             dataset_inputs=inputs,
                                             dataset_labels=labels[:, 0])
            log_message(file_handle=log,
                        message="{model}\t{score}\t{params}".format(model=model,
                                                                    score=score,
                                                                    params=params))

            # --- delimit tasks ---
            log_message(file_handle=log,
                        message="\n")

            task = "optimization"
            model = "LinearSVC"
            model_id = "{model} - {task}".format(model=model, task=task)

            log_message(file_handle=log,
                        message="Task: {task}\nModel\tScore\tParams".format(task=task))
            score, params = cross_validation(model_id=model_id,
                                             model=get_SVM_model(),
                                             param_grid={"C": [1.0, 1.5, 2.0, 3.0]},
                                             dataset_inputs=inputs,
                                             dataset_labels=labels[:, 1])
            log_message(file_handle=log,
                        message="{model}\t{score}\t{params}".format(model=model,
                                                                    score=score,
                                                                    params=params))

            model = "KNN"
            model_id = "{model} - {task}".format(model=model, task=task)
            score, params = cross_validation(model_id=model_id,
                                             model=get_KNN_model(),
                                             param_grid={"n_neighbors": [3, 5],
                                                         "weights": ["uniform", "distance"]},
                                             dataset_inputs=inputs,
                                             dataset_labels=labels[:, 1])
            log_message(file_handle=log,
                        message="{model}\t{score}\t{params}".format(model=model,
                                                                    score=score,
                                                                    params=params))

            # --- delimit dataset from another ---
            log_message(file_handle=log,
                        message="\n--- --- ---\n")


def linearSVC_best_tuning_CV(train_path: str, input_dict_path: str, ngrams_size: int,
                             compiler_dict_path: str, optimization_dict_path: str) -> None:
    """
    Performs K-fold cross-validation on the whole dataset using the best performing model.
    :param train_path: path to the training set file (JSONL-formatted)
    :param input_dict_path: path to the input dictionary file
    :param ngrams_size: sliding window size to be used when building ngrams (if 1, ngrams are NOT used)
    :param compiler_dict_path: path to the compilers input/output dictionary file
    :param optimization_dict_path: path to the optimization levels input/output dictionary file
    :return: None
    """

    inputs, labels = load_dataset(path=train_path,
                                  input_dict_path=input_dict_path,
                                  compiler_dict_path=compiler_dict_path,
                                  optimization_dict_path=optimization_dict_path,
                                  ngrams_size=ngrams_size,
                                  train=True)

    log_message(file_handle=None,
                message="Task: {compiler}\nScore\tParams")
    score, params = cross_validation(model_id="LinearSVC - compiler",
                                     model=get_SVM_model(),
                                     param_grid={"C": [BEST_C]},
                                     dataset_inputs=inputs,
                                     dataset_labels=labels[:, 0])
    log_message(file_handle=None,
                message="{score}\t{params}".format(score=score,
                                                   params=params))

    log_message(file_handle=None,
                message="Task: {optimization}\nScore\tParams")
    score, params = cross_validation(model_id="LinearSVC - optimization",
                                     model=get_SVM_model(),
                                     param_grid={"C": [BEST_C]},
                                     dataset_inputs=inputs,
                                     dataset_labels=labels[:, 1])
    log_message(file_handle=None,
                message="{score}\t{params}".format(score=score,
                                                   params=params))


def train_and_export(model_id: str,
                     train_path: str, input_dict_path: str, ngrams_size: int,
                     compiler_dict_path: str, optimization_dict_path: str,
                     export_dir: str) -> None:
    """
    Trains the best performing model and exports it to file.
    :param model_id: name of the model
    :param train_path: path to the training set file (JSONL-formatted)
    :param input_dict_path: path to the input dictionary file
    :param ngrams_size: sliding window size to be used when building ngrams (if 1, ngrams are NOT used)
    :param compiler_dict_path: path to the compilers input/output dictionary file
    :param optimization_dict_path: path to the optimization levels input/output dictionary file
    :param export_dir: path to directory where to export the models to
    :return: None
    """

    log_message(file_handle=None,
                message="Training started: {model_id}".format(model_id=model_id))

    inputs, labels = load_dataset(path=train_path,
                                  input_dict_path=input_dict_path,
                                  compiler_dict_path=compiler_dict_path,
                                  optimization_dict_path=optimization_dict_path,
                                  ngrams_size=ngrams_size,
                                  train=True)

    log_message(file_handle=None,
                message="\t- training for compiler classification")
    svm_compiler = get_SVM_model(C=BEST_C,
                                 verbose=2)
    svm_compiler.fit(inputs, labels[:, 0])

    log_message(file_handle=None,
                message="\t- training for optimization level classification")
    svm_optimization = get_SVM_model(C=BEST_C,
                                     verbose=2)
    svm_optimization.fit(inputs, labels[:, 1])

    log_message(file_handle=None,
                message="Training ended.\nExporting to {path}...".format(path=export_dir))

    path = "{dir}/{model_id}_compiler.model".format(dir=export_dir,
                                                    model_id=model_id)
    with open(path, "wb"):
        pass
    dump(svm_compiler, path)

    path = "{dir}/{model_id}_optimization.model".format(dir=export_dir,
                                                        model_id=model_id)
    with open(path, "wb"):
        pass
    dump(svm_optimization, path)

    log_message(file_handle=None,
                message="Done.")


# --- best tuning ---
BEST_C = 1.5
# --- --- ---


if __name__ == "__main__":
    #find_best_model(train_path="../data/train_dataset.jsonl",
    #                input_dict_paths=["../data/mnemonics.txt",
    #                                  "../data/2grams.txt"],
    #                ngrams_sizes=[1, 2],
    #                compiler_dict_path="../data/compilers.txt",
    #                optimization_dict_path="../data/optimizations.txt",
    #                log_path="../misc/cv.log")

    #linearSVC_best_tuning_CV(train_path="../data/train_dataset.jsonl",
    #                         input_dict_path="../data/2grams.txt",
    #                         ngrams_size=2,
    #                         compiler_dict_path="../data/compilers.txt",
    #                         optimization_dict_path="../data/optimizations.txt")

    train_and_export(model_id="svm",
                     train_path="../data/train_dataset.jsonl",
                     input_dict_path="../data/2grams.txt",
                     ngrams_size=2,
                     compiler_dict_path="../data/compilers.txt",
                     optimization_dict_path="../data/optimizations.txt",
                     export_dir="../misc/")

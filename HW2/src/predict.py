"""
Prediction and creation of output files in the given format.
"""

import numpy as np
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List

from src.preprocess import load_dataset
from src.train import log_message


def class_label(indices: np.ndarray, reverse_dictionary: List[str]) -> List[str]:
    """
    Retrieves the associated class labels to the given indices array.

    :param indices: Predicted class indices
    :param reverse_dictionary: Class index to label mapping

    :return: Array of class labels
    """

    return [reverse_dictionary[i] for i in indices]


def predict(model_path: str, test_set_dir: str,
            classes: Dict[str, int] = None, output_path: str = None,
            batch_size: int = 32) -> np.ndarray:
    """
    Loads a model from a H5 file and uses it for predicting classes for the given test set.
    Optionally, writes the output to a CSV file, in which each row contains the class label associated to the image,
    in the order given by Keras's flow_from_directory().

    :param model_path: Path to the model's H5 file
    :param test_set_dir: Path to the test set directory
    :param classes: Optional dictionary mapping class labels to indices (REQUIRED in case of CSV output)
    :param output_path: Optional path to the CSV file to write the output to
    :param batch_size: Size of batches to use (default: 32)

    :return: Class indices predictions as NumPy array
    """

    assert not (output_path is not None and classes is None), "classes dictionary must be provided if predictions will be printed to an output file"

    datagenerator = ImageDataGenerator(rescale=(1. / 255))
    generator = datagenerator.flow_from_directory(directory=test_set_dir,
                                                  target_size=(48, 48),
                                                  color_mode="rgb",
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=False)

    # reverse dictionary for classes
    rev_classes = [k for k, v in classes.items()] if classes is not None else None

    print("Predicting...")
    model = load_model(filepath=model_path)
    preds = model.predict_generator(generator=generator,
                                    steps=ceil(generator.n/batch_size))
    print("Done.")

    # scalar representation from the 1-of-K encoding one
    preds = np.argmax(preds, axis=1)

    # output to CSV file
    if output_path is not None:
        print("Exporting to CSV...")
        preds_label = class_label(indices=preds, reverse_dictionary=rev_classes)

        with open(output_path, "w"):
            pass

        with open(output_path, "a") as csv:
            for pred in preds_label:
                log_message(file_handle=csv,
                            message="{pred}".format(pred=pred),
                            with_time=False,
                            to_stdout=False)

        print("Done.")

    return preds


def evaluate(ground_truth: np.ndarray, predictions: np.ndarray, classes: Dict[str, int] = None) -> None:
    """
    Evaluates a model's performance given its predictions and the reference ground truth values.
    Both ground_truth and predictions are expected to be in the SCALAR REPRESENTATION and not in the 1-of-K encoding.

    :param ground_truth: NumPy array containing ground truth class indices
    :param predictions: NumPy array containing predicted class indices
    :param classes: Optional dictionary mapping class labels to indices

    :return: None
    """

    # reverse dictionary for classes
    rev_classes = [k for k, v in classes.items()] if classes is not None else None

    print("Accuracy: {v:.3f}\n".format(v=np.count_nonzero(np.equal(ground_truth, predictions)) / len(predictions)))

    if rev_classes is not None:
        ground_truth = class_label(indices=ground_truth, reverse_dictionary=rev_classes)
        predictions = class_label(indices=predictions, reverse_dictionary=rev_classes)

    print("Classification report:")
    print(classification_report(y_true=ground_truth,
                                y_pred=predictions))

    print("Confusion matrix:")
    print(confusion_matrix(y_true=ground_truth,
                           y_pred=predictions))


if __name__ == "__main__":
    train = "../data/MWI-Dataset-full/"
    test = "../data/TestSet_Weather/Weather_Dataset/"

    datagen = ImageDataGenerator(rescale=(1. / 255))
    gen = datagen.flow_from_directory(directory=train,
                                      target_size=(48, 48),
                                      color_mode="rgb",
                                      batch_size=32,
                                      class_mode="categorical",
                                      shuffle=False)

    class_dict = gen.class_indices
    del gen

    gen = datagen.flow_from_directory(directory=test,
                                      target_size=(48, 48),
                                      color_mode="rgb",
                                      batch_size=32,
                                      class_mode="categorical",
                                      shuffle=False)

    _, gold_labels = load_dataset(generator=gen,
                                  labels_only=True)
    del gen
    gold_labels = np.argmax(gold_labels, axis=1)

    preds_emanet = predict(model_path="../misc/EmaNet.h5",
                           test_set_dir=test,
                           classes=class_dict,
                           output_path="../misc/EmaNet_preds.csv")

    preds_transfernet = predict(model_path="../misc/TransferNet.h5",
                                test_set_dir=test,
                                classes=class_dict,
                                output_path="../misc/TransferNet_preds.csv")

    print("--- EmaNet ---")
    evaluate(ground_truth=gold_labels, predictions=preds_emanet, classes=class_dict)

    print("--- TransferNet ---")
    evaluate(ground_truth=gold_labels, predictions=preds_transfernet, classes=class_dict)

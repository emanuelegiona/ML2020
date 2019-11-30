"""
Pre-processing utilities for the feature engineering phase.
"""

import PIL
from PIL.Image import Image
from os import listdir
from os.path import isfile, isdir, join
from typing import Generator
import numpy as np


def image_dir_generator(directory: str, recursive: bool = True) -> Generator[Image, None, None]:
    """
    Iterate the contents of the given directory, providing instances of the Image class.
    :param directory: Path to the directory to iterate
    :param recursive: if True, enables recursive iteration (default: True)
    :return: PIL.Image instances, in a generator fashion
    """

    for name in listdir(directory):
        path = join(directory, name)
        if isfile(path):
            with PIL.Image.open(path) as img:
                yield img
        elif recursive and isdir(path):
            for img in image_dir_generator(directory=path, recursive=recursive):
                yield img


def show_stats(dataset_path: str) -> None:
    """
    Shows some statistics about the given dataset.
    :param dataset_path: Path to the main directory of the dataset
    :return: None
    """

    print("Computing stats for dataset located in: {dir} ...".format(dir=dataset_path))
    heights = []
    widths = []
    for img in image_dir_generator(directory=dataset_path):
        curr_width, curr_height = img.size
        widths.append(curr_width)
        heights.append(curr_height)

    heights = np.asarray(heights)
    widths = np.asarray(widths)

    def stats(array): return np.min(array), np.mean(array), np.std(array), np.median(array), np.max(array)

    def pretty_print(stats_array): return "{min} / {avg:.2f} +- {std:.2f} / {med} / {max}".format(min=stats_array[0],
                                                                                                  avg=stats_array[1],
                                                                                                  std=stats_array[2],
                                                                                                  med=stats_array[3],
                                                                                                  max=stats_array[4])

    print("Min/Avg/Median/Max height:\t{s}".format(s=pretty_print(stats(heights))))
    print("Min/Avg/Median/Max width:\t{s}\n".format(s=pretty_print(stats(widths))))


if __name__ == "__main__":
    dataset1 = "../data/MWI-Dataset-1.1.1_400"
    dataset2 = "../data/MWI-Dataset-1.1_2000"
    show_stats(dataset_path=dataset1)
    show_stats(dataset_path=dataset2)

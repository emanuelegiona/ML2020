"""
Pre-processing utilities for the feature engineering phase.
"""
from json import JSONDecoder
from collections import namedtuple
from copy import deepcopy
import re
import numpy as np
from typing import List, Dict


"""
NamedTuple to model a single example in the dataset, in which:
- instructions: List containing the instructions of the function, each of them being List[str] (mnemonic, arguments)
- optimization: str encoding the optimization level that was used (or None)
- compiler: str encoding the compiler that was used (or None)
"""
Function = namedtuple("Function", "instructions optimization compiler")

"""
NamedTuple to model the dictionaries built from the dataset, in which:
- mnemonics: Dict[str, int] with mnemonics of each instruction as keys, and their occurrences as values
- arguments: Dict[str, int] with arguments of each instruction as keys, and their occurrences as values
- compilers: Dict[str, int] with compilers as keys, and their occurrences as values
- optimizations: Dict[str, int] with optimization levels as keys, and their occurrences as values
"""
Dictionaries = namedtuple("Dictionaries", "mnemonics arguments compilers optimizations")


class Preprocessor:
    """
    Pre-processing class.
    """

    # special token for unknown mnemonics, arguments, compilers or optimization levels
    UNK_TOKEN = "<UNK>"

    # special token for mnemonics or arguments under a certain threshold
    FEW_TOKEN = "<FEW>"

    def __init__(self, train: bool = False, debug: bool = False):
        """
        Constructor.
        :param train: if True, it is assumed to be working on a training dataset, therefore "opt" and "compiler" fields of JSON objects must be present
        :param debug: if True, some debug information is printed
        """

        self.__decoder = JSONDecoder()
        self.__train = train
        self.__debug = debug

    def take(self, json_line: str) -> Function:
        """
        Accepts a line from a JSONL-formatted file and returns a higher-level Function object.
        :param json_line: single line from a JSONL-formatted file
        :return: Function object built from the given line
        """

        json_obj = self.__decoder.decode(json_line)

        instr_list = [instr.split(" ") for instr in json_obj["instructions"]]
        tmp_list = []

        # re-join bracketed arguments into a single one
        for instr in instr_list:
            tmp_instr = []
            bracket_start, bracket_end = None, None
            for pos, arg in enumerate(instr):
                if arg.startswith("[") and bracket_start is None:
                    bracket_start = pos
                elif arg.endswith("]") and bracket_start is not None:
                    bracket_end = pos
                elif bracket_start is None or pos < bracket_start or bracket_end is not None:
                    tmp_instr.append(arg)

            if bracket_start is not None and bracket_end is not None:
                tmp_instr.insert(bracket_start, " ".join(instr[bracket_start:bracket_end+1]))
            tmp_list.append(tmp_instr)

        # substitute the corrected instruction list
        instr_list = tmp_list

        opt_level = json_obj["opt"] if self.__train else None
        compiler = json_obj["compiler"] if self.__train else None

        ret = Function(instructions=instr_list,
                       optimization=opt_level,
                       compiler=compiler)

        return ret

    def feature_engineer(self, function: Function) -> np.ndarray:
        pass

    def stats(self, training_set: str) -> None:
        """
        Prints some statistics about the given training set.
        :param training_set: Path to the training set file (JSONL-formatted)
        :return: None
        """

        instr_num = []
        instr_lens = []
        compiler_distr = {}
        opt_distr = {}

        with open(training_set) as dataset:
            for line in dataset:
                function = self.take(json_line=line)

                instr_num.append(len(function.instructions))
                instr_lens.append([len(instr) for instr in function.instructions])

                compiler_count = compiler_distr.get(function.compiler, 0)
                compiler_distr[function.compiler] = compiler_count + 1

                opt_count = opt_distr.get(function.optimization, 0)
                opt_distr[function.optimization] = opt_count + 1

        print("Min/Avg/Max number of instructions:\n{minl}/{avgl:.3f}/{maxl}\n".format(minl=min(instr_num),
                                                                                       avgl=(sum(instr_num) / len(instr_num)),
                                                                                       maxl=max(instr_num)))

        print("Min/Max instruction length per function:\n{minl}/{maxl}\n".format(minl=min([min(instr_len) for instr_len in instr_lens]),
                                                                                 maxl=max([max(instr_len) for instr_len in instr_lens])))

        print("Compiler distribution:\n{d}\n".format(d=compiler_distr))

        print("Optimization level distribution:\n{d}\n".format(d=opt_distr))

    def build_dictionaries(self, training_set: str) -> Dictionaries:
        """
        Builds dictionaries from the given training set.
        :param training_set: Path to the training set file (JSONL-formatted)
        :return: Dictionaries object built from the given dataset
        """

        mnemonics = {}
        arguments = {}
        compilers = {}
        optimizations = {}

        with open(training_set) as dataset:
            for line in dataset:
                function = self.take(json_line=line)

                # iterate on instructions to inspect mnemonics and possible arguments
                for instr in function.instructions:
                    mnemonic_count = mnemonics.get(instr[0], 0)
                    mnemonics[instr[0]] = mnemonic_count + 1

                    # process instruction arguments, if any
                    if len(instr) > 1:
                        for arg in instr[1:]:
                            if isinstance(arg, list):
                                print(arg)
                            arg_count = arguments.get(arg, 0)
                            arguments[arg] = arg_count + 1

                compiler_count = compilers.get(function.compiler, 0)
                compilers[function.compiler] = compiler_count + 1

                opt_count = optimizations.get(function.optimization, 0)
                optimizations[function.optimization] = opt_count + 1

        ret = Dictionaries(mnemonics=mnemonics,
                           arguments=arguments,
                           compilers=compilers,
                           optimizations=optimizations)

        return ret

    def build_ngrams(self, training_set: str, size: int = 2) -> Dict[str, int]:
        """
        Builds ngrams for mnemonics.
        :param training_set: Path to the training set file (JSONL-formatted)
        :param size: Size of ngrams to be created
        :return: Dictionary of strings to integers, with ngrams as keys and occurrences as values.
        """

        ngrams = {}
        with open(training_set) as dataset:
            for line in dataset:
                function = self.take(json_line=line)

                # only consider mnemonics
                instructions = [instr[0] for instr in function.instructions]
                del function

                # iterate on instructions with a sliding window of the given size
                for i in range(len(instructions) - size + 1):
                    mnemonics = instructions[i:i+size]
                    mnemonics = ";".join(mnemonics)
                    count = ngrams.get(mnemonics, 0) + 1
                    ngrams[mnemonics] = count

        return ngrams

    def threshold_replacement(self, dictionary: Dict[str, int], replacement: str, threshold: int = 5) -> (Dict[str, int], List[str]):
        """
        Replaces all the keys with a number of occurrences less than the given threshold with the given replacement token.
        :param dictionary: Dictionary, as returned from Preprocessor.build_dictionaries
        :param replacement: Replacement token, as str
        :param threshold: Threshold value to apply the replacement, as int
        :return: Tuple (input dictionary after applying the threshold-based replacement, list of replaced keys)
        """

        replaced = []
        ret = deepcopy(dictionary)
        for key, value in dictionary.items():
            if value <= threshold:
                replaced.append(key)

                if self.__debug:
                    print("Replacing {k} with {v} occurrences.".format(k=key,
                                                                       v=value))

                occ = ret[key]
                ret.pop(key)
                new_occ = ret.get(replacement, 0) + occ
                ret[replacement] = new_occ

        if self.__debug:
            print("Replaced {n} keys under the occurrence threshold.".format(n=len(replaced)))

        return ret

    @staticmethod
    def export_dictionary(dictionary: Dict, path: str) -> None:
        """
        Exports a dictionary to textual file containing a key per line.
        :param dictionary: Dict to export
        :param path: Path to file to write
        :return: None
        """

        with open(path, mode="w"):
            pass

        with open(path, mode="a") as file:
            for key in dictionary.keys():
                file.write("{k}\n".format(k=key))
            file.flush()

    def export_dictionaries(self, dictionaries: Dictionaries,
                            mnemonics_path: str = None, arguments_path: str = None,
                            compilers_path: str = None, optimizations_path: str = None) -> None:
        """
        Exports dictionaries to textual files containing a dictionary key per line.
        :param dictionaries: Dictionaries object to export
        :param mnemonics_path: Path to file to which the mnemonics dictionary will be exported to (optional)
        :param arguments_path: Path to file to which the arguments dictionary will be exported to (optional)
        :param compilers_path: Path to file to which the compilers dictionary will be exported to (optional)
        :param optimizations_path: Path to file to which the optimizations dictionary will be exported to (optional)
        :return: None
        """

        paths = [mnemonics_path, arguments_path, compilers_path, optimizations_path]
        for i, path in enumerate(paths):
            if path is not None:
                if self.__debug:
                    print("Writing to {path}...".format(path=path))

                if i == 0:
                    dictionary = dictionaries.mnemonics
                elif i == 1:
                    dictionary = dictionaries.arguments
                elif i == 2:
                    dictionary = dictionaries.compilers
                else:
                    dictionary = dictionaries.optimizations

                Preprocessor.export_dictionary(dictionary=dictionary,
                                               path=path)

                if self.__debug:
                    print("Done.\n")

    @staticmethod
    def import_dictionary(path: str, starting_tokens: List[str] = None) -> Dict[str, int]:
        """
        Imports a dictionary from a textual file containing a key per line.
        :param path: Path to file containing a key per line
        :param starting_tokens: List of tokens to prepend to the dictionary
        :return: Dictionary containing the starting tokens and the rest of the keys from the file.
        """

        dictionary = {}
        if starting_tokens is not None:
            for token in starting_tokens:
                dictionary[token] = len(dictionary)

        with open(path) as file:
            for line in file:
                line = line.strip("\n")
                dictionary[line] = len(dictionary)

        return dictionary

    def import_dictionaries(self,
                            mnemonics_path: str = None, arguments_path: str = None,
                            compilers_path: str = None, optimizations_path: str = None) -> Dictionaries:
        """
        Imports dictionaries from textual files.
        :param mnemonics_path: Path to file to which the mnemonics dictionary will be imported from (optional)
        :param arguments_path: Path to file to which the arguments dictionary will be imported from (optional)
        :param compilers_path: Path to file to which the compilers dictionary will be imported from (optional)
        :param optimizations_path: Path to file to which the optimizations dictionary will be imported from (optional)
        :return: Dictionaries object.
        """

        mnemonics = Preprocessor.import_dictionary(path=mnemonics_path,
                                                   starting_tokens=[self.UNK_TOKEN, self.FEW_TOKEN]) if mnemonics_path is not None else None

        arguments = Preprocessor.import_dictionary(path=arguments_path,
                                                   starting_tokens=[self.UNK_TOKEN, self.FEW_TOKEN]) if arguments_path is not None else None

        compilers = Preprocessor.import_dictionary(path=compilers_path,
                                                   starting_tokens=[self.UNK_TOKEN]) if compilers_path is not None else None

        optimizations = Preprocessor.import_dictionary(path=optimizations_path,
                                                       starting_tokens=[self.UNK_TOKEN]) if optimizations_path is not None else None

        return Dictionaries(mnemonics=mnemonics,
                            arguments=arguments,
                            compilers=compilers,
                            optimizations=optimizations)


if __name__ == "__main__":
    train_path = "../data/train_dataset.jsonl"
    k = 5
    pre = Preprocessor(train=True, debug=True)
    pre.build_ngrams(training_set=train_path)

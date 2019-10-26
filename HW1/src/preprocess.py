"""
Pre-processing utilities for the feature engineering phase.
"""
from json import JSONDecoder
from collections import namedtuple
from copy import deepcopy
from typing import Dict


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

    def threshold_replacement(self, dictionary: Dict[str, int], replacement: str, threshold: int = 5) -> Dict[str, int]:
        """
        Replaces all the keys with a number of occurrences less than the given threshold with the given replacement token.
        :param dictionary: Dictionary, as returned from Preprocessor.build_dictionaries
        :param replacement: Replacement token, as str
        :param threshold: Threshold value to apply the replacement, as int
        :return: Input dictionary, after applying the threshold-based replacement
        """

        replaced = 0
        ret = deepcopy(dictionary)
        for key, value in dictionary.items():
            if value <= threshold:
                replaced += 1

                if self.__debug:
                    print("Replacing {k} with {v} occurrences.".format(k=key,
                                                                       v=value))

                occ = ret[key]
                ret.pop(key)
                new_occ = ret.get(replacement, 0) + occ
                ret[replacement] = new_occ

        if self.__debug:
            print("Replaced {n} keys under the occurrence threshold.".format(n=replaced))

        return ret

    def regex_replacement(self, dictionary: Dict[str, int], replacement: str, regex: str) -> Dict[str, int]:
        # TODO: support replacement for keys matching the regex (i.e. brackets presence or memory addresses)
        pass

    def feature_engineering(self, dicts: Dictionaries):
        d = self.threshold_replacement(dictionary=dicts.arguments,
                                       replacement="TOKEN")


if __name__ == "__main__":
    train_path = "../data/train_dataset.jsonl"
    k = 5
    pre = Preprocessor(train=True, debug=True)
    ds = pre.build_dictionaries(training_set=train_path)

    #pre.feature_engineering(dicts=ds)

    with open("../misc/args.csv", "w") as f:
        for k, v in ds.arguments.items():
            f.write("{k},{v}\n".format(k=k, v=v))

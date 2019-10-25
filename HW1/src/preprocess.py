"""
Pre-processing utilities for the feature engineering phase.
"""
from json import JSONDecoder
from collections import namedtuple


"""
NamedTuple to model a single example in the dataset, in which:
- instructions: List containing the instructions of the function, each of them being List[str] (mnemonic, arguments)
- optimization: str encoding the optimization level that was used (or None)
- compiler: str encoding the compiler that was used (or None)
"""
Function = namedtuple("Function", "instructions optimization compiler")


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
        opt_level = json_obj["opt"] if self.__train else None
        compiler = json_obj["compiler"] if self.__train else None

        ret = Function(instructions=instr_list,
                       optimization=opt_level,
                       compiler=compiler)

        if self.__debug:
            print("Parsed:\n{fn}".format(fn=ret))

        return ret


if __name__ == "__main__":
    k = 5
    pre = Preprocessor(train=True)
    with open("../data/train_dataset.jsonl") as f:
        for line in f:
            if k == 0:
                break
            k -= 1

            res = pre.take(json_line=line)
            print("{n}. {l}\t{opt}\t{cmp}".format(n=(5-k),
                                                  l=len(res.instructions),
                                                  opt=res.optimization,
                                                  cmp=res.compiler))

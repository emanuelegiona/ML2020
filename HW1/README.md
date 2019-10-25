# HW1: Compiler provenance

Given the binary code of a function as a list of instructions,
the goal is to detect which compiler has produced it, and at which
optimization level.

For this homework, the compilers are the following:

- gg;
- icc;
- clang;

while the optimization levels are two:

- high;
- low.

## Directory structure

```
- data              # general data directory, containing training and test sets
- src               # source file directory
    |__ ...
- main.py           # Main Python script to run the homework
- README.md         # this file
```

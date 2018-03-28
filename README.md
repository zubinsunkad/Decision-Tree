# Decision Tree

A Learners guide for Decision Tree with post pruning algorithm. Best attribute can be selected using two different heuristics - Information gain heuristic, Variance impurity heuristic.

### Prerequisites and Installation

1. Python3

For windows:
* [Python Installation](https://www.python.org/downloads/)

For Mac:

Use the command:

```
brew install python3
```

2. Packages – pandas, numpy

For Windows:

* [Canopy Installation](https://store.enthought.com/downloads/)

For Mac: Canopy is not required, download packages using pip.

```
pip install pandas
```

```
pip install numpy
```

These commands assume you already have pip installed.

## Running the tests

Sample Commands: 

```
$$ python DecisionTree.py --k 1 --l 2 --train "C: \ DataSet1\training_set.csv" --validation "C:\ DataSet1\ validation_set.csv" --test "C:\DataSet1\test_set.csv" --to_print yes
```

```
$$ python DecisionTree.py --k 3 --l 4 --train "C: \ DataSet1\training_set.csv" --validation "C:\ DataSet1\ validation_set.csv" --test "C:\DataSet1\test_set.csv"
```

Following are the input functions:

--k : Used to provide k value. Has a default value 1.

--l : Used to provide l value. Has a default value 1.

--train: Provide the path to training data. Default value is in same path as program. If dataset is in same path as program, then this option is not necessary.

--validation: Provide the path to validation data. Default value is in same path as program. If dataset is in same path as program, then this option is not necessary.

--test: Provide the path to test data. Default value is in same path as program. If dataset is in same path as program, then this option is not necessary.

--to_print: If you want the tree to print, provide yes as the option. Default value is no and if anything else is provided, tree does not print.

Extra tips : If you want the output of above program to be printed in file, add ‘>> output.txt’ at end of above code.

## Results

Results were documented in the test file output1.txt and output2.txt.

## Authors

* **Zubin Sunkad**

## Acknowledgments

* [Arthur-e](https://github.com/arthur-e/Programming-Collective-Intelligence/blob/master/chapter7/treepredict.py)
* [Marius Przydatek](https://mariuszprzydatek.com/2014/10/31/measuring-entropy-data-disorder-and-information-gain/)
* [Joeya James](https://github.com/joeyajames/Python/blob/master/BinarySearchTree.py)

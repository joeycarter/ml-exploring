# Exploring Machine Learning

Exploring machine learning with Scikit-Learn, Keras, and TensorFlow. Watch me learn (and fail!) in public.

## Example

```console
$ cd ml-exploring
$ ./create-dataset.py -n 64
$ ./create-dataset.py -n 1024
$ ./01-first-neural-net.py train data/dataset.n1024.csv
$ ./01-first-neural-net.py evaluate data/dataset.n64.csv
$ ./01-first-neural-net.py predict data/dataset.n64.csv
```

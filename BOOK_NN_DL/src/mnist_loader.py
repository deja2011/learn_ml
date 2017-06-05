"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.
One hot-encoding is not included in load_data(). It is supposed to be performed
within learning models rather than in data loading utilities.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training inputs,
    training results, validation inputs, validation results, test inputs,
    and test data.

    The ``training_inputs`` is returned as a 2-D array of shape 50,000 x 784.

    The ``training_results`` is returned as a 1-D array of shape 50,000.

    The ``validation_*`` and ``test_*`` variables are similar to ``training_*``
    variables, except that they only have 10,000 pairs of inputs and results.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    training_inputs, training_results = \
            [np.array(t) for t in training_data]
    validation_inputs, validation_results = \
            [np.array(t) for t in validation_data]
    test_inputs, test_results = \
            [np.array(t) for t in test_data]
    return (training_inputs, training_results), \
            (validation_inputs, validation_results), \
            (test_inputs, test_results)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

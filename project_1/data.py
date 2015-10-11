# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
from sklearn.utils import check_random_state


def make_data(n_samples=500, noise_std=0.5, random_state=None):
    """ Generate an artificial binary classification task

    Parameters
    ----------
    n_samples : int, optional (default=500)
        Number of samples of the dataset

    noise_std : float, optional (default=0.2)
        A Gaussian noise with zero mean and noise_std standard deviation
        is added to the features

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    """
    random_state = check_random_state(random_state)

    y = random_state.randint(0, 2, size=n_samples)
    X = random_state.normal(scale=noise_std, size=(n_samples, 2))

    angle = random_state.uniform(-np.pi / 8, 4 * np.pi, size=n_samples)
    X[:, 0] += angle * np.cos(angle) * (2 * y - 1)
    X[:, 1] += angle * np.sin(angle) * (2 * y - 1)

    return X, y


if __name__ == "__main__":
    from numpy.testing import assert_equal
    from numpy.testing import assert_array_almost_equal

    # Ensure that the generated dataset has proper shapes
    X, y = make_data(n_samples=10, random_state=0)
    assert_equal(X.shape, (10, 2))
    assert_equal(y.shape, (10, ))

    # Check reproducibility
    X_2, y_2 = make_data(n_samples=10, random_state=0)
    assert_array_almost_equal(X, X_2)
    assert_array_almost_equal(y, y_2)

    # Display dataset (uncomment if needed)
    import matplotlib.pyplot as plt
    X, y = make_data(random_state=0)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50, cmap=plt.cm.cool)
    plt.savefig('artificial_data.png')
    plt.close()

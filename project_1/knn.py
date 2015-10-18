#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

from itertools import izip

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data
from plot import plot_boundary

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    class TrainingSample:
        def __init__(self, X, y):
            """One training sample with its expected class.

            Parameters
            ----------
            X: array-like, shape = [n_features]
                The sample.
            y: the class of the sample

            """

            self.X = x
            self.y = y

        def dist(self, X):
            """The distance the training sample and a set of features composing
            a predicted sample.

            Parameters
            ----------
            X: array-like of shape = [n_features]

            Returns
            -------
            The sum of the square of the differences of the respective features
            of the training sample and the predicted sample.
            """

            if self.X.shape != X.shape:
                raise ValueError(
                    "The two samples must have the same number of features"
                )

            return sum(
                (feature1 - feature2)**2
                for feature1, feature2 in izip(self.X, X)
            )

    def __init__(self, n_neighbors=1):
        """K-nearest classifier classifier

        Parameters
        ----------
        n_neighbors: int, optional (default=1)
            Number of neighbors to consider.

        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit a k-nn model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Parameter validation
        if self.n_neighbors < 0:
            raise ValueError("n_neighbors muster greater than 0, got %s"
                             % self.neighbors)

        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # Puts the training input samples into a Python list of TrainingSamples.
        # Makes the tracking of the target output value ('y') easier when
        # sorting the 'n_neighbors' nearest neighbors in the prediction
        # algorithms.
        self.train_set = [
            TrainingSample(sample, target) for sample, target in izip(X, y)
        ]

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.empty(X.shape[0]) # Output

        for i, sample in enumerate(X):
            #
            # Finds the 'n_neighbors' nearest neighbors by sorting them.
            #

            def cmp_training_samples(training_sample1, training_sample2):
                return cmp(
                    training_sample1.dist(sample), training_sample2.dist(sample)
                ) 

            nearests = sorted(
                self.train_set, cmp=cmp_training_samples
            )[:self.n_neighbors]

            #
            # Selects the majority class in the nearest samples.
            #

            classes = {}
            for nearest in nearests:
                if nearest.y in classes:
                    classes[nearest.y] += 1
                else:
                    classes[nearest.y] = 1

            y[i] = max(classes.iteritems(), key=lambda (y, n): n)[0]

        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # TODO your code here.
        pass

if __name__ == "__main__":
    # (Question 2): K-nearest-neighbors
    # TODO your code here

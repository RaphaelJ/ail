#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import math

import numpy as np

from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import cross_val_score, train_test_split
import sklearn.neighbors as skl

from data import make_data
from plot import plot_boundary

class TrainingSample:
    def __init__(self, X, y):
        """One training sample with its expected class.

        Parameters
        ----------
        X: array-like, shape = [n_features]
            The sample.
        y: the class of the sample

        """

        self.X = X
        self.y = y

    def dist(self, X):
        """The Eclidian distance between the training sample and a set of
        features composing a sample to predict.

        Parameters
        ----------
        X: array-like of shape = [n_features]

        Returns
        -------
        The Euclidian distance.
        """

        if self.X.shape != X.shape:
            raise ValueError(
                "The two samples must have the same number of features"
            )

        # Remarks: as we only use this distance to sort the learning samples
        #          by their distance to a sample to predict, we could get
        #          rid of the 'sqrt()' call.

        return math.sqrt(
            sum(
                (feature1 - feature2)**2
                for feature1, feature2 in zip(self.X, X)
            )
        )

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1):
        """K-nearest classifier classifier.

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
            TrainingSample(sample, target) for sample, target in zip(X, y)
        ]

        # Collects all the different classes by lexicographic order.
        self.classes_ = sorted(set(y))

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

        #
        # Selects the class returned by 'predict_proba()' with the highest
        # probability.
        #

        # Indexes (in 'classes_') of the classes with the highest probability,
        # for each sample.
        class_ixs = np.argmax(self.predict_proba(X), axis=1)

        return np.array([self.classes_[ix] for ix in class_ixs])

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            Test samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order. The order of the classes corresponds to that
            in the attribute classes_.
        """

        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.empty((X.shape[0], len(self.classes_))) # Output

        for i, sample in enumerate(X):
            #
            # Finds the 'n_neighbors' nearest neighbors by sorting them by their
            # distance to 'sample'.
            #

            nearests = sorted(
                self.train_set, key=lambda ts: ts.dist(sample)
            )[:self.n_neighbors]

            #
            # Counts the number of neighbors for each class.
            #

            classes_count = dict((c, 0) for c in self.classes_)
            for nearest in nearests:
                assert nearest.y in classes_count
                classes_count[nearest.y] += 1

            #
            # Copies the class neighbor count into the 'y' output, in
            # lexicographic order regarding their label.
            #

            # Sorts classes by the lexicographic order of their label.
            sorted_classes_count = sorted(
                classes_count.items(), key=lambda i: i[0]
            )

            # Removes the class labels and copies them in the output matrix.
            y[i, :] = np.array([v for k, v in sorted_classes_count])

        return y

if __name__ == "__main__":
    # (Question 2): K-nearest-neighbors

    N_SAMPLES  = 2000
    N_TRAINING = 150
    RANDOM_STATE = 1

    MAX_N_NEIGHBORS = 20

    X, y = make_data(N_SAMPLES, random_state=RANDOM_STATE)

    # Splits the training and testings sets randomly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_TRAINING, random_state=RANDOM_STATE
    )

    assert (len(X_train) == N_TRAINING)
    assert (len(y_train) == N_TRAINING)
    assert (len(X_test)  == N_SAMPLES - N_TRAINING)
    assert (len(y_test)  == N_SAMPLES - N_TRAINING)

    scores_test = []
    scores_train = []
    scores_ten_fold = []

    for n_neighbors in range(1, MAX_N_NEIGHBORS + 1):
        # Uncomment the following line to use our KNN classifier instead of the
        # one provided by scikit-learn. Both give the same results, but the one
        # provided by scikit-learn is significantly faster.

        #knc = KNeighborsClassifier(n_neighbors=n_neighbors)
        knc = skl.KNeighborsClassifier(n_neighbors=n_neighbors)

        # Trains the classifier on a simple train/test split of the data.
        knc.fit(X_train, y_train)

        score_test = knc.score(X_test, y_test)
        score_train = knc.score(X_train, y_train)

        scores_test.append(score_test)
        scores_train.append(score_train)

        # Computes the score using a 10-fold cross-validation.
        score_ten_fold = cross_val_score(knc, X, y, cv=10).mean()
        scores_ten_fold.append(score_ten_fold)

        print(
            "Neighs.: {} Score (test): {} Score (train): {} Score (10-fold): {}"
                .format(n_neighbors, score_test, score_train, score_ten_fold)
        )

        plot_boundary(
            "knn_out/knn_{n_neighbors}".format(n_neighbors=n_neighbors), knc,
            X_test, y_test
        )

    plt.figure()
    plt.title("K-nearest neighbors error rate")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Error")
    plt.ylim(0, 1)
    depth_range = range(1, len(scores_test) + 1)
    plt.plot(
        depth_range, [1 - s for s in scores_test], 'r', label="Testing set"
    )
    plt.plot(
        depth_range, [1 - s for s in scores_train], 'g', label="Training set"
    )
    plt.plot(
        depth_range, [1 - s for s in scores_ten_fold], 'b',
        label="10-fold cross-validation"
    )

    def find_best_score(scores):
        """
        Given a list of scores, returns the best score, and the depth at which
        it was achieved.
        """
        best_score = max(scores)
        best_score_depth = scores.index(best_score) + 1
        return best_score, best_score_depth

    best_test_score, best_test_depth = find_best_score(scores_test)
    plt.plot(best_test_depth, 1 - best_test_score, "ro")

    best_ten_fold_score, best_ten_fold_depth = find_best_score(scores_ten_fold)
    plt.plot(best_ten_fold_depth, 1 - best_ten_fold_score, "bo")

    plt.legend()

    plt.savefig("knn_out/Scores.pdf")
    plt.close()


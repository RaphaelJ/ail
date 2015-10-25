#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import RidgeClassifier
from sklearn.cross_validation import train_test_split

from data import make_data
from plot import plot_boundary

def to_polar(X, y):
    """ Transforms the samples in polar coordinates, using arctan2.

    Parameters
    ----------
    X : array-like, shape = [n_samples, 2]
        The training input samples.

    y : array-like, shape = [n_samples]
        The target values.

    Returns
    -------
    X : array-like, shape = [n_samples, 2]
        The training input samples, in polar cooardinates.
    """

    # First we need to compute the polar cooardinates (angle and distance) for
    # every sample.
    X_angle = np.arctan2(X[:, 1], X[:, 0])
    X_dist = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    # Polar cooardinates (first colum: angle, second column: distance).
    X_polar = np.array([X_angle, X_dist]).T

    # Now that we have the samples in polar coordinates, we must adjust the
    # angle (which ranges between [-pi, pi]) so it takes into account when the
    # spiral wraps around 'pi' to '-pi'.
    #
    # We do this by sorting the samples by their distance to the origin, and by
    # detecting when the angle move from the second quadrant to the third
    # (accounting for a new turn), or the reverse (accounting for a turn
    # backward). As there are two spirals (one for each sample class), we apply
    # the transformation separatly for each class.

    # Sorts samples by distance.
    sorted_ixs = np.argsort(X_dist)

    def quadrant(angle):
        """Returns the quadrant (between 1 and 4) of an angle (between -pi and
        pi)."""

        if angle >= 0:
            if angle <= np.pi / 2:
                return 1
            else:
                return 2
        else:
            if angle <= - np.pi / 2:
                return 3
            else:
                return 4

    def adjust_angles(class_):
        prec_angle_quadrant = None
        n_turns = 0
        for ix in sorted_ixs:
            if y[ix] != class_:
                continue

            angle_quadrant = quadrant(X_polar[ix, 0])
            new_angle = X_polar[ix, 0]
            
            if prec_angle_quadrant != None:
                if prev_angle >= 0 and new_angle <= 0 and prev_angle - new_angle > np.pi:
                    n_turns += 1
                elif prev_angle <= 0 and new_angle >= 0 and new_angle - prev_angle > np.pi:
                    n_turns -= 1

            prev_angle = X_polar[ix, 0]
            prec_angle_quadrant = angle_quadrant

            X_polar[ix, 0] += n_turns * 2 * np.pi

    adjust_angles(class_=1)
    adjust_angles(class_=0)

    return X_polar


if __name__ == "__main__":
    # (Question 3) linear model

    N_SAMPLES = 2000
    N_TRAINING = 150
    RANDOM_STATE = 0
    for RANDOM_STATE in range(0, 10):

        X, y = make_data(N_SAMPLES, random_state=RANDOM_STATE)

        # Comment this line to not convert the sample into polar coordinates
        # before applying the linear model.
        X = to_polar(X, y)

        # Splits the training and testings sets randomly.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=N_TRAINING, random_state=RANDOM_STATE
        )

        assert (len(X_train) == N_TRAINING)
        assert (len(y_train) == N_TRAINING)
        assert (len(X_test)  == N_SAMPLES - N_TRAINING)
        assert (len(y_test)  == N_SAMPLES - N_TRAINING)

        rc = RidgeClassifier(alpha = 0)

        # Trains the classifier on a simple train split of the data.
        rc.fit(X_train, y_train)
        y_predicted = rc.predict(X_test)

        plot_boundary(
            "lin_out/test_classes_Rand{}".format(RANDOM_STATE), rc, X_test,
             y_test, title = "Test sample classes (reference)"
        )

        plot_boundary(
            "lin_out/predict_test_Rand{}".format(RANDOM_STATE), rc, X_test,
             y_predicted, title = 
                        "Linear model classifier prediction on the test sample"
        )

        score_test = rc.score(X_test, y_test)
        score_train = rc.score(X_train, y_train)

        print(
            "Random_state = {} Linear model: Score (test): {} Score (train): {}"
            .format(RANDOM_STATE, score_test, score_train)
        )

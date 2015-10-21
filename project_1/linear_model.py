#! /usr/bin/env python
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

def dist(x,y):   
    return np.sqrt((x)**2 + (y)**2)

class PolarCoordsSample:
    def __init__(self, angle, distance):
        self.angle = angle
        self.distance = distance

if __name__ == "__main__":
    # (Question 3) ols.py: Ordinary least square
    
    
    N_SAMPLES  = 2000
    N_TRAINING = 150
    RANDOM_STATE = 1

    X, y = make_data(N_SAMPLES, random_state=RANDOM_STATE)

    # Splits the training and testings sets randomly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_TRAINING, random_state=RANDOM_STATE
    )

    assert (len(X_train) == N_TRAINING)
    assert (len(y_train) == N_TRAINING)
    assert (len(X_test)  == N_SAMPLES - N_TRAINING)
    assert (len(y_test)  == N_SAMPLES - N_TRAINING)

    y_predicted = []
    rc = RidgeClassifier(alpha = 0)

    # Trains the classifier on a simple train split of the data.
    rc.fit(X_train, y_train)
    y_predicted = rc.predict(X_test)
    
    plot_boundary(
            "lin_out/lin_{predicted}", rc, X_test, y_predicted
        )
    plot_boundary(
            "lin_out/lin_{real}", rc, X_test, y_test
        )

    score_test = rc.score(X_test, y_test)
    score_train = rc.score(X_train, y_train)

    print("Linear model: Score (test): {} Score (train): {} "
                .format(score_test, score_train)
        )

    # Transformation of the input data
    angles_b = []
    distances_b = []
    angles_r = []
    distances_r = []

    for i in range(0, len(X_train)):
        if y_train[i] == 0:
            angle = np.arctan2(X_train[i, 0], X_train[i, 1])
            angles_r.append(angle)
            distance = dist(X_train[i, 0], X_train[i, 1])
            distances_r.append(distance)
        if y_train[i] == 1:
            angle = np.arctan2(X_train[i, 0], X_train[i, 1])
            angles_b.append(angle)
            distance = dist(X_train[i, 0], X_train[i, 1])
            distances_b.append(distance)

    polar_coords_samples_b = []       
    for i in range(0, len(angles_b)):
        polar_coords_samples_b.append(
            PolarCoordsSample(angles_b[i], distances_b[i])
        )

    polar_coords_samples_b.sort(key=lambda pcs: pcs.distance)
    
    angles_b_sort = []
    distances_b_sort = []
    previous_angle = polar_coords_samples_b[0].angle
    for i in range(0, len(angles_b)):
        angle =  polar_coords_samples_b[i].angle 
        if previous_angle + 3*np.pi < angle:   
            angle = angle - 4*np.pi
        elif previous_angle + np.pi < angle:   
            angle = angle - 2*np.pi
        
        angles_b_sort.append(angle) 
        distance =  polar_coords_samples_b[i].distance  
        distances_b_sort.append(distance) 
        previous_angle = angle
        
    
    plt.figure()
    plt.title("plot")
    plt.ylabel("angle")
    plt.xlabel("distance")
    
    plt.plot(distances_b_sort,angles_b_sort, "o" )
    
    """
    plt.plot(distances_b, angles_b, "o" )
    plt.plot(distances_r, angles_r, "ro")
    """


    plt.savefig("lin_out/angle_dist.pdf")
    plt.close()
    


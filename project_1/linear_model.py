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
    
    # Plot of Q3.1 results
    plot_boundary(
            "lin_out/lin_test_predicted", rc, X_test, y_predicted, 
            title = "Prediction of the linear model classifier on the test sample"
        )
    plot_boundary(
            "lin_out/lin_test", rc, X_test, y_test, 
            title = "Linear model classifier on test sample"
        )

    score_test = rc.score(X_test, y_test)
    score_train = rc.score(X_train, y_train)

    print("Linear model: Score (test): {} Score (train): {} "
                .format(score_test, score_train)
        )
    # ---------------------------Train Transformation---------------------------
    # Transformation of the input train data
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

    """ The arctan gives angles between [-pi, pi]. We have to find a way to 
    count turns. 
    We are going to create a vector classified thanks to the distance between 
    the center and coordonates and see when there is a "junp" in angle values.
    We replicate the process for both events on the training set. """
    
    # Create pairs distance-angle
    polar_coords_samples_b = []       
    for i in range(0, len(angles_b)):
        polar_coords_samples_b.append(
            PolarCoordsSample(angles_b[i], distances_b[i])
        )

    # Sort by distance
    polar_coords_samples_b.sort(key=lambda pcs: pcs.distance)
    
    # Verification of angles "jumps"
    angles_b_sort = []
    distances_b_sort = []
    previous_angle = polar_coords_samples_b[0].angle
    for i in range(0, len(angles_b)):
        angle =  polar_coords_samples_b[i].angle 
        # If jump of two turns
        if previous_angle + 3.2*np.pi < angle:   
            angle = angle - 4*np.pi
        # if jump of one turn
        elif previous_angle + 1.2*np.pi < angle:   
            angle = angle - 2*np.pi
        
        angles_b_sort.append(angle) 
        distance =  polar_coords_samples_b[i].distance  
        distances_b_sort.append(distance) 
        previous_angle = angle
        
    
    # Create second pairs distance-angle
    polar_coords_samples_r = []       
    for i in range(0, len(angles_r)):
        polar_coords_samples_r.append(
            PolarCoordsSample(angles_r[i], distances_r[i])
        )

    # Sort by distance
    polar_coords_samples_r.sort(key=lambda pcs: pcs.distance)
    
    # Verification of angles "jumps"
    angles_r_sort = []
    distances_r_sort = []
    previous_angle = polar_coords_samples_r[0].angle
    for i in range(0, len(angles_r)):
        angle =  polar_coords_samples_r[i].angle 
        # If jump of two turns
        if previous_angle + 3.2*np.pi < angle:   
            angle = angle - 4*np.pi
        # if jump of one turn
        elif previous_angle + 1.2*np.pi < angle:   
            angle = angle - 2*np.pi
        
        angles_r_sort.append(angle) 
        distance =  polar_coords_samples_r[i].distance  
        distances_r_sort.append(distance) 
        previous_angle = angle
        
    angles_sort = np.append(angles_b_sort, angles_r_sort)
    distances_sort = np.append(distances_b_sort, distances_r_sort)
    new_X_train = np.array([angles_sort, distances_sort]).transpose()
    new_y_train = np.zeros(len(y_train), dtype=np.int)
    new_y_train[0:len(angles_b_sort)] = 1

    # ---------------------------Test Transformation---------------------------
    # Transformation of the input test data 
    test_angles_b = []
    test_distances_b = []
    test_angles_r = []
    test_distances_r = []

    for i in range(0, len(X_test)):
        if y_test[i] == 0:
            angle = np.arctan2(X_test[i, 0], X_test[i, 1])
            test_angles_r.append(angle)
            distance = dist(X_test[i, 0], X_test[i, 1])
            test_distances_r.append(distance)
        if y_test[i] == 1:
            angle = np.arctan2(X_test[i, 0], X_test[i, 1])
            test_angles_b.append(angle)
            distance = dist(X_test[i, 0], X_test[i, 1])
            test_distances_b.append(distance)

    """ The arctan gives angles between [-pi, pi]. We have to find a way to 
    count turns. 
    We are going to create a vector classified thanks to the distance between 
    the center and coordonates and see when there is a "junp" in angle values.
    We replicate the process for both events on the training set. """
    
    # Create pairs distance-angle
    test_polar_coords_samples_b = []       
    for i in range(0, len(test_angles_b)):
        test_polar_coords_samples_b.append(
            PolarCoordsSample(test_angles_b[i], test_distances_b[i])
        )

    # Sort by distance
    test_polar_coords_samples_b.sort(key=lambda pcs: pcs.distance)
    
    # Verification of angles "jumps"
    test_angles_b_sort = []
    test_distances_b_sort = []
    previous_angle = test_polar_coords_samples_b[0].angle
    for i in range(0, len(test_angles_b)):
        angle =  test_polar_coords_samples_b[i].angle 
        # If jump of two turns
        if previous_angle + 3.2*np.pi < angle:   
            angle = angle - 4*np.pi
        # if jump of one turn
        elif previous_angle + 1.2*np.pi < angle:   
            angle = angle - 2*np.pi
        
        test_angles_b_sort.append(angle) 
        distance =  test_polar_coords_samples_b[i].distance  
        test_distances_b_sort.append(distance) 
        previous_angle = angle
        
    
    # Create second pairs distance-angle
    test_polar_coords_samples_r = []       
    for i in range(0, len(test_angles_r)):
        test_polar_coords_samples_r.append(
            PolarCoordsSample(test_angles_r[i], test_distances_r[i])
        )

    # Sort by distance
    test_polar_coords_samples_r.sort(key=lambda pcs: pcs.distance)
    
    # Verification of angles "jumps"
    test_angles_r_sort = []
    test_distances_r_sort = []
    previous_angle = test_polar_coords_samples_r[0].angle
    for i in range(0, len(test_angles_r)):
        angle =  test_polar_coords_samples_r[i].angle 
        # If jump of two turns
        if previous_angle + 3.2*np.pi < angle:   
            angle = angle - 4*np.pi
        #if jump of one turn
        elif previous_angle + 1.2*np.pi < angle:   
            angle = angle - 2*np.pi
        
        test_angles_r_sort.append(angle) 
        distance =  test_polar_coords_samples_r[i].distance  
        test_distances_r_sort.append(distance) 
        previous_angle = angle
        
    test_angles_sort = np.append(test_angles_b_sort, test_angles_r_sort)
    test_distances_sort = np.append(test_distances_b_sort, test_distances_r_sort)
    new_X_test = np.array([test_angles_sort, test_distances_sort]).transpose()
    new_y_test = np.zeros(len(y_test), dtype=np.int)
    new_y_test[0:len(test_angles_b_sort)] = 1

    # Creation of the classiffier
    new_rc = RidgeClassifier(alpha = 0)
    new_y_predicted = []

    # Trains the classifier on a simple train split of the data.
    new_rc.fit(new_X_train, new_y_train)
    new_y_predicted = new_rc.predict(new_X_test)

    new_score_train = new_rc.score(new_X_train, new_y_train)
    new_score_test = new_rc.score(new_X_test, new_y_test)

    # Plot of Q3.2-Q3.3 results after transformation
    plot_boundary(
            "lin_out/New_lin_train", new_rc, new_X_train, new_y_train,
            title="Linearization on train sample"
        )
    plot_boundary(
            "lin_out/New_lin_test_predicted", new_rc, new_X_test, new_y_test,
            title="Linearization on test sample"
        )
    
   

    print("Linear model: New Score (test): {} New Score (train): {} "
                .format(new_score_test, new_score_train)
        )
   
    plt.figure()
    plt.subplot(121)
    plt.title("Test sample")
    plt.ylabel("angle")
    plt.xlabel("distance")
    plt.plot(test_distances_b_sort,test_angles_b_sort, "o" )
    plt.plot(test_distances_r_sort,test_angles_r_sort, "ro" )

    plt.subplot(122)
    plt.title("Training sample")
    plt.xlabel("distance")
    plt.plot(distances_b_sort,angles_b_sort, "o" )
    plt.plot(distances_r_sort,angles_r_sort, "ro" )

    plt.savefig("lin_out/angle_distance_transformation.pdf")
    plt.close()
    


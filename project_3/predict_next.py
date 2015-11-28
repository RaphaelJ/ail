#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Authors: Maxime Javaux    <maximejavaux@hotmail.com>
#          Raphael Javaux   <raphaeljavaux@gmail.com>

"""
Trains a decision tree regressor to predict a position at T + 1 from a position
at T. Predicts the given testing dataset and writes the output on 'stdout'.
"""

import argparse, csv

import numpy as np
from sklearn.cross_validation import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from dataset import load_dataset

DT_MAX_DEPTH = 15
KNN_NEIGHBORS = 5
RANDOM_STATE = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(__doc__))

    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--test', type=str)

    args = parser.parse_args()
    train_path = args.train_dataset
    test_path = args.test

    dataset = list(
        sample
        for sample in load_dataset(train_path)
        if len(sample.trip) > 2
    )

    train, test = train_test_split(
        dataset, test_size=0.3, random_state=RANDOM_STATE
    )

    #for depth in range(1, 30):
    for neighs in range(14, 15):

        train_X = np.array([
            np.concatenate((
            #[sample.as_numpy_arr[3]], [sample.as_numpy_arr[5]],
            #[sample.as_numpy_arr[6]], [sample.as_numpy_arr[7]],
            #[sample.as_numpy_arr[8]],
            #[sample.as_numpy_arr[7]], [sample.as_numpy_arr[10]],
            sample.trip_as_numpy_arr[0],
            sample.trip_as_numpy_arr[
                np.random.randint(0, len(sample.trip_as_numpy_arr))
              ]
            ))
            for sample in train
        ])

        train_y = np.array([
            sample.trip_as_numpy_arr[-1]
            for sample in train
        ])

        def weighting_func(dists):
            return np.ones(dists.shape)

        #regr = DecisionTreeRegressor(max_depth=depth)
        regr = KNeighborsRegressor(n_neighbors=neighs, weights=weighting_func)

        # Fitting on the training set
        regr.fit(train_X, train_y)

        if test_path is None:
            # Computes and prints the score on testing samples from the training
            # dataset.

            # Randomly removes positions in the test case paths.

            test_X = np.array([
                np.concatenate((
                #[sample.as_numpy_arr[3]], [sample.as_numpy_arr[5]],
                #[sample.as_numpy_arr[6]], [sample.as_numpy_arr[7]],
                #[sample.as_numpy_arr[8]], 
                #[sample.as_numpy_arr[7]], [sample.as_numpy_arr[10]],
                sample.trip_as_numpy_arr[0],
                sample.trip_as_numpy_arr[
                    np.random.randint(0, len(sample.trip_as_numpy_arr))
                  ]
                ))
                for sample in test
            ])

            train_y = np.array([
                sample.trip_as_numpy_arr[-1]
                for sample in test
            ])
            
            predicted = regr.predict(test_X)
            
            score = np.mean( np.abs( train_y - predicted ) ) 
            #print('depth: {} Score: {}'.format( depth, score ) )
            print('neighs: {} Score: {}'.format( neighs, score ) )
        else:
            # Predicts the provided testing dataset.

            print('"TRIP_ID","LATITUDE","LONGITUDE"')

            for sample in load_dataset(test_path):
                trip = sample.trip_as_numpy_arr
                predicted = regr.predict(np.concatenate([trip[0], trip[-1]]))[0]

                print('"{}",{},{}'.format(
                    sample.trip_id, predicted[0], predicted[1])
                )

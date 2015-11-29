#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Authors: Maxime Javaux    <maximejavaux@hotmail.com>
#          Raphael Javaux   <raphaeljavaux@gmail.com>

"""
Provides a script to finds cluster of coordinates from a set dataset, using the
K-means algorithm. Prints the computed clusters coordinates on 'stdout', one
line per cluster with it latitude and longitude.
"""

import argparse

from sys import stderr

import numpy as np

from sklearn.cluster import KMeans

from dataset import load_dataset

def load_clusters(filepath):
    pass

if __name__ == '__main__':
    #
    # Loads the dataset from the given input file.
    #

    parser = argparse.ArgumentParser(description=(__doc__))

    parser.add_argument('dataset', type=str)
    parser.add_argument('n_clusters', type=int)
    parser.add_argument(
        '--n_jobs', type=int, default=-1,
        help='The number of jobs to use for the computation. This works by '
             'computing each of the n_init runs in parallel. If -1 all CPUs '
             'are used.'
    )
    parser.add_argument(
        '--precomp', action='store_true', default=False,
        help='Precompute distances (faster but takes more memory).'
    )
    parser.add_argument(
        '--begin', action='store_const', const='begin', dest='coord_type',
        help='Only use the first coordinates of the trips (default: use all)'
    )
    parser.add_argument(
        '--end', action='store_const', const='end', dest='coord_type',
        help='Only use the last coordinates of the trips (default: use all)'
    )

    args = parser.parse_args()
    filepath = args.dataset
    n_clusters = args.n_clusters
    n_jobs = args.n_jobs
    precompute_distances = args.precomp
    coord_type = args.coord_type

    #
    # Collects all the coordinates and groups them in 'n_clusters' using
    # K-means.
    #

    samples = (
        sample for sample in load_dataset(filepath) if len(sample.trip) > 1
    )

    if coord_type == 'begin':
        # Collect only the first coordinates.
        coords = np.array([sample.trip_vec[0] for sample in samples])
    elif coord_type == 'end':
        coords = np.array([sample.trip_vec[-1] for sample in samples])
    else:
        # Collect all the coordinates.
        coords = np.concatenate([sample.trip_vec for sample in samples])

    print('Clustering on {} coordinates'.format(coords.shape[0]), file=stderr)

    km = KMeans(
        n_clusters, n_jobs=n_jobs, precompute_distances=precompute_distances
    )
    km.fit(coords)

    for center in km.cluster_centers_:
        print('{},{}'.format(center[0], center[1]))

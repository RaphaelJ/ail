#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Authors: Maxime Javaux    <maximejavaux@hotmail.com>
#          Raphael Javaux   <raphaeljavaux@gmail.com>

"""Script that strip a training dataset to the given number of samples."""

import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Strips a training dataset to the given number of samples. '
            'The new dataset is generated on stdout.'
        )
    )

    parser.add_argument('filepath', type=str, help='the input dataset')
    parser.add_argument(
        'n_samples', type=int, help='number of samples in the generated dataset'
    )

    args = parser.parse_args()
    filepath = args.filepath
    n_samples = args.n_samples

    # Reads the dataset a first time to count the number of samples.
    with open(filepath) as f:
        # Ignores the header line
        f.__next__()

        n_samples_input = 0
        for line in f:
            n_samples_input += 1

    assert n_samples_input >= n_samples

    # Generates a random distribution of samples indexes to keep.
    to_kept = set(np.random.choice(n_samples_input, n_samples, replace=False))

    # Gnerates the output dataset.
    with open(filepath) as f:
        # Copies the header line.
        print(f.__next__(), end='')

        for i, line in enumerate(f):
            if i in to_kept:
                print(line, end='')

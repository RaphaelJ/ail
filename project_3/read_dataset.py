"""Provide a function to read a CSV dataset from a file."""

import ast, csv, datetime

import numpy as np

def read_sample(sample):
    assert len(sample) == 9

    if sample[7] != 'False': # Ignores missing datasets
        return None

    assert sample[1] in ['A', 'B', 'C']
    call_type = ord(sample[1])

    if call_type == ord('A'):
        origin_call = float(sample[2])
    else:
        origin_call = -1

    if call_type == ord('B') and sample[3] != '':
        origin_stand = float(sample[3])
    else:
        origin_stand = -1

    taxi_id = float(sample[4])

    # Decompozes the timestamp in a date and a time.
    dt = datetime.datetime.fromtimestamp(round(float(sample[5])))

    year = dt.year
    month = dt.month
    day = dt.day
    weekday = dt.weekday()
    hour = dt.hour
    minute = dt.minute

    assert sample[6] in ['A', 'B', 'C']
    day_type = ord(sample[6])

    # Decomposes the trip in begin and end coordinates.
    trip = ast.literal_eval(sample[8])

    if len(trip) < 2:
        return None

    #start_long = trip[0][0]
    #start_lat = trip[0][1]
    start_long = trip[-2][0]
    start_lat = trip[-2][1]

    end_long = trip[-1][0]
    end_lat = trip[-1][1]

    return [
        call_type, origin_call, origin_stand, taxi_id, year, month, day,
        weekday, hour, minute, day_type, start_long, start_lat, end_long,
        end_lat
    ]

def read_dataset(filepath):
    samples = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)

        # Skips the header
        reader.__next__()

        for line in reader:
            sample = read_sample(line)
            if sample != None:
                samples.append(sample)

    return np.array(samples)

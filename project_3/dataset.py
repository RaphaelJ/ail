#-*- coding: utf-8 -*-
# Authors: Maxime Javaux    <maximejavaux@hotmail.com>
#          Raphael Javaux   <raphaeljavaux@gmail.com>

"""
Provides a function to read and interpret samples fomr a CSV file dataset.
"""

import ast, collections, csv, datetime

import numpy as np

Coordinates = collections.namedtuple('Coordinates', ['lat', 'lon'])

class Sample:
    """
    Contains a raw sample from the CSV file.

    Fields
    ------
    trip_id:    (str) Contains an identifier for each trip (be careful, some id
                are not unique).
    call_type:  (char) Identifies the way used to demand this service. It 
                contain one of three possible values:
                    - 'A' if this trip was dispatched from the central;
                    - 'B' if this trip was demanded directly to a taxi driver on
                      a specific stand;
                    - 'C' otherwise (i.e. a trip demanded on a random street).
    origin_call:
                (int) Contains an unique identifier for each phone number which
                was used to demand, at least, one service. It identifies the
                trip's customer if call_type=='A'. Otherwise, it will be a
                'None'.
    origin_stand:
                (int) Contains an unique identifier for the taxi stand. It
                identifies the starting point of the trip if call_type=='B'.
                Otherwise, it assumes a 'None' value;
    taxi_id:    (int) Contains an unique identifier for the taxi driver that
                performed each trip;
    start_time:
                (datetime) Identifies the trip's start;
    day_type:
                (char) Identifies the daytype of the trip’s start. It assumes
                one of three possible values:
                    - 'B' if this trip started on a holiday or any other special
                       day (i.e. extending holidays, floating holidays, etc.);
                    - 'C' if the trip started on a day before a type-B day;
                    - 'A' otherwise (i.e. a normal day, workday or weekend).

    missing_data:
                (bool) is 'False' when the GPS data stream is complete and
                'True' whenever one (or more) locations are missing in 'trips'
    trip:       (list of Coordinates) list of GPS coordinates of the taxi
                trip.
    begin:      (Coordinates) 'trip[0]' or 'None' if the trip is empty.
    end:        (Coordinates) 'trip[-1]' or 'None' if the trip is empty.
    """

    def __init__(self, csv_line):
        """Construct a sample from a CSV line of the dataset."""
        assert len(csv_line) == 9

        self.trip_id = csv_line[0]

        assert csv_line[1] in ['A', 'B', 'C']
        self.call_type = csv_line[1]

        if self.call_type == 'A':
            self.origin_call = round(float(csv_line[2]))
        else:
            self.origin_call = None

        if self.call_type == 'B' and csv_line[3] != '':
            self.origin_stand = round(float(csv_line[3]))
        else:
            self.origin_stand = None

        self.taxi_id = round(float(csv_line[4]))

        self.start_time = datetime.datetime.fromtimestamp(
            round(float(csv_line[5]))
        )

        assert csv_line[6] in ['A', 'B', 'C']
        self.day_type = ord(csv_line[6])

        self.missing_data = (csv_line[7] == 'True')

        # Decomposes the trip in begin and end coordinates.
        self.trip = [
            Coordinates(lat=lat, lon=lon)
            for lon, lat in ast.literal_eval(csv_line[8])
        ]

    @property
    def begin(self):
        if len(self.trip) < 1:
            return None
        else:
            return self.trip[0]

    @property
    def end(self):
        if len(self.trip) < 1:
            return None
        else:
            return self.trip[-1]

    #
    # Export the sample as a Numpy vector
    #

    # Contains all the feature names that can be used with 'arr()'.
    features = [
        'call_type', 'origin_call', 'origin_stand', 'taxi_id',
        'start_time.year', 'start_time.month', 'start_time.day',
        'start_time.weekday', 'start_time.hour', 'start_time.minute',
        'day_type',
    ]

    # Used to converts the feature name into 
    _feature_getters = {
        'call_type': lambda self: self.call_type,
        'origin_call': lambda self: self.origin_call,
        'origin_stand': lambda self: self.origin_stand,
        'start_time.year': lambda self: self.start_time.year,
        'start_time.month': lambda self: self.start_time.month,
        'start_time.day': lambda self: self.start_time.day,
        'start_time.weekday': lambda self: self.start_time.weekday(),
        'start_time.hour': lambda self: self.start_time.month,
        'start_time.minute': lambda self: self.start_time.day,
        'day_type': lambda self: self.day_type,
    }

    def vec(self, features=features, trip_coords=[0, -1]):
        """
        Returns the sample as a Numpy vector of
        'len(features) + 2 * len(trip_features)' elements.

        'features' specifies the names of the features that must be in the
        vector, while 'trip_coords' specifies the pair of coordinates that must
        be added to the vector.

        Features are returned in the same order as in the provided 'features'
        parameter, and are then followed by the coordinates (latitude and
        longitude) of the trip as specified in 'trip_coords'.

        Example:
            >>> sample.vec(['start_time.day', 'start_time.year'], [0, -1])
            array([    3.      ,  2013.      ,    41.148864,    -8.585649,
                      41.237253,    -8.669925])
        """

        assert all(trip_coord < len(self.trip) for trip_coord in trip_coords)

        return np.concatenate([[
                Sample._feature_getters[feature_name](self)
                for feature_name in features
            ]] + [
                [self.trip[trip_coord].lat, self.trip[trip_coord].lon]
                for trip_coord in trip_coords
            ]
        )

    #
    # Export the sample's trip as a Numpy vector.
    #

    @property
    def trip_vec(self):
        """
        Gives a Numpy array of shape [len(self.trip), 2] containing a two
        elements vector [latitude, longitude] for each GPS coordinate of the
        path.

        Example:
            >>> sample.trip_vec
            array([[ 41.148864,  -8.585649],
                   [ 41.148963,  -8.586549],
                   [ 41.149368,  -8.587998])
        """

        return np.array([ [coord.lat, coord.lon] for coord in self.trip ])

def load_dataset(filepath):
    """
    Generates a stream of 'Sample's from the given datafile.

    Example:
        for sample in load_dataset('data/train_data.csv'):
            print(len(sample.trip))
    """

    with open(filepath) as f:
        reader = csv.reader(f)

        # Skips the header
        next(reader)

        for line in reader:
            yield Sample(line)

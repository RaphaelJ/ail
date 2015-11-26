# Authors: Maxime Javaux    <maximejavaux@hotmail.com>
#          Raphael Javaux   <raphaeljavaux@gmail.com>

"""Provide a function to read a CSV dataset from a file."""

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

    @property
    def as_numpy_arr(self):
        """
        Returns the sample as a Numpy vector of 15 elements.
        """

        return np.array([
            self.call_type, self.origin_call, self.origin_stand, self.taxi_id,
            self.start_time.year, self.start_time.month, self.start_time.day,
            self.start_time.weekday(), self.start_time.hour,
            self.start_time.minute, self.day_type, self.begin.lat,
            self.begin.lon, self.end.lat, self.end.lon,
        ])

    @property
    def trip_as_numpy_arr(self):
        """
        Returns a Numpy array of shape [len(self.trip), 2] containing a two
        elements vector [latitude, longitude] for each GPS coordinate of the
        path.
        """

        return np.array([ [coord.lat, coord.lon] for coord in self.trip ])

def load_dataset(filepath):
    """
    Generates a stream of 'Sample's from the given datafile.

    Example:
        for sample in load_dataset('data/train_data.csv'):
            print(len(sample.trip))
    """

    with open(filepath, newline='') as f:
        reader = csv.reader(f)

        # Skips the header
        reader.__next__()

        for line in reader:
            yield Sample(line)

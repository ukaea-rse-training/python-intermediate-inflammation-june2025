"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os

import numpy as np

from inflammation import models


def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    data = data_source.load_inflammation_data()

    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return {
        'standard deviation by day': daily_standard_deviation,
    }


class JSONDataSource:
    """JSON inflammation data file loader

    Parameters
    ----------
    data_dir : str
        The path to the directory with the JSON files
    """
    def __init__(self, data_dir: str):
        self._data_dir = data_dir

    def load_inflammation_data(self):
        """Load the inflammation data from JSON files in the specified directory

        Returns
        -------
        list[np.ndarray]:
            A list of the 2D inflammation data in each file read.
        """
        data_file_paths = glob.glob(os.path.join(self._data_dir, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data JSON files found in path {self._data_dir}")
        return list(map(models.load_json, data_file_paths))


class CSVDataSource:
    def __init__(self, data_dir: str):
        self._data_dir = data_dir

    def load_inflammation_data(self):
        """Load the inflammation data from CSV files in the specified directory

        Parameters
        ----------
        data_dir : str
            The path to the directory with the CSV files

        Returns
        -------
        list[np.ndarray]:
            A list of the 2D inflammation data in each file read.
        """
        data_file_paths = glob.glob(os.path.join(self._data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self._data_dir}")
        return list(map(models.load_csv, data_file_paths))

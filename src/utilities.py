import pandas as pd
import warnings
from typing import Union


def read_data(path: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Reads the data form the given file path
    :param path: path of the file containing the data or a pandas Dataframe. In the last case, the func simply returns this parameter
    :return: a pandas dataframe containing the data
    """
    # if path is already a dataframe, return it (so the function is callable without checking if you have a path or the dataframe itself already)
    if isinstance(path, pd.DataFrame):
        return path
    # otherwise read the data and return the dataframe
    col_names = ["ID", "FRAME", "X", "Y", "Z"]
    data = pd.read_csv(path, sep=" ", header=None, names=col_names)
    return data


def create_complete_dataframe(original_data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Creates a complete dataframe containing extra - derived - data, e.g. speed and mean spacing for each pedestrian
    :param original_data: original data as a pandas dataframe or path to the file containing the data
    :return: a pandas dataframe containing the original data plus some derived data
    """
    # if original_data is a path, read the data; if it's a dataframe, simply return the dataframe itself
    extended_df = read_data(original_data)

    # add speed to the data: frame rate of 16Hz, compute space between the current position and the next one and divide by 1/16
    pass

    # add mean spacing of each pedestrian to the data
    pass

    return extended_df

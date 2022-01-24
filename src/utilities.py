import pandas as pd
import warnings
import copy


def read_data(path):
    """
    Reads the data form the given file path
    :param path: path of the file containing the data
    :return: a pandas dataframe containing the data
    """
    col_names = ["ID", "FRAME", "X", "Y", "Z"]
    data = pd.read_csv(path, sep=" ", header=None, names=col_names)
    return data


def create_complete_dataframe(original_df: pd.DataFrame = None, original_data_path: str = None):
    """
    Creates a complete dataframe containing extra - derived - data, e.g. speed and mean spacing for each pedestrian
    :param original_df: (optional) original data as a pandas dataframe, if None, read using func "read_data"
    :param original_data_path: (optional) path to a file containing the data. If original_df is None, this must be not None
    :return: a pandas dataframe containing the original data plus some derived data
    """
    # check on attributes
    if original_df is None and original_data_path is None:
        raise AttributeError("One attribute between original_df and original_data_path must be provided, got None and None")
    if original_df is not None and original_data_path is not None:
        warnings.warn("Both original_df and original_data_path are not None. original_data_path is going to be ignored", RuntimeWarning)

    # at this point, only one between original_df and original_data_path is not None
    if original_df is None:
        original_df = read_data(original_data_path)
    extended_df = original_df

    # add speed to the data: frame rate of 16Hz, compute space between the current position and the next one and divide by 1/16
    pass

    # add mean spacing of each pedestrian to the data
    pass

    return extended_df

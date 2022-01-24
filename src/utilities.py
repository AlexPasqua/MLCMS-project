import numpy as np
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


def add_speeds(data: Union[pd.DataFrame, str], frame_rate: float = 16, save_path: str = None) -> pd.DataFrame:
    """
    Add, for each frame and for each pedestrian, the current speed. Computed as space (between the current and next position of the pedestrian)
    divided by the time (derived having the frame rate)
    :param data: data as a pandas dataframe or path to the file containing the data
    :param frame_rate: frame rate of the data (default is 16 as indicated by te data providers (https://zenodo.org/record/1054017#.Ye5F4_7MJPY))
    :param save_path: if not None, save the dataframe to the specified path
    :return: a pandas dataframe extended with the speeds for each pedestrian and for each frame
    """
    # read the data (if data it's a dataframe, this instruction has no effect)
    data = read_data(data)

    # sort the data by pedestrian ID and the by frame number
    data.sort_values(by=['ID', 'FRAME'], axis=0, inplace=True)

    # compute the speeds
    speeds = []
    rows_to_drop = []
    for index, row in data.iterrows():
        # check that this row and the next on refer to the same pedestrian
        next_idx = index + 1
        try:
            if data.iloc[index]['ID'] == data.iloc[next_idx]['ID']:
                pos1 = data.iloc[index][['X', 'Y']].to_numpy()
                pos2 = data.iloc[next_idx][['X', 'Y']].to_numpy()
                space = np.linalg.norm(pos1 - pos2)
                speeds.append(space * frame_rate)
            else:
                # in this case save the row number for dropping it later (now not possible because we're iterating)
                rows_to_drop.append(index)
        except IndexError:
            # it means we reached the last row and data.iloc[index + 1] gives an error. Drop last row as well (didn't compute a speed for it)
            rows_to_drop.append(index)

    # drop rows representing the last frame for each pedestrian, because we didn't calculate a speed for those rows
    data.drop(rows_to_drop, inplace=True)

    # add the speeds as an additional column to the dataframe
    data = data.assign(SPEED=speeds)

    # in case save the dataframe
    if save_path is not None:
        data.to_csv(save_path, sep=" ", header=False, index=False)

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
    extended_df = add_speeds(extended_df)

    # add mean spacing of each pedestrian to the data
    pass

    return extended_df


if __name__ == '__main__':
    create_complete_dataframe("../data/Pedestrian_Trajectories/Corridor_Data/ug-180-015.txt")

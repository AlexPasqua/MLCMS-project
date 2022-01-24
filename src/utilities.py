import numpy as np
import pandas as pd
import math
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
    data['SPEED'] = speeds

    # in case save the dataframe
    if save_path is not None:
        data.to_csv(save_path, sep=" ", header=False, index=False)

    return data


def add_mean_spacings(data: Union[pd.DataFrame, str], number_of_neighbors: int = 10, save_path: str = None) -> pd.DataFrame:
    """
    Add the mean spacing of each pedestrian and its n nearest neighbors
    :param data: data as a pandas dataframe or path to the file containing the data
    :param number_of_neighbors: N Nearest NeighborS
    :param save_path: if not None, save the dataframe to the specified path
    :return: a pandas dataframe extended with the mean spacings for each pedestrian and for each frame
    """
    # TODO: add memoization

    # read the data (if data it's a dataframe, this instruction has no effect)
    data = read_data(data)

    # find the number_of_neighbors nearest neighbors
    mean_spacings_data = []
    # iterate for each frame
    frame_numbers = data['FRAME'].unique()
    for frame in frame_numbers:
        # select the portion of dataframe containing data regarding the current frame
        curr_frame_data = data[data['FRAME'] == frame]
        # save the ids of all the pedestrian present in the scenario in the current frame
        ids = curr_frame_data['ID'].unique()
        for curr_id in ids:
            knn_dists = np.full(shape=(number_of_neighbors,), fill_value=-math.inf)
            for neighbor_id in ids:
                # skip the case where the current id and the neighbor's id are the same
                if curr_id == neighbor_id:
                    continue

                # compute the distance between the current pedestrian and the current neighbor
                pos1 = curr_frame_data[curr_frame_data['ID'] == curr_id][['X', 'Y']].to_numpy()
                pos2 = curr_frame_data[curr_frame_data['ID'] == neighbor_id][['X', 'Y']].to_numpy()
                dist = np.linalg.norm(pos1 - pos2)
                assert dist > -math.inf
                if dist > min(knn_dists):
                    knn_dists[np.argmin(knn_dists)] = dist

            # compute the mean distancing
            knn_dists = knn_dists[knn_dists != -math.inf]
            mean_spacing = np.mean(knn_dists)
            mean_spacings_data.append((curr_id, frame, mean_spacing))

    # save the mean spacings data into the dataframe as a new column
    data.sort_values(by=['ID', 'FRAME'], axis=0, inplace=True)
    mean_spacings_data = sorted(mean_spacings_data, key=lambda x: (x[0], x[1]))  # order first by pedestrian id and the by frame number
    mean_spacings_data = [mean_spacings_data[i][2] for i in range(len(mean_spacings_data))]
    data['MEAN_SPACING'] = mean_spacings_data
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
    extended_df = add_mean_spacings(extended_df)
    return extended_df


if __name__ == '__main__':
    add_mean_spacings("../data/Pedestrian_Trajectories/Corridor_Data/ug-180-015.txt")

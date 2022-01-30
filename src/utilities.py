import numpy as np
import pandas as pd
import math
from typing import Union
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


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
    # scale the coordinates between 0 and 1
    # scaler = MinMaxScaler()
    # data['X'] = scaler.fit_transform(np.expand_dims(data['X'], 1))
    # data['Y'] = scaler.fit_transform(np.expand_dims(data['Y'], 1))
    # drop Z coordinate
    data.drop(labels='Z', inplace=True, axis=1)
    # scale X and Y
    data['X'] = data['X'] / 100
    data['Y'] = data['Y'] / 100
    return data


def add_speeds(data: Union[pd.DataFrame, str], frame_rate: float = 16) -> pd.DataFrame:
    """
    Add, for each frame and for each pedestrian, the current speed. Computed as space (between the current and next position of the pedestrian)
    divided by the time (derived having the frame rate)
    :param data: data as a pandas dataframe or path to the file containing the data
    :param frame_rate: frame rate of the data (default is 16 as indicated by te data providers (https://zenodo.org/record/1054017#.Ye5F4_7MJPY))
    :return: a pandas dataframe extended with the speeds for each pedestrian and for each frame
    """
    # read the data (if data it's a dataframe, this instruction has no effect)
    data = read_data(data)

    # sort the data by pedestrian ID and the by frame number
    data.sort_values(by=['ID', 'FRAME'], axis=0, inplace=True)

    # compute the speeds
    speeds = []
    rows_to_drop = []
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Adding speeds"):
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

    return data


def add_neighbors_positions(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    For each pedestrian, at each time step, add the positions of all the other pedestrians
    :param data: data as a pandas dataframe or path to the file containing the data
    :return: pandas dataframe extended with - for every pedestrian at each time step - the positions of all the other pedestrians
    """
    # read the data (if data it's a dataframe, this instruction has no effect)
    data = read_data(data)

    # dataframe to return
    id_frame_others_pos = data[['ID', 'FRAME']]
    id_frame_others_pos = id_frame_others_pos.assign(OTHERS_POSITIONS=lambda x: None)     # add empty column for the other pedestrians' positions
    id_frame_others_pos['OTHERS_POSITIONS'] = id_frame_others_pos['OTHERS_POSITIONS'].astype(object)

    # iterate for each frame
    frame_numbers = data['FRAME'].unique()
    for frame in tqdm(frame_numbers, desc="Adding other pedestrians' positions"):
        # select the portion of dataframe containing data regarding the current frame
        curr_frame_data = data[data['FRAME'] == frame]
        # save the ids of all the pedestrian present in the scenario in the current frame
        ids = curr_frame_data['ID'].unique()
        # iterate on all the pedestrians
        for curr_id in ids:
            others_pos_and_dist = []
            # iterate on all the pedestrians other than curr_id (i.e. all its neighbors)
            for neighbor_id in ids:
                if curr_id == neighbor_id:
                    continue

                # add the neighbor's position to this list in order to save it
                xy = curr_frame_data.loc[curr_frame_data['ID'] == neighbor_id, ['X', 'Y']]
                x, y = np.squeeze(xy.to_numpy())

                # compute the distance between the current pedestrian and the current neighbor
                pos1 = curr_frame_data[curr_frame_data['ID'] == curr_id][['X', 'Y']].to_numpy()
                pos2 = curr_frame_data[curr_frame_data['ID'] == neighbor_id][['X', 'Y']].to_numpy()
                dist = np.linalg.norm(pos1 - pos2)
                others_pos_and_dist.append((x, y, dist))

            # add the others' positions in the dataframe
            rows_selector = (id_frame_others_pos['ID'] == curr_id) & (id_frame_others_pos['FRAME'] == frame)  # boolean list with only 1 True
            row_number = np.argmax(rows_selector)
            id_frame_others_pos.at[row_number, 'OTHERS_POSITIONS'] = others_pos_and_dist.sort(key=lambda elem: elem[2])    # sorted by distance

    # merge the 2 dataframes, having the effect of adding a column for the mean spacing to the data
    extended_data = data.merge(id_frame_others_pos, how='inner', on=['ID', 'FRAME'])
    return extended_data


def create_extended_dataframe(original_data: Union[pd.DataFrame, str], save_path: str = None) -> pd.DataFrame:
    """
    Creates an extended dataframe containing extra - derived - data, e.g. speeds and other pedestrians' positions
    :param original_data: original data as a pandas dataframe or path to the file containing the data
    :param save_path: (optional) if not None, save the model to the specified path
    :return: a pandas dataframe containing the original data plus some derived data
    """
    # if original_data is a path, read the data; if it's a dataframe, simply return the dataframe itself
    extended_df = read_data(original_data)

    # add mean spacing of each pedestrian to the data
    extended_df = add_neighbors_positions(extended_df)

    # add speed to the data: frame rate of 16Hz, compute space between the current position and the next one and divide by 1/16
    extended_df = add_speeds(extended_df)

    # in case save the dataframe
    if save_path is not None:
        extended_df.to_csv(save_path, sep=" ", header=False, index=False)
        print("Saved " + save_path)

    return extended_df


if __name__ == '__main__':
    data = create_extended_dataframe(
        original_data="../data/Pedestrian_Trajectories/Corridor_Data/ug-180-015.txt",
        save_path="../data/to_delete"
    )

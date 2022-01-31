from typing import Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from utilities import read_data


def _create_num_neighbours_df(data: pd.DataFrame, neighbour_num: int = 10) -> pd.DataFrame:
    """
    filter frames from dataframe where there are not enough neighbours
    :param data: dataframe
    :param neighbour_num: number of neighbours required in each frame
    :return: filtered dataframe
    """
    frames = data['FRAME'].unique()
    frames_to_delete = []
    for frame in frames:
        if len(data[data['FRAME'] == frame]) < (neighbour_num + 1):  # has to take into account also the pedestrian itself
            frames_to_delete.append(frame)
    df = data[~data['FRAME'].isin(frames_to_delete)]
    return df


def _add_mean_spacing(data: pd.DataFrame, k: int = 10, keep_dist: bool = False, re_sort: bool = True) -> pd.DataFrame:
    """
    Add the mean spacing of each pedestrians w.r.t. its k nearest neighbors
    :param data: source dataframe containing the data
    :param k: number of nearest neighbors to consider
    :param keep_dist: if True, for each pedestrian, keep the distance of each one of its k nearest neighbors, otherwise remove this info
    :param re_sort: for each row, sort the other's positions and distances by distance, to be sure to have the correct nearest neighbors.
        This piece of data should already be sorted, so it shouldn't be necessary, it's a safety measure
    :return: dataframe containing the mean spacing of each pedestrians w.r.t. its k nearest neighbors and shortened list of neighbors positions
        (new length = k) with removed distance from the current pedestrian (if keep_dist = False)
    """
    data = data.assign(MEAN_SPACING=lambda x: None)
    for i, curr_row in data.iterrows():
        # keep only the k nearest neighbors for each pedestrian at each time step
        others_pos = curr_row['OTHERS_POSITIONS']
        if re_sort:
            # sort the list of tuples (x_neighbor, y_neighbor, dist_neighbor) by distance to make sure we have the correct nearest neighbors
            # (this piece of data should already be sorted in data, this is a safety measure)
            others_pos.sort(key=lambda x: x[2])
            data.at[i, 'OTHERS_POSITIONS'] = others_pos

        k_others_pos = others_pos[:k]
        dists = [triple[2] for triple in k_others_pos]
        if not keep_dist:
            data.at[i, 'OTHERS_POSITIONS'] = np.array([nn[0:2] for nn in k_others_pos]).flatten()
        else:
            data.at[i, 'OTHERS_POSITIONS'] = np.array(k_others_pos).flatten()

        # compute the mean spacing of each pedestrian at each time step w.r.t. its k nearest neighbors
        data.loc[i, 'MEAN_SPACING'] = np.mean(dists)

    return data


def _add_speeds(data: Union[pd.DataFrame, str], frame_rate: float = 16) -> pd.DataFrame:
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


def _create_relative_neighbours_df(extended_data):
    """
    Construct the relative positions of every k nearest neighbour wrt the reference pedestrian
    :param extended_data: data complete of neighbour positions and reference position
    :return: return dataframe complete also of relative positions of neighbours
    """
    extended_data['KNN_RELATIVE_POSITIONS'] = np.empty(shape=extended_data.shape[0])
    extended_data['KNN_RELATIVE_POSITIONS'] = extended_data['KNN_RELATIVE_POSITIONS'].astype(object)
    for index, row in tqdm(extended_data.iterrows(), total=len(extended_data), desc="Adding relative positions"):
        relative_pos_list = []
        reference_p = [row['X'], row['Y']]
        for i in range(len(row['OTHERS_POSITIONS'])):
            reference_index = 0 if i % 2 == 0 else 1
            relative_pos_list.append(row['OTHERS_POSITIONS'][i]-reference_p[reference_index])
        extended_data.at[index, 'KNN_RELATIVE_POSITIONS'] = np.array(relative_pos_list)
    return extended_data




def _add_neighbors_positions(data: Union[pd.DataFrame, str]) -> pd.DataFrame:
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
                others_pos_and_dist.append([x, y, dist])

            # add the others' positions in the dataframe
            rows_selector = (id_frame_others_pos['ID'] == curr_id) & (id_frame_others_pos['FRAME'] == frame)  # boolean list with only 1 True
            row_number = np.argmax(rows_selector)
            others_pos_and_dist.sort(key=lambda elem: elem[2])  # sorted by distance
            id_frame_others_pos.at[row_number, 'OTHERS_POSITIONS'] = others_pos_and_dist

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
    extended_df = _add_neighbors_positions(extended_df)

    # add speed to the data: frame rate of 16Hz, compute space between the current position and the next one and divide by 1/16
    extended_df = _add_speeds(extended_df)

    # in case save the dataframe
    if save_path is not None:
        extended_df.to_pickle(save_path)
        print("Saved " + save_path)

    return extended_df



def create_dataset(original_data: Union[pd.DataFrame, str], k: int = 10, extended_save_path: str = None, dataset_save_path: str = None) -> pd.DataFrame:
    """
    Creates the dataset for the NN starting from the raw data
    :param original_data: original data as a pandas dataframe or path to the file containing the data
    :param k: number of nearest neighbors to consider
    :param extended_save_path: (optional) if not None, save the extended dataframe to the specified path
    :param dataset_save_path: (optional) if not None, save the dataset to the specified path
    :return:
    """
    # extend data with more information (e.g. neighbors positions etc)
    # extended_data = create_extended_dataframe(original_data, save_path=extended_save_path)
    extended_data = pd.read_pickle(original_data)

    # remove frames with less than k nearest neighbors
    dataset = _create_num_neighbours_df(extended_data, neighbour_num=k)

    # add the mean spacing between each pedestrian and its current k nearest neighbors
    dataset = _add_mean_spacing(dataset, k=k, keep_dist=False, re_sort=False)

    # create relative positions of k nearest neighbors
    dataset = _create_relative_neighbours_df(dataset)

    # drop columns not to be fed to the network
    dataset.drop(['ID', 'FRAME', 'X', 'Y', 'OTHERS_POSITIONS'], axis=1, inplace=True)

    # save
    if dataset_save_path is not None:
        dataset.to_pickle(dataset_save_path)

    return dataset


if __name__ == '__main__':
    original_data_path = "../data/Pedestrian_Trajectories/Corridor_Data/ug-180-030.txt"
    extended_path = "../data/corridor_30_extended.pickle"
    save_path = "../data/dataset_corridor_30.pickle"
    dataset = create_dataset(original_data=extended_path, k=10, extended_save_path=extended_path, dataset_save_path=save_path)

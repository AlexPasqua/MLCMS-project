import numpy as np
import pandas as pd
import os.path


def create_num_neighbours_df(file_path, neighbour_num=10):
    """
    filter frames from dataframe where there are not enough neighbours
    :param file_path: path to fetch the dataframe
    :param save_file_path: path where to save the filtered dataframe
    :param neighbour_num: number of neighbours required in each frame
    :return: filtered dataframe
    """
    df = pd.read_pickle(file_path)
    frames = df['FRAME'].unique()
    frames_to_delete = []
    for frame in frames:
        if len(df[df['FRAME'] == frame]) < (neighbour_num + 1):  # has to take into account also the pedestrian itself
            frames_to_delete.append(frame)
    df = df[~df['FRAME'].isin(frames_to_delete)]
    return df


def add_mean_spacing(df: pd.DataFrame, k: int = 10, keep_dist: bool = False, re_sort: bool = True) -> pd.DataFrame:
    """
    Add the mean spacing of each pedestrians w.r.t. its k nearest neighbors
    :param df: source dataframe containing the data
    :param k: number of nearest neighbors to consider
    :param keep_dist: if True, for each pedestrian, keep the distance of each one of its k nearest neighbors, otherwise remove this info
    :param re_sort: for each row, sort the other's positions and distances by distance, to be sure to have the correct nearest neighbors.
        This piece of data should already be sorted, so it shouldn't be necessary, it's a safety measure
    :return: dataframe containing the mean spacing of each pedestrians w.r.t. its k nearest neighbors and shortened list of neighbors positions
        (new length = k) with removed distance from the current pedestrian (if keep_dist = False)
    """
    for i, curr_row in df.iterrows():
        # keep only the k nearest neighbors for each pedestrian at each time step
        others_pos = curr_row['OTHERS_POSITIONS']
        dists = []
        if re_sort:
            # sort the list of tuples (x_neighbor, y_neighbor, dist_neighbor) by distance to make sure we have the correct nearest neighbors
            # (this piece of data should already be sorted in df, this is a safety measure)
            others_pos.sort(key=lambda x: x[2])
            df.at[i, 'OTHERS_POSITIONS'] = others_pos
        df.at[i, 'OTHERS_POSITIONS'] = others_pos[:k]
        for triple in others_pos[:k]:
            dists.append(triple[2])
            if not keep_dist:
                # keep only the k nearest neighbors
                df.at[i, 'OTHERS_POSITIONS'] = [nn[0:2] for nn in others_pos[:k]]

        # compute the mean spacing of each pedestrian at each time step w.r.t. its k nearest neighbors
        df.at[i, 'MEAN_SPACING'] = np.mean(dists)

    return df


def training_data_preprocess(df, save_path):
    """
    get the df with correct number of neighbours, filter unuseful columns and return array which can be fed to model
    :param df: dataframe with correct number of neighbours per each row
    :return: model feedable input array
    """
    columns_to_delete = ['ID', 'FRAME', 'X', 'Y']
    df.drop(columns_to_delete, axis=1, inplace=True)
    df.to_csv(save_path, sep=" ", header=None)
    return df.to_numpy()


if __name__ == '__main__':
    task = 'corridor'
    num_neighbours = 10
    col_names = ['ID', 'FRAME', 'X', 'Y', 'MEAN_SPACING', 'OTHERS_POSITIONS', 'SPEED']
    complete_dataframe_path = f"../data/to_delete"
    save_path = f"../data/to_delete_filtered_{num_neighbours}-nn_dataframe"
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, sep=" ", header=None, names=col_names)
    else:
        data = create_num_neighbours_df(complete_dataframe_path, neighbour_num=num_neighbours)
    data = add_mean_spacing(data)
    data = training_data_preprocess(data, save_path)

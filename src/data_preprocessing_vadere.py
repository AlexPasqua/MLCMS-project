import numpy as np
import pandas as pd


def get_time_delta(base_path):
    """
    time deltas between measurements are different because of the simulation, this function returns the minimum of all
    deltas as approximation
    :param base_path: where to get the trajectories for calculating the deltass
    :return: minimum time delta
    """
    peds_pos = np.loadtxt(base_path + "postvis.traj", skiprows=1)
    peds_pos_starting_delta, peds_pos_ending_delta = peds_pos[:, 1], peds_pos[:, 2]
    time_delta = np.subtract(peds_pos_ending_delta, peds_pos_starting_delta)
    return min(set([round(i, 1) for i in time_delta]))


def get_basic_dataset_fields(knn_dataset):
    """
    set time_step, id and mean_spacing fields
    :param knn_dataset:
    :return: updated dataset
    """
    knn_df = pd.DataFrame()
    knn_df['TIME_STEP'] = knn_file[:, 0]
    knn_df['ID'] = knn_file[:, 1]
    knn_df['MEAN_SPACING'] = knn_file[:, 2]
    return knn_df


def add_relative_positions(knn_df):
    """
    add relative positions of nearest neighbours
    :param knn_df: get dataframe without relative positions
    :return: get dataframe with relative positions
    """
    knn_positions = knn_file[:, 3:-2]  # already relative positions
    knn_df['KNN_RELATIVE_POSITIONS'] = np.empty(shape=knn_positions.shape[0])
    knn_df['KNN_RELATIVE_POSITIONS'] = knn_df['KNN_RELATIVE_POSITIONS'].astype(object)
    for i in range(knn_positions.shape[0]):
        knn_df.at[i, 'KNN_RELATIVE_POSITIONS'] = np.array(
            [(knn_positions[i, j], knn_positions[i, j + 1]) for j in range(0, knn_positions.shape[1], 2)]).flatten()
    return knn_df


def add_speed(knn_df, knn_file, time_step):
    """
    add speed to each row (last row of each pedestrians and rows with pedestrians staying still)
    :param knn_df: dataframe without pedestrian speeds
    :param knn_file: file original out of vadere elaboration
    :param time_step: estimated time passing between two simulation steps
    :return: dataframe with pedestrian speeds
    """
    knn_pedestrian_position = np.array([(knn_file[i, -2], knn_file[i, -1]) for i in range(knn_file.shape[0])])
    knn_df['PEDESTRIAN_POSITION'] = np.empty(shape=knn_file.shape[0])
    knn_df['PEDESTRIAN_POSITION'] = knn_df['PEDESTRIAN_POSITION'].astype(object)
    for i in range(knn_pedestrian_position.shape[0]):
        knn_df.at[i, 'PEDESTRIAN_POSITION'] = knn_pedestrian_position[i]
    ped_ids = knn_df['ID'].unique()
    # set all speed to -1.0 so that at the end we can cut them off
    knn_df['SPEED'] = -np.ones(shape=knn_file.shape[0])
    # compute speed for every instant
    for ped_id in ped_ids:
        tmp_df = knn_df[knn_df['ID'] == ped_id]
        tmp_df_index = tmp_df.index
        for i in range(len(tmp_df_index) - 1):
            pos_a, pos_b = tmp_df.iloc[i]['PEDESTRIAN_POSITION'], tmp_df.iloc[i + 1]['PEDESTRIAN_POSITION']
            delta_t = tmp_df.iloc[i + 1]['TIME_STEP'] - tmp_df.iloc[i]['TIME_STEP']
            knn_df.iloc[tmp_df_index[i], list(knn_df.columns).index('SPEED')] = np.linalg.norm(pos_a - pos_b) / (delta_t * time_step)

    # delete all rows with speed = -1.0 (are last rows for a given pedestrian) or 0.0 (pedestrian is standing still)
    knn_df = knn_df[~knn_df['SPEED'].isin([-1.0, 0.0])]
    knn_df = knn_df.drop(['ID', 'TIME_STEP', 'PEDESTRIAN_POSITION'], axis=1, inplace=False)
    return knn_df


def create_complete_dataset_vadere(knn_file, time_step, dataset_save_path=None):
    """
    Return a vadere dataset (and save it if required), ready with the input and output field for the NN
    file can be fed to NN simply calling utilities.read_dataset on the saved pickle path
    :param knn_file: file coming out of vadere
    :param time_step: estimated time passing between two simulation steps
    :param dataset_save_path: where to save the complete dataset
    :return: complete dataset full of speed, relative positions and mean spacing
    """
    # get sim_times, pedestrian ids and mean_spacing
    knn_df = get_basic_dataset_fields(knn_file)

    # add relative positions of knns wrt to defined pedestrian
    knn_df = add_relative_positions(knn_df)

    # add speed of pedestrians
    knn_df = add_speed(knn_df, knn_file, time_step)

    # save dataset if requested
    if dataset_save_path is not None:
        knn_df.to_pickle(dataset_save_path)

    return knn_df


if __name__ == '__main__':
    base_path = "../vadere-projects/output/bottleneck_vadere_final/"
    time_step = get_time_delta(base_path)  # dependent on the simulation, pay attention!
    knn_file = np.loadtxt(base_path + "out.txt", skiprows=1)  # get the file containing knns, remove header line
    knn_df = create_complete_dataset_vadere(knn_file, time_step, dataset_save_path="../data/vadere_bottleneck_100_90")

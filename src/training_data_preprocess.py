import pandas as pd
import os.path


def create_num_neighbours_df(file_path, save_file_path, neighbour_num=10):
    """
    filter frames from dataframe where there are not enough neighbours
    :param file_path: path to fetch the dataframe
    :param save_file_path: path where to save the filtered dataframe
    :param neighbour_num: number of neighbours required in each frame
    :return: filtered dataframe
    """
    df = pd.read_csv(file_path, sep=" ", header=None)
    frames = df['FRAME'].unique()
    frames_to_delete = []
    for frame in frames:
        if len(df[df['FRAME'] == frame]) < (neighbour_num + 1):  # has to take into account also the pedestrian itself
            frames_to_delete.append(frame)
    df = df[~df['FRAME'].isin(frames_to_delete)]
    df.to_csv()
    return df.to_csv(save_file_path, sep=" ")


def training_data_preprocess(df):
    """
    get the df with correct number of neighbours, filter unuseful columns and return array which can be fed to model
    :param df: dataframe with correct number of neighbours per each row
    :return: model feedable input array
    """
    columns_to_delete = ['ID', 'FRAME', 'X', 'Y', 'Z']
    df.drop(columns_to_delete, axis=1, inplace=True)
    return df.to_numpy()


if __name__ == '__main__':
    task = 'corridor'
    complete_dataframe_path = f"../data/{task}_complete_dataframe"
    data_path = f"../data/{task}_filtered_dataframe"
    num_neighbours = 10
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, sep=" ", header=None)
    else:
        data = create_num_neighbours_df(complete_dataframe_path, data_path, num_neighbours=num_neighbours)
    data = training_data_preprocess(data)

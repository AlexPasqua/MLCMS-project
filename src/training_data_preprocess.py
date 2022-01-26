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
    col_names = ['ID', 'FRAME', 'X', 'Y', 'MEAN_SPACING', 'OTHER_POSITIONS', 'SPEED']
    df = pd.read_csv(file_path, sep=" ", header=None, names=col_names)
    frames = df['FRAME'].unique()
    frames_to_delete = []
    for frame in frames:
        if len(df[df['FRAME'] == frame]) < (neighbour_num + 1):  # has to take into account also the pedestrian itself
            frames_to_delete.append(frame)
    df = df[~df['FRAME'].isin(frames_to_delete)]
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
    col_names = ['ID', 'FRAME', 'X', 'Y', 'MEAN_SPACING', 'OTHER_POSITIONS', 'SPEED']
    complete_dataframe_path = f"../data/{task}_15_complete_dataframe_2"
    save_path = f"../data/{task}_15_filtered_{num_neighbours}-nn_dataframe"
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, sep=" ", header=None, names=col_names)
    else:
        data = create_num_neighbours_df(complete_dataframe_path, neighbour_num=num_neighbours)
    data = training_data_preprocess(data, save_path)

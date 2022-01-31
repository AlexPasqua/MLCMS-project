import numpy as np
import pandas as pd
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


def read_dataset(path: str) -> (np.ndarray, np.ndarray):
    """
    Read the complete dataset to be fed to the NN
    :param path: path of the pickle file containing the dataset
    :return: data and targets in the form of 2 numpy ndarrays
    """
    dataset = pd.read_pickle(path)
    targets = dataset[['SPEED']].to_numpy()
    mean_spacing = dataset[['MEAN_SPACING']].to_numpy()
    knn_relative_positions = dataset['KNN_RELATIVE_POSITIONS'].to_numpy()
    # join the mean spacing with the knn_relative_positions
    data = np.empty(shape=(len(dataset), len(knn_relative_positions[0]) + 1))
    for i in range(len(dataset)):
        row = np.concatenate((mean_spacing[i], knn_relative_positions[i]))
        data[i, :] = row
    return data, targets

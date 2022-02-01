import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple
from fd_model_nn import FD_Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def read_data(path: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Reads the data form the given file path
    :param path: path of the file containing the data or a pandas Dataframe. In the last case, the func simply returns this parameter
    :return: a pandas dataframe containing the data
    """
    # if path is already a dataframe, return it
    # (so the function is callable without checking if you have a path or the dataframe itself already)
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


def read_dataset(path: str, fd_training=False) -> (np.ndarray, np.ndarray):
    """
    Read the complete dataset to be fed to the NN
    :param path: path of the pickle file containing the dataset
    :param fd_training: if True keep also k nearest neighbour for input, False leaves only mean spacing for FDNetwork
    :return: data and targets in the form of 2 numpy ndarrays
    """
    dataset = pd.read_pickle(path)
    targets = dataset[['SPEED']].to_numpy()
    mean_spacing = dataset[['MEAN_SPACING']].to_numpy()
    if not fd_training:
        knn_relative_positions = dataset['KNN_RELATIVE_POSITIONS'].to_numpy()
        # join the mean spacing with the knn_relative_positions
        data = np.empty(shape=(len(dataset), len(knn_relative_positions[0]) + 1))
        for i in range(len(dataset)):
            row = np.concatenate((mean_spacing[i], knn_relative_positions[i].flatten()))
            data[i, :] = row
        return data, targets
    return mean_spacing, targets


def plot_fd_and_speeds(data_path: str, plot_title: str = "", fd_epochs: int = 50, nn_epochs: int = 50, hidden_dims: Tuple[int] = (3,),
                       training_plots: bool = True):
    """
    Plots the speeds predicted by the network and the FD curve depending on the mean spacing
    :param data_path: path of the file containing the data
    :param plot_title: title of the plot
    :param fd_epochs: number of epochs of training for the FD
    :param nn_epochs: number of epochs of training for the NN
    :param hidden_dims: tuple containing the dimensions of the hidden layers of the NN
    :param training_plots: if True, plots the training curves of the FD and NN
    """
    fd_data, fd_targets = read_dataset(data_path, fd_training=True)
    nn_data, nn_targets = read_dataset(data_path, fd_training=False)

    # train the FD model
    model = FD_Network()
    model.compile(optimizer='sgd', loss='mse')
    hist = model.fit(x=fd_data, y=fd_targets, epochs=fd_epochs)
    loss_fd = hist.history['loss']

    # train the speed predictor neural network
    layers = [Dense(units=d) for d in hidden_dims] + [Dense(units=1)]
    nn = Sequential(layers)
    nn.compile(optimizer='sgd', loss='mse')
    hist = nn.fit(x=nn_data, y=nn_targets, epochs=nn_epochs)
    loss_nn = hist.history['loss']

    # training plots
    if training_plots:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
        # FD
        ax[0].plot(loss_fd)
        ax[0].set_title("FD training")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("MSE")
        # NN
        ax[1].plot(loss_nn)
        ax[1].set_title("NN training")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("MSE")
        fig.show()

    # plot
    mean_spacings = np.expand_dims(np.linspace(start=0.5, stop=3.5, num=1000), axis=1)
    fd_speeds = model.predict(x=mean_spacings)
    nn_speeds = nn.predict(x=nn_data)
    plt.plot(mean_spacings, fd_speeds)
    plt.scatter(nn_data[:, 0], nn_speeds)
    plt.xlabel("Mean spacing")
    plt.ylabel("Speed")
    plt.title(plot_title)
    plt.show()


if __name__ == '__main__':
    path = "../data/vadere_bottleneck_100_90"
    plot_fd_and_speeds(data_path=path)

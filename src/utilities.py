import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple
from fd_model_nn import FD_Network
from nn_utilities import *


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
    return mean_spacing.astype(float), targets.astype(float)


def plot_fd_and_original(data_path: str, plot_title: str = "", fd_epochs: int = 50,
                         test_data=None, test_targets=None, run_eagerly=False, verbose=1):
    """
    Plots the observed speeds and the ones predicted by the FD model, depending on the mean spacing
    :param data_path: path of the file containing the data
    :param plot_title: title of the plot
    :param fd_epochs: number of epochs of training for the FD
    :return fd trained model
    """
    fd_data, fd_targets = read_dataset(data_path, fd_training=True)


    # to stop the computation when model is at its cap
    callback = EarlyStopping(monitor='loss', patience=10)  # default on val_loss

    # train the FD model
    model = FD_Network()
    model.compile(optimizer='adam', loss='mse', run_eagerly=run_eagerly)
    model.fit(x=fd_data, y=fd_targets, epochs=fd_epochs, verbose=verbose, callbacks=[callback])

    # generate the FD speeds with prediction
    stop = np.max(fd_data) * 1.5
    mean_spacings = np.expand_dims(np.linspace(start=0.5, stop=stop, num=1000), axis=1)
    if test_data is not None:
        mean_spacings = test_data
    fd_speeds = model.predict(x=mean_spacings)
    if test_targets is not None:
        model.mse = np.mean((fd_speeds-test_targets)**2)


    # plot the FD prediction over the observations
    plt.plot(mean_spacings, fd_speeds, c='orange')  # fd model data
    plt.scatter(fd_data, fd_targets, s=1)  # original data
    plt.xlabel("Mean spacing")
    plt.ylabel("Speed")
    plt.title(plot_title)
    plt.show()
    return model


def plot_fd_and_speeds(data_path: str, plot_title: str = "", fd_epochs: int = 50, nn_epochs: int = 50, hidden_dims: Tuple[int] = (3,),
                       hidden_activation_func: str = "sigmoid", dropout = -1, training_plots: bool = True, run_eagerly=False, verbose=1):
    """
    Plots the speeds predicted by the network and the FD curve depending on the mean spacing
    :param data_path: path of the file containing the data
    :param plot_title: title of the plot
    :param fd_epochs: number of epochs of training for the FD
    :param nn_epochs: number of epochs of training for the NN
    :param hidden_dims: tuple containing the dimensions of the hidden layers of the NN
    :param training_plots: if True, plots the training curves of the FD and NN
    :return nn model for predicting speeds and model for approximating weidmann
    """
    fd_data, fd_targets = read_dataset(data_path, fd_training=True)
    nn_data, nn_targets = read_dataset(data_path, fd_training=False)

    # to stop the computation when model is at its cap
    callback = EarlyStopping(monitor='loss', patience=10)  # default on val_loss

    # train the speed predictor neural network
    print("Training the NN model..")
    nn = create_nn(hidden_dims, dropout=dropout)
    nn.compile(optimizer='adam', loss='mse')
    hist = nn.fit(x=nn_data, y=nn_targets, epochs=nn_epochs, callbacks=[callback], verbose=verbose)
    loss_nn = hist.history['loss']

    # create the speed for FD to learn
    nn_speeds = nn.predict(x=nn_data)

    # train the FD model
    print("Training the FD model..")
    model = FD_Network()
    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(x=fd_data, y=nn_speeds, epochs=fd_epochs, callbacks=[callback], run_eagerly=run_eagerly, verbose=verbose)
    loss_fd = hist.history['loss']

    # training plots
    if training_plots:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
        # FD
        ax[0].plot(loss_fd)
        ax[0].set_title("FD training")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("MSE")
        # NN
        ax[1].plot(loss_nn, c='red')
        ax[1].set_title("NN training")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("MSE")
        fig.show()

    # plot
    stop = np.max(fd_data) * 1.5
    mean_spacings = np.expand_dims(np.linspace(start=0.5, stop=stop, num=1000), axis=1)
    fd_speeds = model.predict(x=mean_spacings)
    fig, ax = plt.subplots(1, 1)
    ax.plot(mean_spacings, fd_speeds, c='orange')
    ax.scatter(nn_data[:, 0], nn_speeds, s=1, c='red')
    ax.set_xlabel("Mean spacing")
    ax.set_ylabel("Speed")
    fig.suptitle(plot_title)
    plt.show()

    return nn, model


def get_all_result_data(results):
    tr_mean = []
    tr_std = []
    val_mean = []
    val_std = []
    test_mean = []
    test_std = []
    for key in results.keys():
        tr_mean.append(results[key]['tr'][0])
        tr_std.append(results[key]['tr'][1])
        val_mean.append(results[key]['val'][0])
        val_std.append(results[key]['val'][1])
        test_mean.append(results[key]['test'][0])
        test_std.append(results[key]['test'][1])
    return tr_mean, tr_std, val_mean, val_std, test_mean, test_std


def plot_results(results, tr_mean, tr_std, val_mean, val_std, test_mean, test_std, plot_val=False, title=""):
    fig, ax = plt.subplots()
    ax.fill_between(range(len(tr_std)), [tr_mean[i] + tr_std[i] for i in range(len(tr_std))], [tr_mean[i] - tr_std[i] for i in range(len(tr_std))],
                    alpha=0.2, color='orange')
    ax.plot(tr_mean, label='training_loss', c='orange')
    ax.scatter(range(len(tr_mean)), tr_mean, c='orange')

    if plot_val:
        ax.fill_between(range(len(val_std)), [val_mean[i] + val_std[i] for i in range(len(val_std))],
                        [val_mean[i] - val_std[i] for i in range(len(val_std))], alpha=0.2)
        ax.plot(val_mean, label='validation_loss')
        ax.scatter(range(len(val_mean)), val_mean, c='blue')

    ax.fill_between(range(len(test_std)), [test_mean[i] + test_std[i] for i in range(len(test_std))],
                    [test_mean[i] - test_std[i] for i in range(len(test_std))], alpha=0.2, color='red')
    ax.plot(test_mean, label='testing_loss', c='red')
    ax.scatter(range(len(test_mean)), test_mean, c='red')
    plt.legend()
    ax.set_ylabel('MSE')
    ax.set_xlabel('model conf')
    ax.set_xticks(range(len(results.keys())), labels=results.keys())
    plt.title(title)
    plt.show()


def _get_data_for_train_both_models(base_path, task_data, train: bool):
    """
    Internal usage function, called from 'train_both_models' to get the training/testing data
    :param base_path:
    :param task_data:
    :param train:
    :return:
    """
    X_t, y_t = None, None
    fd_x_t = None
    for data in task_data:
        path = base_path + f"train_{data}_data"
        try:
            f = open(path)
        except IOError:
            create_and_save_training_testing_data(data, base_path)

        X_train, y_train, X_test, y_test = read_train_test(data, base_path)
        if train:
            X = X_train
            y = y_train
        else:
            X = X_test
            y = y_test
        fd_x = X[:, 0].reshape(-1, 1)

        if X_t is None:
            X_t = X
            y_t = y
            fd_x_t = fd_x
        else:
            X_t = np.concatenate((X_t, X), axis=0)
            y_t = np.concatenate((y_t, y), axis=0)
            fd_x_t = np.concatenate((fd_x_t, fd_x), axis=0)

    X, y, fd_x = X_t, y_t, fd_x_t
    return X, y, fd_x


def train_both_models(task_train, task_test):
    base_path = "../data/training_data/"
    X_train, y_train, fd_x_train = _get_data_for_train_both_models(base_path=base_path, task_data=task_train, train=True)
    X_test, y_test, fd_x_test = _get_data_for_train_both_models(base_path=base_path, task_data=task_test, train=False)

    # X_t, y_t = None, None
    # fd_x_t = None
    # for train_data in task_train:
    #     training_path = base_path + f"train_{train_data}_data"
    #     try:
    #         f = open(training_path)
    #     except IOError:
    #         create_and_save_training_testing_data(train_data, base_path)
    #
    #     X_train, y_train, _, _ = read_train_test(train_data, base_path)
    #     fd_x_train = X_train[:, 0].reshape(-1, 1)
    #
    #     if X_t is None:
    #         X_t = X_train
    #         y_t = y_train
    #         fd_x_t = fd_x_train
    #     else:
    #         X_t = np.concatenate((X_t, X_train), axis=0)
    #         y_t = np.concatenate((y_t, y_train), axis=0)
    #         fd_x_t = np.concatenate((fd_x_t, fd_x_train), axis=0)
    #
    # X_train, y_train, fd_x_train = X_t, y_t, fd_x_t
    #
    # X_t, y_t = None, None
    # fd_x_t = None
    # for test_data in task_test:
    #     testing_path = base_path + f"train_{test_data}_data"
    #     try:
    #         f = open(testing_path)
    #     except IOError:
    #         create_and_save_training_testing_data(test_data, base_path)
    #
    #     _, _, X_test, y_test = read_train_test(test_data, base_path)
    #     fd_x_test = X_test[:, 0].reshape(-1, 1)
    #
    #     if X_t is None:
    #         X_t = X_test
    #         y_t = y_test
    #         fd_x_t = fd_x_test
    #     else:
    #         X_t = np.concatenate((X_t, X_test), axis=0)
    #         y_t = np.concatenate((y_t, y_test), axis=0)
    #         fd_x_t = np.concatenate((fd_x_t, fd_x_test), axis=0)
    # X_test, y_test, fd_x_test = X_t, y_t, fd_x_t

    # train fd
    model = FD_Network()
    fd_losses = bootstrapped_cv(hidden_dims=None, data=fd_x_train, targets=y_train, test_data=fd_x_test, test_targets=y_test,
                                kfolds=5, epochs=1000, batch_size=32, n_bootstraps=15, bootstrap_dim=1000, model=model)
    # train speed nn
    hidden_dims = (3,)
    nn_losses = bootstrapped_cv(hidden_dims=hidden_dims, data=X_train, targets=y_train, test_data=X_test, test_targets=y_test,
                                kfolds=5, epochs=1000, batch_size=32, n_bootstraps=15, bootstrap_dim=1000)

    # once we have the selection stats, train an nn on the whole train to give predictions needed for FD model training
    nn = create_nn(hidden_dims, dropout=-1)
    nn.compile(optimizer='adam', loss='mse')
    # to stop the computation when model is at its cap
    callback = EarlyStopping(monitor='loss', patience=10)  # default on val_loss
    nn.fit(x=X_train, y=y_train, epochs=1000, callbacks=[callback], verbose=0)
    fd_nn_prediction_speeds = nn.predict(x=X_train)

    # train fd
    model = FD_Network()
    fd_prediction_losses = bootstrapped_cv(hidden_dims=None, data=fd_x_train, targets=fd_nn_prediction_speeds, test_data=fd_x_test,
                                           test_targets=y_test, kfolds=5, epochs=1000, batch_size=32, n_bootstraps=15, bootstrap_dim=1000,
                                           model=model)
    return nn_losses, fd_losses, fd_prediction_losses


if __name__ == '__main__':
    path = "../data/vadere_bottleneck_100_90"
    plot_fd_and_speeds(data_path=path)

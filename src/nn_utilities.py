import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utilities import *


def _create_sets_from_folds(data_folds: list, targets_folds: list, val_fold: int):
    """
    From a list containing the folds, creates actual training and validation sets
    :param data_folds:
    :param targets_folds:
    :return:
    """
    assert len(data_folds) == len(targets_folds)
    kfolds = len(data_folds)
    val_data = data_folds[val_fold]
    val_targets = targets_folds[val_fold]
    tr_folds = list(range(kfolds))
    tr_folds.remove(val_fold)
    tr_data = data_folds[tr_folds[0]]
    tr_targets = targets_folds[tr_folds[0]]
    for tr_fold in tr_folds[1:]:
        tr_data = np.concatenate((tr_data, data_folds[tr_fold]), axis=0)
        tr_targets = np.concatenate((tr_targets, targets_folds[tr_fold]), axis=0)
    return tr_data, tr_targets, val_data, val_targets


def create_nn(hidden_dims, dropout=-1):
    """
    Create feed-forward fully-connected neural network with eventual dropout
    :param hidden_dims:
    :param dropout:
    :return:
    """
    layers = [Dense(units=d, activation='sigmoid') for d in hidden_dims] + [Dense(units=1, activation='linear')]
    # add dropout if needed
    if dropout != -1:  # user asks for dropout
        if dropout >= 1:
            print("Dropout value is too high (has to be less than 1)")
        else:
            for j in range(len(layers)):
                if type(layers[j]) == Dense and j != len(layers) - 1:
                    layers.insert(j + 1, Dropout(dropout))
            print("Added dropout:", layers)
    model = Sequential(layers)
    return model


def cross_validation(hidden_dims: Tuple[int], data: np.ndarray, targets: np.ndarray, test_data: np.ndarray,
                     test_targets: np.ndarray, kfolds: int, epochs: int, batch_size, dropout=-1, model=None):
    """
    Performs cross validation
    :param kfolds:
    :return:
    """
    # random shuffle data and split input and output
    indexes = list(range(len(data)))
    np.random.shuffle(indexes)
    data = data[indexes]
    targets = targets[indexes]
    data_folds = np.array_split(data, kfolds)  # divide the data in k equal folds
    targets_folds = np.array_split(targets, kfolds)  # divide the data in k equal folds
    losses = {'tr': [], 'val': [], 'test': []}
    callback = EarlyStopping(patience=10)  # default on val_loss
    for val_fold in range(kfolds):
        tr_data, tr_targets, val_data, val_targets = _create_sets_from_folds(data_folds, targets_folds, val_fold)
        if model is None:
            model = create_nn(hidden_dims=hidden_dims, dropout=dropout)
        batch_size = batch_size if batch_size is not None else len(tr_data)
        model.compile(optimizer='adam', loss='mse')
        hist = model.fit(x=tr_data, y=tr_targets, epochs=epochs,
                         batch_size=batch_size, validation_data=(val_data, val_targets), callbacks=[callback])
        losses['tr'].append(hist.history['loss'][-1])
        losses['val'].append(hist.history['val_loss'][-1])

        # predict on test data
        pred = model.predict(test_data, batch_size=batch_size)
        losses['test'].append(np.mean((test_targets - pred)**2))
    return {'tr': np.mean(losses['tr']), 'val': np.mean(losses['val']), 'test': np.mean(losses['test'])}


def bootstrapped_cv(hidden_dims: Tuple[int], data: np.ndarray, targets: np.ndarray, test_data: np.ndarray, test_targets: np.ndarray, kfolds: int,
                    epochs: int, n_bootstraps, bootstrap_dim, batch_size, dropout=-1, model=None):
    bootstrap_losses = {'tr': [], 'val': [], 'test': []}
    for i in range(n_bootstraps):
        indexes = np.arange(len(data))
        indexes = np.random.choice(indexes, size=bootstrap_dim, replace=True)
        data_bootstrap = data[indexes]
        targets_bootstrap = targets[indexes]
        cv_losses = cross_validation(hidden_dims=hidden_dims, data=data_bootstrap, targets=targets_bootstrap,
                                     test_data=test_data, test_targets=test_targets, dropout=dropout,
                                     kfolds=kfolds, epochs=epochs, batch_size=batch_size, model=model)
        bootstrap_losses['tr'].append(cv_losses['tr'])
        bootstrap_losses['val'].append(cv_losses['val'])
        bootstrap_losses['test'].append(cv_losses['test'])
    bootstrap_losses = {'tr': (np.mean(np.array(bootstrap_losses['tr'])), np.std(np.array(bootstrap_losses['tr']))),
                        'val': (np.mean(np.array(bootstrap_losses['val'])), np.std(np.array(bootstrap_losses['val']))),
                        'test': (np.mean(np.array(bootstrap_losses['test'])), np.std(np.array(bootstrap_losses['test'])))}
    return bootstrap_losses


def create_and_save_training_testing_data(task: str, base_path: str, test_size=0.5):
    """
    check if training data exists, otherwise create it and save it
    :param task: needed for full path composition
    :param base_path: directory path
    :param test_size: percentage of how much of the training set to transform in test
    :return:
    """
    print("Training and testing data do not exist, creating it..")
    data, targets = read_dataset(f"../data/dataset_{task}")
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=test_size, random_state=100)

    # save input and output of both training and testing
    with open(base_path + f"train_{task}_data", 'wb') as f:
        np.save(f, X_train)
    with open(base_path + f"train_{task}_targets", 'wb') as f:
        np.save(f, y_train)
    with open(base_path + f"test_{task}_data", 'wb') as f:
        np.save(f, X_test)
    with open(base_path + f"test_{task}_targets", 'wb') as f:
        np.save(f, y_test)


def read_train_test(task: str, base_path: str):
    """
    read all the data needed for model training
    :param task: string defining the particular task e.g. bottleneck070
    :param base_path: directory path
    :return: training input, training output, test input, test output
    """
    with open(base_path + f"train_{task}_data", 'rb') as f:
        X_train = np.load(f)
    with open(base_path + f"train_{task}_targets", 'rb') as f:
        y_train = np.load(f)
    with open(base_path + f"test_{task}_data", 'rb') as f:
        X_test = np.load(f)
    with open(base_path + f"test_{task}_targets", 'rb') as f:
        y_test = np.load(f)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    tasks = ["corridor_30", "corridor_85", "corridor_140", "bottleneck_070", "bottleneck_120", "bottleneck_180"]
    task = tasks[5]
    base_path = "../data/training_data/"

    training_path = base_path + f"train_{task}_data"
    try:
        f = open(training_path)
    except IOError:
        create_and_save_training_testing_data(task, base_path)
        
    print("Working on task:", task)

    hidden_dims = [(1,), (2,), (3,), (4, 2), (5, 2), (5, 3), (6, 3), (10, 4)]  #[(1,), (2,), (3,), (4, 2), (5, 2), (5, 3), (6, 3), (10, 4)]
    dropouts = [-1, -1, -1, -1, -1, -1, -1, -1]
    X_train, y_train, X_test, y_test = read_train_test(task, base_path)

    res_bootstrap_losses = {}
    for i in range(len(hidden_dims)):
        res_bootstrap_losses[str(hidden_dims[i])+"-"+str(dropouts[i])] = bootstrapped_cv(hidden_dims=hidden_dims[i],
                                                         data=X_train, targets=y_train, test_data=X_test,
                                                         test_targets=y_test, kfolds=5, epochs=1000, batch_size=32,
                                                         n_bootstraps=5, bootstrap_dim=5000, dropout=dropouts[i])
    with open(f"../data/results_{task}.txt", "w") as f:
        print(res_bootstrap_losses, file=f)

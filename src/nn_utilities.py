import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense


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


def cross_validation(hidden_dims: Tuple[int], data: np.ndarray, targets: np.ndarray, kfolds: int, epochs: int, batch_size):
    """
    Performs cross validation
    :param kfolds:
    :return:
    """
    indexes = list(range(len(data)))
    np.random.shuffle(indexes)
    data = data[indexes]
    targets = targets[indexes]
    data_folds = np.array_split(data, kfolds)
    targets_folds = np.array_split(targets, kfolds)
    losses = {'tr': [], 'val': []}
    for val_fold in range(kfolds):
        tr_data, tr_targets, val_data, val_targets = _create_sets_from_folds(data_folds, targets_folds, val_fold)
        layers = [Dense(units=d, activation='sigmoid') for d in hidden_dims] + [Dense(units=1, activation='linear')]
        model = Sequential(layers)
        batch_size = batch_size if batch_size is not None else len(tr_data)
        model.compile(optimizer='adam', loss='mse')
        hist = model.fit(x=tr_data, y=tr_targets, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_targets))
        losses['tr'].append(hist.history['loss'])
        losses['val'].append(hist.history['val_loss'])
    return {'tr': np.mean(losses['tr']), 'val': np.mean(losses['val'])}


def bootstrapped_cv(hidden_dims: Tuple[int], data: np.ndarray, targets: np.ndarray, kfolds: int, epochs: int, n_bootstraps, bootstrap_dim, batch_size):
    bootstrap_losses = {'tr': 0, 'val': 0}
    for i in range(n_bootstraps):
        indexes = np.arange(len(data))
        indexes = np.random.choice(indexes, size=bootstrap_dim, replace=True)
        data_bootstrap = data[indexes]
        targets_bootstrap = targets[indexes]
        cv_losses = cross_validation(hidden_dims=hidden_dims, data=data_bootstrap, targets=targets_bootstrap, kfolds=kfolds, epochs=epochs, batch_size=batch_size)
        bootstrap_losses['tr'] += cv_losses['tr']
        bootstrap_losses['val'] += cv_losses['val']
    bootstrap_losses['tr'] /= n_bootstraps
    bootstrap_losses['val'] /= n_bootstraps
    return bootstrap_losses


if __name__ == '__main__':
    from utilities import read_dataset
    data, targets = read_dataset("../data/dataset_bottleneck_070")
    print(bootstrapped_cv(hidden_dims=(3,), data=data, targets=targets, kfolds=5, epochs=20, batch_size=300, n_bootstraps=10, bootstrap_dim=1000))

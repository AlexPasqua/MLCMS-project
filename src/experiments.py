import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from utilities import read_dataset


if __name__ == '__main__':
    dataset = read_dataset("../data/dataset_corridor_30.pickle")
    exit()

    knn = 10
    neural_network = Sequential(
        Dense(units=4, input_shape=(2 * knn + 1,), activation='sigmoid'),
        Dense(units=2, activation='sigmoid'),
        Dense(units=1, activation='linear')
    )

    neural_network.compile(optimizer='sgd', lr=0.1, loss='mse')
    neural_network.fit(x=data, y=targets)

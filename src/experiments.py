import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer


if __name__ == '__main__':
    data = pd.read_pickle("../data/extended_corridor_df")
    print(data.head())

    net = Sequential(Dense(units=3, input_shape=()))

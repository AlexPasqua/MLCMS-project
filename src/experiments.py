import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer


if __name__ == '__main__':
    data = pd.read_pickle("../data/extended_corridor_df")
    print(data.head())

    knn = 10
    neural_network = Sequential(
        Dense(units=4, input_shape=(2 * knn + 1,), activation='sigmoid'),
        Dense(units=2, activation='sigmoid'),
        Dense(units=1, activation='linear')
    )




import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utilities import read_dataset


if __name__ == '__main__':
    data, targets = read_dataset("../data/dataset_corridor_30.pickle")

    knn = 10
    neural_network = Sequential([
        Dense(units=4, input_shape=(2 * knn + 1,), activation='sigmoid'),
        Dense(units=2, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    neural_network.compile(optimizer='sgd', loss='mse')
    history = neural_network.fit(x=data, y=targets, epochs=50)
    loss_history = history.history['loss']
    plt.plot(loss_history)
    plt.show()

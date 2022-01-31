import tensorflow as tf
import numpy as np
from utilities import read_dataset
import matplotlib.pyplot as plt


class FD_Network(tf.keras.Model):
    def __init__(self):
        super(FD_Network, self).__init__()
        # define all layers in init
        # input layer
        self.input_layer = tf.keras.layers.Dense(10)
        # output layer
        self.desired_speed = tf.keras.layers.Dense(1)
        self.pedestrian_size = tf.keras.layers.Dense(1)
        self.time_gap = tf.keras.layers.Dense(1)

    def call(self, mean_spacing):
        x = self.input_layer(mean_spacing)
        x = tf.keras.activations.sigmoid(x)
        v0 = self.desired_speed(x)
        vo = tf.keras.activations.softplus(v0)  # if bidirectional not needed!
        l = self.pedestrian_size(x)
        l = tf.keras.activations.softplus(l)
        t = self.time_gap(x)
        t = tf.keras.activations.softplus(t)
        return v0 * (1 - tf.exp((l - mean_spacing) / (v0 * t)))  # TODO check dimensions and operation v0 * t


if __name__ == "__main__":
    data, targets = read_dataset("../data/vadere_corridor_100", fd_training=True)
    model = FD_Network()
    model.compile(optimizer='sgd', loss='mse')
    history = model.fit(x=data, y=targets, epochs=50)
    loss_history = history.history['loss']
    plt.plot(loss_history)
    plt.show()

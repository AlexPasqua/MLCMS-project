import tensorflow as tf


class FD_Network(tf.keras.Model):
    def __init__(self):
        super(FD_Network, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(10)
        # output layers producing the 3 parameters of the FD
        self.desired_speed = tf.keras.layers.Dense(1)
        self.pedestrian_size = tf.keras.layers.Dense(1)
        self.time_gap = tf.keras.layers.Dense(1)

    def call(self, mean_spacing):
        x = self.hidden_layer(mean_spacing)
        x = tf.keras.activations.sigmoid(x)
        v0 = self.desired_speed(x)
        v0 = tf.keras.activations.softplus(v0)  # if bidirectional not needed!
        l = self.pedestrian_size(x)
        l = tf.keras.activations.softplus(l)
        t = self.time_gap(x)
        t = tf.keras.activations.softplus(t)
        return v0 * (1 - tf.exp((l - mean_spacing) / (v0 * t)))

import tensorflow as tf
from tensorflow.keras import layers, activations


class SpeedPredictor(tf.keras.Model):
    def __init__(self):
        super(SpeedPredictor, self).__init__()
        self.hidden = layers.Dense(3)
        self.output_layer = layers.Dense(1)

    def call(self, x):
        x = self.hidden(x)
        x = activations.sigmoid(x)
        return self.output_layer(x)

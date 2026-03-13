import tensorflow as tf
from tensorflow.keras import layers

class Head(tf.keras.layers.Layer):
    def __init__(self, d_token, out_dim=1):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.linear = layers.Dense(out_dim)

    def call(self, TL):
        # TL shape: (B, tokens, d_token)
        cls = TL[:, 0]  # (B, d_token)
        x = self.norm(cls)
        x = tf.nn.relu(x)
        return self.linear(x)  # (B, out_dim)

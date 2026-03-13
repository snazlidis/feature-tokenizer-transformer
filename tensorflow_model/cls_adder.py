import tensorflow as tf
from tensorflow.keras import layers

class CLSAdder(tf.keras.layers.Layer):
    def __init__(self, d_token):
        super().__init__()
        # Learnable CLS token: shape (1, 1, d_token)
        # Keras uses add_weight to create trainable parameters
        self.cls_token = self.add_weight(
            shape=(1, 1, d_token),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )

    def call(self, T):
        # T shape: (B, N, d_token)
        B = tf.shape(T)[0]
        
        # Duplicate CLS token for each batch item: (B, 1, d_token)
        cls = tf.tile(self.cls_token, [B, 1, 1])

        # Concatenate: CLS in front of tokens
        return tf.concat([cls, T], axis=1)
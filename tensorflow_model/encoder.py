import tensorflow as tf
from tensorflow.keras import layers

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_token, n_heads, d_ff, dropout):
        super().__init__()

        # Self-attention
        self.att = layers.MultiheadAttention(
            num_heads=n_heads,
            # MUST: d_token % n_heads == 0
            key_dim=d_token//n_heads,
            dropout=dropout
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)

        # Feed-forward network
        self.ff = tf.keras = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_token),
        ])

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Multi-head attention block

        """ 
            attn_output = self.att(x, x, training=training)
            x = x + self.dropout2(ff_output, training=training)
            x = self.norm2(x)
        """
        y = self.norm1(x)
        attn_out = self.att(y, y, training=training)
        x = x + self.dropout1(attn_out, training=training)

        y = self.norm2(x)
        ff_out = self.ff(y, y, training=training)
        x = x + self.dropout2(ff_out, training=training)
        return x
    
class FTEncoder(tf.keras.layers.Layer):
    def __init__(self, d_token, n_layers=3, n_heads=4, d_ff=128,dropout=0.1):
        super().__init__()
        self.layers = [
            TransformerEncoderBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
    
    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x
    

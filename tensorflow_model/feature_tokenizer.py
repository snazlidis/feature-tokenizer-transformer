import tensorflow as tf
from tensorflow.keras import layers

class FeatureTokenizer(tf.keras.layers.Layer):
    def __init__(self, num_numeric_features, categorical_cardinalities, d_token):
        super().__init__()
        self.d_token = d_token

        # Numeric projections: one Dense(1 -> d_token) per numeric feature
        self.numeric_proj = [
            layers.Dense(d_token)
            for _ in range(num_numeric_features)
        ]
        # Categorical Embeddings: one embedding per categorical feature
        self.category_embeddings = [
            layers.Embedding(cardinality + 1, d_token )
            for cardinality in categorical_cardinalities
        ]
    
    def call(self, x_cat, x_num):
        # x_cat: (Batch, num_cat)
        # x_num: (Batch, num_num)
        tokens = []

        # Process numeric features
        for i,proj in enumerate(self.numeric_proj):
            # Column i: Shape (B, )->(B, 1)
            col = tf.expand_dims(x_num[:, i], axis=-1)
            # Linear projection: (B, 1)->(B, d_token)
            tok = proj(col)
            tokens.append(tok)

        # Process categorical features
        x_cat_shifted = tf.cast(x_cat, tf.int32) + 1
        for i, emb in enumerate(self.category_embeddings):
            tok = emb(x_cat_shifted[:, i])
            tokens.append(tok)
        
        # Stack tokens: (Batch, num_tokens, d_token)
        T = tf.stack(tokens, axis=1)
        return T


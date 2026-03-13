from tensorflow.keras import layers, Model
from tokenizer import FeatureTokenizer, CLSAdder
from encoder import FTEncoder, Head

class FTTransformer(Model):
    def __init__(self,
                 num_numeric_features,
                 categorical_cardinalities,
                 d_token=32,
                 n_layers=4,
                 n_heads=4,
                 d_ff=256,
                 dropout=0.2,
                 out_dim=1):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            num_numeric_features=num_numeric_features,
            categorical_cardinalities=categorical_cardinalities,
            d_token=d_token
        )
        # Output shape of CLS: (B, num_feats+1, d_token)
        self.cls_adder = CLSAdder(d_token=d_token)

        self.backbone = FTEncoder(
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        self.head = Head(
            d_token=d_token,
            out_dim=out_dim
        )

    def call(self, inputs, training=False):
        
        x_cat, x_num = inputs
        # Tokenizer output: (B, num_feats, d_token)
        T = self.tokenizer(x_cat, x_num, training=training)
        T0 = self.cls_adder(T)

        TL = self.backbone(T0, training=training)
        logits = self.head(TL)

        return logits
    
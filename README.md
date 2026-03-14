# Feature Tokenizer Transformer

A Tensorflow implementation of the **Feature Tokenizer Transformer (FT-Transformer)** for binary classification on tabular data. Based on the architecture proposed by Gorishniy
et al. in [*Revisiting Deep Learning Models for Tabular Data*](https://arxiv.org/abs/2106.11959)

---

## Overview

The FT transformer adapts the Transformer architecture to tabular data by converting every feature - both numeric and categorical - into a token (embedding vector). A special `[CLS]` token is prepended to the token sequence and passed through a standard Transformer encoder. The final representation of the `[CLS]` token is used for classification.

This implementation was developed as part of a diploma thesis and evaluated on the **NSL-KDD** network intrusion detection dataset.

---

## Architecture summary

```
Input (x_cat, x_num)
        │
        ▼
 FeatureTokenizer        # Projects each feature into a d_token-dimensional embedding
        │
        ▼
   CLSAdder              # Prepends a learnable [CLS] token to the sequence
        │
        ▼
   FTEncoder             # Stack of Transformer encoder blocks (Pre-LN)
        │
        ▼
     Head                # Extracts [CLS] token → LayerNorm → Linear projection
        │
        ▼
    Output (logits)
```

Each **Transformer Encoder Block** follows the Pre-LayerNorm design:
- Multi-Head self attention
- Feed-Forward network (GELU activation)
- Residual connections and dropout




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

---

## Requirements

```
tensorflow>=2.10
numpy
pandas
```
Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Dataset

The model was evaluated on the **NSL-KDD** dataset, a benchmark dataset for network intrusion detection. It contains a mix of numeric and categorical features representing network connection records, labeled as normal or attack traffic.

- [NSL-KDD Dataset](https:/www.unb.ca/cic/datasets/nsl.html)

## Reference 

```bibtex
@article{gorishniy2021reviesiting,
 title={Revisiting Deep Learning Models for Tabular Data},
 author={Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
 journal={Advances in neural Information Processing Systems},
 year={2021}
}
```



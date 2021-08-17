# AFEX
Attention-like Feature Explanation for Tabular Data [[Paper]](https://arxiv.org/abs/2108.04855)

AFEX is a new method for local and global explanation of a black-box model in case of tabular data.
It consists of two parts. The first part is a set of the one-feature neural networks. There are *k* distinct networks (basis shape functions) for each feature.
The second part produces shape functions corresponding to each feature as the weighted sum of the basis shape functions.
Weights are computed by using an attention-like mechanism.
The main advantage is that AFEX is trained an end-to-end manner on a whole dataset only once such that it does not require to train neural networks again at the explanation stage.
Also, AFEX identifies pairwise interactions between features based on pairwise multiplications of shape functions corresponding to different features.


import torch
from torch import nn
from typing import Optional, List
import itertools


class LeastSquares:
    """See https://github.com/pytorch/pytorch/issues/27036"""
    def __init__(self, max_attempts=10):
        self.max_attempts = max_attempts

    def lstq(self, A, Y, lamb=0.0, attempt=0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        try:
            rank = torch.linalg.matrix_rank(A)
        except:
            rank = -1
        if cols == rank:
            q, r = torch.linalg.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            if attempt > self.max_attempts:
                print(f"Can't compute pseudoinverse, attempt: {attempt}; lambda: {lamb}")
                raise Exception("Can't compute pseudoinverse")
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = self.lstq(A_dash, Y_dash, lamb=lamb, attempt=attempt + 1)
        return x


def _solve_pinverse(a, b, lamb=0.01, attempt=0, max_attempts=10):
    try:
        return a.pinverse() @ b
    except RuntimeError as ex:
        cols = a.shape[1]
        a_dash = a.permute(1, 0) @ a + lamb * torch.eye(cols)
        b_dash = a.permute(1, 0) @ b
        print(f"Can't compute pinverse, attempt: {attempt}; lambda: {lamb}; exception: {ex}")
        if attempt > max_attempts:
            raise Exception("Can't compute pinverse :(")
        return _solve_pinverse(a_dash, b_dash, lamb=lamb, attempt=attempt + 1)


class LinearRegressionAttention(nn.Module):
    def __init__(self, dim, add_bias=None, method: str = 'qr'):
        super().__init__()
        self.dim = dim
        self.add_bias = add_bias
        self.method = method

    def forward(self, q, k, v, need_weights=True):
        """Calculate attention.

        :param q: Query of shape (n_features,).
        :param k: Keys of shape (n_keys, n_features).
        :param v: Values of shape (n_keys, n_values_features).
        """
        # weights = torch.tensor(torch.linalg.lstsq(k.T, q).solution)
        if self.add_bias is None:
            left = k.T
        else:
            left = torch.cat((k.T, torch.ones(k.T.shape[0], 1)), axis=1)
        if self.method == 'qr' or isinstance(self.method, float):
            lamb = self.method if isinstance(self.method, float) else 0.1
            weights = LeastSquares().lstq(left, q, lamb=lamb)
        elif self.method == 'pinverse':
            weights = _solve_pinverse(left, q)
        else:
            raise ValueError(f"Incorrect linear regression solver method: {self.method}")
        right = v
        if self.add_bias is False:
            weights = weights[:-1]
        elif self.add_bias is True:
            right = torch.cat((v, torch.ones(1, v.shape[1])), axis=0)
        weighted_values = weights @ right
        if self.add_bias is True:
            weights = weights[:-1]
        if need_weights:
            return weighted_values, weights

        return weighted_values



class SimpleAttention(nn.Module):
    def __init__(self, dim, use_softmax: bool = False, classical: bool = False):
        super().__init__()
        self.dim = dim
        self.use_softmax = use_softmax
        self.classical = classical
        self.softmax = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self, q, k, v, need_weights=True):
        """Calculate attention.

        :param q: Query of shape (n_features,).
        :param k: Keys of shape (n_keys, n_features).
        :param v: Values of shape (n_keys, n_values_features).
        """
        eps = 1e-9
        if not self.classical:
            mk = (k - torch.mean(k, axis=1).view(-1, 1)) / (torch.std(k, axis=1).view(-1, 1) + eps)
            mq = (q - torch.mean(q)) / (torch.std(q) + eps)
        else:  # self.classical
            mk = k / torch.clamp(torch.linalg.norm(k, axis=0), min=eps)
            mq = q / torch.clamp(torch.linalg.norm(q, axis=0), min=eps)

        scores = mk @ mq
        weights = scores
        if self.use_softmax:
            weights = self.softmax(weights)
        weighted_values = weights @ v
        if need_weights:
            return weighted_values, weights

        return weighted_values


class PolynomialFeatures(nn.Module):
    def __init__(self, degree: int = 2, with_replacement: bool = False,
                 same_divider: int = 1, pairwise_fn: str = 'mul'):
        super().__init__()
        self.degree = degree
        self.with_replacement = with_replacement
        self.feature_indices: Optional[list] = None
        self.same_divider = same_divider
        self.pairwise_fn = pairwise_fn
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        assert x.ndim == 2
        assert self.degree == 2, "Only 2nd degree is supported"

        if self.feature_indices is None:
            indices = range(x.shape[1])
            if self.with_replacement:
                self.feature_indices = list(
                    itertools.combinations_with_replacement(indices, r=self.degree)
                )
            else:
                self.feature_indices = list(
                    itertools.combinations(indices, r=self.degree)
                )
                self.feature_indices = list(
                    filter(
                        lambda t: t[0] // self.same_divider != t[1] // self.same_divider,
                        self.feature_indices
                    )
                )
        # extended_x = torch.cat([x] + [
        #     reduce(lambda left, j: left * x[:, j], indices, 1).view(-1, 1)
        #     for indices in self.feature_indices
        # ], axis=1)
        pairwise_fns = dict(
            mul=(lambda a, b: a * b),
            min=(torch.min),
            max=(torch.max),
            l2=(lambda a, b: torch.abs(a - b)),
            mul_sigmoid=(lambda a, b: self.sigmoid(a * b)),
            mul_tanh=(lambda a, b: self.tanh(a * b)),
        )

        if isinstance(self.pairwise_fn, str):
            fn = pairwise_fns[self.pairwise_fn]
        else:
            fn = self.pairwise_fn

        extended_x = torch.cat([x] + [
            (
                fn(x[:, indices[0]], x[:, indices[1]])
            ).view(-1, 1)
            for indices in self.feature_indices
        ], axis=1)

        return extended_x


class AttentionExplanationModel(nn.Module):
    def __init__(self, n_samples=20, n_attentions=8,
                 feature_nns: Optional[List[nn.Module]] = None,
                 target_nns: Optional[List[nn.Module]] = None,
                 use_pairwise_features: bool = False,
                 use_quadratic_features: bool = False,
                 pairwise_fn: str = 'mul',
                 n_nns_per_feature: int = 1,
                 attention_type: str = 'simple',
                 attention_params: Optional[dict] = None):
        super().__init__()
        self.n_samples = n_samples
        if attention_params is None:
            attention_params = {}
        self._n_features = -1

        disable_linear = attention_params.pop('disable_linear', False)

        if attention_type in ['simple', 'simple_softmax', 'classical']:
            self.feature_attentions = nn.ModuleList(modules=[
                SimpleAttention(
                    self.n_samples,
                    use_softmax=(attention_type != 'simple'),
                    classical=(attention_type == 'classical'),
                    **attention_params
                )
                for _ in range(n_attentions)
            ])
        elif attention_type == 'linear_regression':
            self.feature_attentions = nn.ModuleList(modules=[
                LinearRegressionAttention(
                    self.n_samples,
                    **attention_params
                )
                for _ in range(n_attentions)
            ])
        else:
            raise ValueError(f'Incorrect attention type: "{attention_type}"')


        if feature_nns is not None:
            self.feature_nns = nn.ModuleList(modules=feature_nns)
        else:
            self.feature_nns = None
        self.n_nns_per_feature = n_nns_per_feature

        if target_nns is not None:
            self.target_nns = nn.ModuleList(modules=target_nns)
        else:
            self.target_nns = None

        if use_pairwise_features:
            self.feature_transform = PolynomialFeatures(
                degree=2,
                with_replacement=use_quadratic_features,
                same_divider=self.n_nns_per_feature,
                pairwise_fn=pairwise_fn
            )
        else:
            self.feature_transform = None

        if disable_linear:
            self.linear = lambda t: t
        else:
            self.linear = nn.Linear(n_attentions, 1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def transform_features(self, x):
        """Apply feature NNs to each feature separately
           and then general feature transform (e.g. `PolynomialFeatures`).

        :param x: Input tensor.
        :return: Transformed tensor.
        """
        if self.feature_nns is not None:
            x = torch.cat([
                fnn(x[:, (i // self.n_nns_per_feature):(i // self.n_nns_per_feature) + 1])
                for i, fnn in enumerate(self.feature_nns)
            ], axis=1)
        if self.feature_transform is not None:
            x = self.feature_transform(x)
        return x

    def feature_indices(self):
        if self.feature_nns is not None:
            # number of generated features after applying feature NNs
            n_features = len(self.feature_nns)
        else:
            n_features = self._n_features
        indices = list(range(n_features))
        if self.feature_transform is not None:
            feature_transform_indices = set(
                i for pair in self.feature_transform.feature_indices
                for i in pair
            )
            assert set(indices) == feature_transform_indices
            indices += self.feature_transform.feature_indices
        return indices

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.ndim == 2
        assert self.feature_nns is None or \
               x.shape[1] == len(self.feature_nns) // self.n_nns_per_feature
        assert self.target_nns is None or len(self.feature_attentions) == len(self.target_nns)
        self._n_features = x.shape[1]

        x = self.transform_features(x)
        flipped_x = torch.transpose(x, 0, 1)
        # flipped_x shape: (n_features, batch_size=1, n_samples)
        all_feature_vectors, all_feature_weights = [], []
        for i, feature_attention in enumerate(self.feature_attentions):
            if self.target_nns is not None:
                cur_query = self.target_nns[i](y).view(-1)
            else:
                cur_query = y.view(-1)
            feature_vectors, feature_weights = feature_attention(
                    cur_query,      # query
                    flipped_x,      # key
                    flipped_x,      # value
                    need_weights=True
            )
            # feature_vectors shape: (1, batch_size, n_samples)
            # feature_weights shape: (batch_size, 1, n_features)
            all_feature_vectors.append(feature_vectors.view(-1, 1))
            all_feature_weights.append(feature_weights.view(1, -1))

        all_feature_vectors = torch.cat(all_feature_vectors, 1)
        # all_feature_vectors shape: (batch_size * n_samples, n_attentions)

        out = self.linear(all_feature_vectors)
        return out, all_feature_weights, all_feature_vectors

    def get_contributions(self, feature_weights: List[torch.Tensor]) -> torch.Tensor:
        """Get feature contributions for given attention weights.

        :param feature_weights: List attention weights for each attention (target nn).
        :return: Tensor with feature contributions.
        """
        return self.linear.weight @ torch.cat(feature_weights, 0)


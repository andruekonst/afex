import numpy as np
import torch
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict
from .explanation_model import AttentionExplanationModel
from .models import WeightedShortcut
from .training_parameters import TrainingParameters
from .training import train
from .metrics import calc_metrics
from .graphs import plot_losses, plot_shape_functions, plot_pairwise_grid_shape_functions, \
                    plot_target_predictions
from .utils import deep_divide, collapse, min_max_normalize
from typing import Optional
from itertools import product


def pretrain_nn(model, xs, y, epochs=100, lr=1e-4):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _epoch in range(epochs):
        out = model(xs)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def init_fn(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def _get_alias(feature, aliases):
    if aliases is None:
        return ""
    if isinstance(feature, tuple):
        return "(" + ", ".join(map(lambda f: str(aliases[f]), feature)) + ")"
    return f"({aliases[feature]})"


def contributions_stats(cur_feature_names, cur_feature_contributions, name: str = '(?)',
                        feature_aliases: Optional[list] = None):
    cur_normalized_contributions = min_max_normalize(cur_feature_contributions)
    print(f"Normalized abs contributions for {name} features:")
    best_feature = np.argmax(cur_normalized_contributions)
    best_alias = _get_alias(cur_feature_names[best_feature], feature_aliases)
    print(f"Best feature: #{best_feature} = {cur_feature_names[best_feature]}{best_alias}")
    feature_indices = np.argsort(-cur_normalized_contributions)
    for ind in feature_indices:
        cur_alias = _get_alias(cur_feature_names[ind], feature_aliases)
        print(f"    [{cur_feature_names[ind]}{cur_alias}]: {cur_normalized_contributions[ind]}")


class ExplainableSurrogateModel(torch.nn.Module):
    def __init__(self, surrogate_model, explanation_model):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.explanation_model = explanation_model

    def feature_indices(self):
        return self.explanation_model.feature_indices()

    def transform_features(self, xs):
        return self.explanation_model.transform_features(xs)

    def get_contributions(self, feature_weights):
        return self.explanation_model.get_contributions(feature_weights)

    def forward(self, xs):
        return self.explanation_model(xs, self.surrogate_model(xs))

    def predict_all(self, xs):
        surrogate_out = self.surrogate_model(xs)
        return surrogate_out, self.explanation_model(xs, surrogate_out)


def run_training(
        train_data_gen,
        lr: float = 1e-2,
        epochs: int = 500,
        batch_size: int = 1,
        n_samples: int = 100,
        n_features: int = 5,
        use_weights: bool = False,
        use_pairwise_features: bool = False,
        pretrain_feature_nns: bool = False,
        use_shortcuts_for_fnns: bool = False,
        n_nns_per_feature: int = 1,
        attention_type: str = 'simple',
        make_feature_nn=None,
        return_training_result: bool = False,
        attention_params: Optional[dict] = None,
        surrogate_model=None,
        surrogate_model_lam=True,
        pairwise_fn: str = 'mul',
        use_double: bool = True,
    ):
    if attention_params is None:
        attention_params = {}
    if make_feature_nn is None:
        def make_feature_nn():
            return torch.nn.Sequential(
                torch.nn.Linear(1, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1)
            )

    training_params = TrainingParameters(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_weights=use_weights,
        fnn_l2=None,
        feature_weights_l1=None,
        verbose=False,
        use_surrogate_model=(surrogate_model_lam if surrogate_model is not None else False),
    )
    use_feature_nns = n_nns_per_feature > 0
    xs, y, _weights = train_data_gen()
    xs = torch.tensor(xs)
    y = torch.tensor(y)
    feature_nns = [
        make_feature_nn()
        for _i in range(n_features * n_nns_per_feature)
    ]
    if use_shortcuts_for_fnns:
        feature_nns = list(map(WeightedShortcut, feature_nns))

    if not use_feature_nns:
        feature_nns = None
        if pretrain_feature_nns:
            raise ValueError("Cannot pretrain feature nns when they are disabled")
        assert n_nns_per_feature == 1

    if feature_nns is not None:
        for fnn in feature_nns:
            fnn.apply(init_fn)

    if pretrain_feature_nns and feature_nns is not None:
        for i, fnn in enumerate(feature_nns):
            cur_xs = xs[:, i].float()
            pretrain_loss = pretrain_nn(fnn, cur_xs.view(-1, 1), cur_xs.view(-1, 1), epochs=500)
            print(f"Pretraining feature NN [{i}]: {pretrain_loss}")

    model = AttentionExplanationModel(
        n_samples=n_samples,
        n_attentions=1,
        feature_nns=feature_nns,
        target_nns=None,
        use_pairwise_features=use_pairwise_features,
        n_nns_per_feature=n_nns_per_feature,
        attention_type=attention_type,
        attention_params=attention_params,
        pairwise_fn=pairwise_fn
    )

    if use_double:
        model = model.double()

    if surrogate_model is not None:
        model = ExplainableSurrogateModel(
            surrogate_model=surrogate_model,
            explanation_model=model
        )

    training_result = train(model, data_gen=train_data_gen, params=training_params, callbacks=[])

    plot_losses(training_result.losses)

    if return_training_result:
        return model, training_result

    return model


DEFAULT_FNN_FIGPARAMS = {"figsize": (16, 9)}


def run_testing(
        model,
        test_data_gen,
        n_features: int,
        use_pairwise_features: bool = False,
        n_nns_per_feature: int = 1,
        recompute_weights: bool = False,
        grid_size: int = 25,
        fnn_figparams: Optional[dict] = None,
        pairwise_figparams: Optional[dict] = None,
        attention_type: str = 'simple',
        use_predicted_targets: bool = False,
        cat_features=None,
    ):
    if fnn_figparams is None:
        fnn_figparams = DEFAULT_FNN_FIGPARAMS
    if pairwise_figparams is None:
        pairwise_figparams = DEFAULT_FNN_FIGPARAMS
    feature_aliases = fnn_figparams.get('feature_aliases', None)

    print("Attention type:", attention_type)
    print("Contributions:")
    contributions = []
    metrics = defaultdict(lambda: [])

    xs, y, _weights = test_data_gen()
    xs = torch.tensor(xs)
    y = torch.tensor(y).view(-1, 1)
    if isinstance(model, ExplainableSurrogateModel):
        out, feature_weights, _feature_vectors = model(xs)
    else:
        out, feature_weights, _feature_vectors = model(xs, y)
    contributions.append(model.get_contributions(feature_weights).detach().numpy().ravel())
    for k, v in calc_metrics(y.detach().numpy().ravel(), out.detach().numpy().ravel()).items():
        metrics[k].append(v)

    contributions = np.stack(contributions, axis=0)
    contributions_mean = contributions.mean(axis=0)

    orig_y = y
    if use_predicted_targets:
        print("Using predicted targets")
        y = model.surrogate_model(xs)

    bias = 0
    if recompute_weights:
        with torch.no_grad():
            xs_embedding = model.transform_features(xs).detach().numpy()
        lasso = Lasso(alpha=1e-4)
        lasso.fit(xs_embedding, y.detach().numpy().ravel())
        contributions_mean = lasso.coef_
        print("Weights are recomputed with LASSO")
        print("Update target")
        out = torch.tensor(lasso.predict(xs_embedding))
        bias += lasso.intercept_

    print("MSE between predicted and target (real, not predicted):", mean_squared_error(
        orig_y.detach().numpy().astype(np.float32),
        out.detach().numpy().astype(np.float32)
    ))

    feature_indices = model.feature_indices()

    # generate a uniform grid
    # calculate std of each shape function network
    # and find shape functions

    def _make_grid_for_feature(xs, i):
        if cat_features is not None:
            if feature_aliases is not None and feature_aliases[i] in cat_features:
                unique = sorted(list(xs[:, i].unique()))
                avg_count = grid_size // len(unique)
                res = []
                for u in unique:
                    res.extend([u] * avg_count)
                if len(res) < grid_size:
                    res += [unique[-1]] * (grid_size - len(res))
                return torch.tensor(res).double()
        return torch.linspace(xs[:, i].min(), xs[:, i].max(), grid_size).double()

    pairwise_grid = None
    with torch.no_grad():
        xs_grid = torch.stack([
            _make_grid_for_feature(xs, i)
            for i in range(xs.shape[1])
        ], axis=1)
        transformed_features = model.transform_features(xs_grid)

        if use_pairwise_features:
            pairwise_grid = dict()
            for ind in product(range(xs.shape[1]), range(xs.shape[1])):
                if ind[0] == ind[1]:
                    continue
                cur_pairwise_xs = torch.zeros(grid_size * grid_size, xs.shape[1]).double()
                first, second = torch.meshgrid(
                    torch.linspace(xs[:, ind[0]].min(), xs[:, ind[0]].max(), grid_size).double(),
                    torch.linspace(xs[:, ind[1]].min(), xs[:, ind[1]].max(), grid_size).double()
                    # _make_grid_for_feature(xs, ind[0]),
                    # _make_grid_for_feature(xs, ind[1])
                )
                cur_pairwise_xs[:, ind[0]] = first.reshape(-1)
                cur_pairwise_xs[:, ind[1]] = second.reshape(-1)
                pairwise_grid[ind] = (cur_pairwise_xs, model.transform_features(cur_pairwise_xs))

    # it is a probably wrong way to estimate shape importance
    feature_stds = transformed_features.std(0).numpy().ravel()
    corrected_contributions = contributions_mean * feature_stds
    norm_contributions = corrected_contributions

    norm_contributions = {
        k: np.sum(v)
        for k, v in collapse(
            zip(feature_indices, norm_contributions),
            deep_divide(n_nns_per_feature)
        ).items()
    }
    shape_functions = {
        k: torch.stack(
            tuple(
                transformed_features[:, i] * contribution_mean
                for i, contribution_mean in values
            ),
            axis=1
        ).sum(axis=1)
        for k, values in collapse(zip(feature_indices, enumerate(contributions_mean)),
                                                  deep_divide(n_nns_per_feature)).items()
    }

    if use_pairwise_features:
        pairwise_grid_xs = {
            k: pairwise_grid[k][0].detach().numpy()
            for k, values in collapse(zip(feature_indices, enumerate(contributions_mean)),
                                                      deep_divide(n_nns_per_feature)).items()
            if isinstance(k, tuple)
        }
        pairwise_shape_functions = {
            k: torch.stack(
                tuple(
                    pairwise_grid[k][1][:, i] * contribution_mean
                    for i, contribution_mean in values
                ),
                axis=1
            ).sum(axis=1)
            for k, values in collapse(zip(feature_indices, enumerate(contributions_mean)),
                                                      deep_divide(n_nns_per_feature)).items()
            if isinstance(k, tuple)
        }

    feature_indices = list(norm_contributions.keys())
    assert feature_indices == list(shape_functions.keys())

    norm_contributions = np.array(list(norm_contributions.values()))
    norm_contributions = np.abs(norm_contributions)

    single_feature_names = [fn for fn in feature_indices if not isinstance(fn, tuple)]
    pairwise_feature_names = [fn for fn in feature_indices if isinstance(fn, tuple)]
    single_feature_contributions = [nc for fn, nc in zip(feature_indices, norm_contributions)
                                    if fn in single_feature_names]
    if use_pairwise_features:
        pairwise_feature_contributions = [nc for fn, nc in zip(feature_indices, norm_contributions)
                                          if fn in pairwise_feature_names]

    contributions_stats(single_feature_names, single_feature_contributions,
                        name='single',
                        feature_aliases=feature_aliases)
    if use_pairwise_features:
        contributions_stats(pairwise_feature_names, pairwise_feature_contributions,
                            name='pairwise',
                            feature_aliases=feature_aliases)
        contributions_stats(feature_indices, norm_contributions,
                            name='common',
                            feature_aliases=feature_aliases)

    # plot shape functions
    shape_functions_numpy = [sf.detach().numpy() for sf in shape_functions.values()]
    plot_shape_functions(
        xs_grid.detach().numpy(), shape_functions_numpy,
        n_features=n_features,
        subplots=True,
        path=None,
        cat_features=cat_features,
        **fnn_figparams
    )

    if use_pairwise_features:
        plot_pairwise_grid_shape_functions(
            pairwise_grid_xs,
            pairwise_shape_functions,
            subplots=True,
            path=None,
            **pairwise_figparams
        )

        # plot corrected shape functions (add corresponding individual shape functions)
        def correct_pairwise_shape_functions(indices, pairwise_sf):
            fst_feature, snd_feature = indices
            s = int(np.sqrt(pairwise_sf.shape[0]))
            shape = (s, s)
            result = pairwise_sf.detach().numpy().copy().reshape(shape)
            result += shape_functions_numpy[fst_feature].reshape((-1, 1))
            result += shape_functions_numpy[snd_feature].reshape((1, -1))
            return torch.tensor(result.ravel())

        corrected_pairwise_shape_functions = {
            k: correct_pairwise_shape_functions(k, v)
            for k, v in pairwise_shape_functions.items()
        }
        plot_pairwise_grid_shape_functions(
            pairwise_grid_xs,
            corrected_pairwise_shape_functions,
            subplots=True,
            path=None,
            **pairwise_figparams
        )

    target = y.detach().numpy().astype(np.float32)
    predicted_target = out.detach().numpy()
    plot_target_predictions(
        xs.detach().numpy(), target, predicted_target,
        features=(0, 1),
        path=None
    )
    print(
        "MSE between unbiased predicted and target:",
         mean_squared_error(
             target - target.mean(),
             predicted_target - predicted_target.mean()
         )
    )
    try:
        print("ROC-AUC between predicted and target:", roc_auc_score(target, predicted_target))
    except Exception:
        pass

    return single_feature_contributions


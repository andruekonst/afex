import numpy as np
from typing import Optional


def clf_checkerboard(xs: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Checkerboard problem.
    :param n_points: Input points.
    :param scale: Scale of board (how much cells will be on board).
    :return: `f(xs)`.
    """
    assert xs.shape[1] >= 2
    y = (np.sin(np.pi * xs[:, 0] * scale) > 0) ^ (np.sin(np.pi * xs[:, 1] * scale) > 0)
    return y


def reg_concat_square_linear(xs: np.ndarray) -> np.ndarray:
    return (xs[:, 1] >= 0.0) * (xs[:, 0] ** 2) + (xs[:, 1] < 0.0) * (xs[:, 0])


def model_generator(xs: np.ndarray, model=None) -> np.ndarray:
    return model.predict(xs)


def binary_clf_model_generator(xs: np.ndarray, model=None) -> np.ndarray:
    return model.predict_proba(xs)[:, 1]


DATA_GENERATORS = {
    'reg_additive_square': (lambda xs: xs[:, 0] ** 2 + 0.5 * xs[:, 1]),
    'reg_pairwise_step': (lambda xs: xs[:, 0] > xs[:, 1]),
    'reg_pairwise_mul': (lambda xs: xs[:, 0] * xs[:, 1]),
    'clf_checkerboard': clf_checkerboard,
    'reg_concat_square_linear': reg_concat_square_linear,
    'model': model_generator,
    'binary_clf_model': binary_clf_model_generator,
}

WEIGHT_FUNCTIONS = {
    'rbf': (lambda xs: np.exp(-(np.linalg.norm(xs - xs.mean(0), axis=1) ** 2) / 2))
}


def make_experiment_data(n_points=20, n_features=2,
                         cube_size: float = 3.0,
                         kind: str = 'pairwise_step',
                         weight_function: str = 'rbf',
                         generator_params: Optional[dict] = None,
                         center_params: Optional[dict] = None,
                         seed: Optional[int] = None,
                         generators: Optional[dict] = None,
                         weight_functions: Optional[dict] = None,
                         min_max: Optional[tuple] = None):
    if generators is None:
        generators = DATA_GENERATORS
    if weight_functions is None:
        weight_functions = WEIGHT_FUNCTIONS
    if generator_params is None:
        generator_params = {}
    rng = np.random.RandomState(seed)

    if center_params is None:
        center_params = {}
    center_shift = center_params.get('shift', 0.0)
    if center_params.get('type', 'random') == 'random':
        center_std = center_params.get('std', 0.1)
        center = rng.normal(center_shift, center_std, size=n_features)
    elif center_params.get('type', None) == 'value':
        center = np.array(center_shift)
    else:
        raise ValueError(f'Incorrect center_params type: {type}')

    if min_max is None:
        xs = rng.uniform(center - cube_size, center + cube_size,
                         size=(n_points, n_features))
    else:
        xs = rng.uniform(min_max[0], min_max[1],
                         size=(n_points, n_features))

    if kind not in generators:
        raise ValueError(f'No {kind} data generator found')
    if weight_function not in weight_functions:
        raise ValueError(f'No {weight_function} weight function found')

    y = (generators[kind](xs, **generator_params))
    weights = weight_functions[weight_function](xs)
    return xs.astype(np.float64), y.astype(np.float64), weights.astype(np.float64)


def make_notebook_data_gen(*args, **kwargs):
    def test_data_gen():
        return make_experiment_data(*args, **kwargs)

    return test_data_gen



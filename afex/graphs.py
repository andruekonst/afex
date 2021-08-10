import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, List, Tuple, Mapping, Union


def save_figure_to_file(path):
    plt.savefig(path, dpi=200)


def save_figure(plt_fn):
    def plt_fn_wrapper(*args, path: Optional[str] = None, **kwargs):
        plt.figure()
        plt_fn(*args, **kwargs)
        if path is not None:
            # plt.savefig(path)
            save_figure_to_file(path)

    return plt_fn_wrapper


@save_figure
def plot_losses(losses: np.ndarray):
    plt.plot(list(range(len(losses))), losses)


@save_figure
def plot_target_predictions(xs: np.ndarray, y: np.ndarray, out: np.ndarray,
                            features: Optional[Tuple[int, int]] = None):
    if features is None:
        features = (0, 1)
    xf, yf = features
    _fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].set_title('Predictions')
    ax[1].set_title('Target')
    ax[0].scatter(xs[:, xf], xs[:, yf], c=out.ravel())
    ax[1].scatter(xs[:, xf], xs[:, yf], c=y.ravel())


def plot_shape_functions(xs, shape_functions, n_features: Optional[int] = None,
                         feature_aliases: Optional[Mapping[int, str]] = None,
                         subplots: bool = False,
                         n_cols: int = 3,
                         same_scale: bool = True,
                         subtract_mean: bool = True,
                         path: Optional[str] = None,
                         x_min_max: Optional[tuple] = None,
                         cat_features: Optional[List[Union[str, int]]] = None,
                         **kwargs):
    if n_features is None:
        n_features = len(shape_functions)
    if subplots:
        plt.figure()
        _fig, axes = plt.subplots(n_features // n_cols + 1
                                  if n_features % n_cols != 0
                                  else n_features // n_cols,
                                  n_cols,
                                  **kwargs)
    if same_scale:
        y_lim = [np.inf, -np.inf]

        if not subtract_mean:
            for i in range(n_features):
                y_lim[0] = min(y_lim[0], np.min(shape_functions[i]))
                y_lim[1] = max(y_lim[1], np.max(shape_functions[i]))
        else:
            for i in range(n_features):
                y_lim[0] = min(y_lim[0], np.min(shape_functions[i] - shape_functions[i].mean()))
                y_lim[1] = max(y_lim[1], np.max(shape_functions[i] - shape_functions[i].mean()))

        lim_diff = y_lim[1] - y_lim[0]
        y_lim[0] -= lim_diff * 0.05
        y_lim[1] += lim_diff * 0.05
    else:
        y_lim = None

    for i in range(n_features):
        if not subplots:
            plt.figure()
            _fig, axes = plt.subplots(1, 1)
            ax = axes
        else:
            ax = axes.ravel()[i]
        x_plot = xs[:, i]  # .detach().numpy()
        y_plot = shape_functions[i]  # .detach().numpy()

        if subtract_mean:
            y_plot = y_plot - y_plot.mean()

        if feature_aliases is None:
            cur_feature_name = i
        else:
            cur_feature_name = feature_aliases[i]

        sort_ind = np.argsort(x_plot)
        if cat_features is not None and cur_feature_name in cat_features:
            ax.bar(x_plot[sort_ind], y_plot[sort_ind])
            # bar labels
            try:
                labels = cat_features[cur_feature_name]
            except:
                labels = sorted(np.unique(x_plot))
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
        else:
            ax.plot(x_plot[sort_ind], y_plot[sort_ind])

        title = f"{cur_feature_name}"
        ax.set_title(title)
        ax.set_xlabel(cur_feature_name)
        ax.set_ylabel('y')
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_min_max is not None:
            ax.set_xlim((x_min_max[0][i], x_min_max[1][i]))
        if not subplots and path is not None:
            assert '%' in path
            save_figure_to_file(path % i)
    plt.tight_layout()
    if subplots and path is not None:
        save_figure_to_file(path)


def plot_pairwise_shape_functions(xs: np.ndarray,
                                  feature_names: List[str],
                                  shape_functions: List[np.ndarray],
                                  feature_aliases: Optional[Mapping[int, str]] = None,
                                  subplots: bool = False,
                                  n_cols: int = 3,
                                  path: Optional[str] = None,
                                  **kwargs):
    limits_min = np.min([
        v.min()
        for f, v in zip(feature_names, shape_functions) if isinstance(f, tuple)
    ])
    limits_max = np.max([
        v.max()
        for f, v in zip(feature_names, shape_functions) if isinstance(f, tuple)
    ])

    pairwise_feature_names = list(filter(lambda fn: isinstance(fn, tuple), feature_names))
    n_pairwise_features = len(pairwise_feature_names)
    if subplots:
        plt.figure()
        fig, axes = plt.subplots(
            n_pairwise_features // n_cols + 1
            if n_pairwise_features % n_cols != 0
            else n_pairwise_features // n_cols,
            n_cols,
            **kwargs
        )

    for i, name in enumerate(pairwise_feature_names):
        if not subplots:
            plt.figure()
            fig, axes = plt.subplots(1, 1)
            ax = axes
        else:
            ax = axes.ravel()[i]
        if feature_aliases is None:
            cur_feature_names = (name[0], name[1])
        else:
            cur_feature_names = (feature_aliases[name[0]], feature_aliases[name[0]])
        title = f"{cur_feature_names}"
        ax.set_title(title)
        ax.set_xlabel(cur_feature_names[0])
        ax.set_ylabel(cur_feature_names[1])
        t = ax.scatter(xs[:, name[0]], xs[:, name[1]], c=shape_functions[i])
        lim_diff = limits_max - limits_min
        t.set_clim(limits_min - lim_diff * 0.05, limits_max + lim_diff * 0.05)
        if not subplots and path is not None:
            assert '%' in path
            save_figure_to_file(path % name)

    plt.tight_layout()

    if subplots:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(t, cax=cbar_ax)

    if subplots and path is not None:
        save_figure_to_file(path)


def plot_pairwise_grid_shape_functions(pairwise_grid_xs: Mapping[tuple, np.ndarray],
                                       pairwise_shape_functions: Mapping[tuple, tuple],
                                       feature_aliases: Optional[Mapping[int, str]] = None,
                                       subplots: bool = False,
                                       subtract_mean: bool = True,
                                       n_cols: int = 3,
                                       path: Optional[str] = None,
                                       only_pairs: Optional[List[tuple]] = None,
                                       **kwargs):
    n_pairwise_features = len(pairwise_shape_functions)
    if only_pairs is not None:
        n_pairwise_features = len(only_pairs)
    if subplots:
        plt.figure()
        fig, axes = plt.subplots(
            n_pairwise_features // n_cols + 1
            if n_pairwise_features % n_cols != 0
            else n_pairwise_features // n_cols,
            n_cols,
            **kwargs
        )

    plots = []
    limits_min, limits_max = np.inf, -np.inf

    for name, shape_function in pairwise_shape_functions.items():
        cur_xs = pairwise_grid_xs[name]
        if feature_aliases is None:
            cur_feature_names = (name[0], name[1])
        else:
            cur_feature_names = (feature_aliases[name[0]], feature_aliases[name[1]])
        if only_pairs is not None:
            if cur_feature_names not in only_pairs:
                continue
        if not subplots:
            plt.figure()
            fig, axes = plt.subplots(1, 1)
            ax = axes
        else:
            ax = axes.ravel()[len(plots)]

        title = f"{cur_feature_names}"
        ax.set_title(title)
        ax.set_xlabel(cur_feature_names[0])
        ax.set_ylabel(cur_feature_names[1])
        s = int(np.sqrt(cur_xs[:, name[0]].shape[0]))
        shape = (s, s)
        if subtract_mean:
            z = shape_function - shape_function.mean()
        else:
            z = shape_function
        t = ax.pcolormesh(cur_xs[:, name[0]].reshape(shape),
                          cur_xs[:, name[1]].reshape(shape),
                          z.reshape(shape), cmap='RdBu')

        limits_min = min(limits_min, z.min())
        limits_max = max(limits_max, z.max())
        plots.append(t)

        if not subplots and path is not None:
            assert '%' in path
            save_figure_to_file(path % name)

    plt.tight_layout()
    if subplots:
        lim_diff = limits_max - limits_min
        for t in plots:
            t.set_clim(limits_min - lim_diff * 0.05, limits_max + lim_diff * 0.05)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(t, cax=cbar_ax)

    if subplots and path is not None:
        save_figure_to_file(path)


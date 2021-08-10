import numpy as np
import torch
from torch import Tensor
from typing import Callable, Tuple, List
from collections import namedtuple
from tqdm import tqdm
from .explanation_model import AttentionExplanationModel
from .training_parameters import TrainingParameters
from .losses import weighted_mse_loss


TrainingResult = namedtuple('TrainingResult', [
    'losses',
    'last_feature_weights',
])


def train(model: AttentionExplanationModel,
          data_gen: Callable[[], Tuple[Tensor, Tensor, Tensor]],
          params: TrainingParameters,
          callbacks: List[Callable[[torch.nn.Module], None]] = None) -> TrainingResult:
    if callbacks is None:
        callbacks = []
    if params.use_weights:
        loss_fn = weighted_mse_loss
    else:
        loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    model.train()
    losses = []
    for epoch in tqdm(range(params.epochs)):
        batch_losses = []
        optimizer.zero_grad()
        for _j in range(params.batch_size):
            xs, y, weights = data_gen()
            xs = torch.tensor(xs)
            y = torch.tensor(y).view(-1, 1)
            if not params.use_surrogate_model:
                out, _feature_weights, _feature_vectors = model(xs, y)
            else:  # params.use_surrogate_model
                surrogate_out, (out, _feature_weights, _feature_vectors) = model.predict_all(xs)

            if params.use_weights:
                loss = loss_fn(out, y, torch.tensor(weights))
            else:
                loss = loss_fn(out, y)

            if isinstance(params.use_surrogate_model, float):
                loss += loss_fn(surrogate_out, y) * params.use_surrogate_model

            if params.fnn_l2 is not None:
                penalty = torch.mean(torch.cat(tuple((fv ** 2) for fv in _feature_vectors)))
                loss += params.fnn_l2 * penalty
            if params.feature_weights_l1 is not None:
                penalty = torch.mean(torch.cat(tuple(torch.abs(fw) for fw in _feature_weights)))
                loss += params.feature_weights_l1 * penalty
            loss.backward()
            losses.append(loss.item())
            batch_losses.append(loss.item())
        # calculate mean gradient over batch
        for p in model.parameters():
            p.grad /= params.batch_size
        # end
        optimizer.step()

        for cb in callbacks:
            cb(model)
        if params.verbose:
            print("Epoch: {}. Avg batch loss: {}".format(epoch, np.mean(batch_losses)))

    return TrainingResult(
        losses=losses,
        last_feature_weights=_feature_weights,
    )

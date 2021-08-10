import torch


class WeightedShortcut(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.weight = torch.nn.Parameter(torch.ones(1) * 0.9, requires_grad=True)
        self.coefficient = torch.nn.Parameter(torch.ones(1) * 0.001, requires_grad=True)

    def forward(self, x):
        assert x.ndim == 2
        assert x.shape[1] == 1
        linear_model = self.coefficient * x
        return linear_model * self.weight + self.model(x) * (1 - self.weight)

import torch
import torch.nn as nn


def params_to_tokens(params: torch.Tensor):
    """Assuming our model outputs a tensor of shape (batch, 2k), we stack it into a
    tensor of shape (batch, k, 2) to allow for metric computation.
    """
    freqs, amps = params.chunk(2, dim=-1)
    return torch.stack((freqs, amps), dim=-1)


def chamfer_loss(predicted: torch.Tensor, target: torch.Tensor):
    predicted_tokens = params_to_tokens(predicted)
    target_tokens = params_to_tokens(target)

    costs = torch.cdist(predicted_tokens, target_tokens)
    min1 = costs.min(dim=1)[0].mean(dim=-1)
    min2 = costs.min(dim=2)[0].mean(dim=-1)

    chamfer_distance = torch.mean(min1 + min2)
    return chamfer_distance


class ChamferLoss(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return chamfer_loss(predicted, target)

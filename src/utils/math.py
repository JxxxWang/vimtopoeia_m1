import torch


def divmod(a, b):
    return torch.div(a, b, rounding_mode="floor"), torch.remainder(a, b)

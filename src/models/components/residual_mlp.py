from typing import Literal, Optional

import torch
import torch.nn as nn

from src.models.components.cnn import ResidualEncoder


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        if out_dim is None and hidden_dim is not None:
            out_dim = hidden_dim
        elif out_dim is None:
            out_dim = in_dim

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.residual = (
            nn.Identity()
            if in_dim == out_dim
            else nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.residual(x) + self.net(x)


class ResidualMLP(nn.Sequential):
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 16,
        num_blocks: int = 6,
    ):
        layers = [
            ResidualMLPBlock(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == (num_blocks - 1) else hidden_dim,
            )
            for i in range(num_blocks)
        ]

        super().__init__(*layers)


class SpectralResidualMLP(ResidualMLP):
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 16,
        num_blocks: int = 6,
    ):
        true_in_dim = in_dim // 2 + 1
        super().__init__(true_in_dim, hidden_dim, out_dim, num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, norm="forward")
        X = torch.abs(X)
        return super().forward(X)


class CNNResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        channels: int = 16,
        encoder_blocks: int = 4,
        trunk_blocks: int = 5,
        hidden_dim: int = 2048,
        out_dim: int = 16,
        kernel_size: int = 7,
        norm: Literal["bn", "ln"] = "bn",
    ):
        super().__init__()

        self.encoder = ResidualEncoder(
            in_dim, channels, hidden_dim, encoder_blocks, kernel_size, norm
        )
        self.trunk = ResidualMLP(hidden_dim, hidden_dim, out_dim, trunk_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.trunk(z)

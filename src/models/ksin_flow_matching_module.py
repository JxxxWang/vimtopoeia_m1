import math
from functools import partial
from typing import Any, Callable, Dict, Literal, Tuple

import torch
from lightning import LightningModule

from src.metrics import ChamferDistance, LogSpectralDistance


def curve(x, a):
    if a == 0.0:
        return x
    return (1 - torch.exp(-a * x)) / (1 - math.exp(-a))


def call_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    y_c = f(x, t, conditioning)
    y_u = f(x, t, None)

    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    f = partial(call_with_cfg, f, conditioning=conditioning, cfg_strength=cfg_strength)
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class KSinFlowMatchingModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cfg_dropout_rate: float = 0.1,
        train_schedule: Literal["uniform", "bias_later", "bias_middle"] = "uniform",
        sample_schedule_curve: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        compile: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field

        self.val_lsd = LogSpectralDistance()
        self.val_chamfer = ChamferDistance()

        self.test_lsd = LogSpectralDistance()
        self.test_chamfer = ChamferDistance()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.vector_field(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_lsd.reset()
        self.val_chamfer.reset()

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        if self.hparams.train_schedule == "uniform":
            return torch.rand(n, 1, device=device)
        elif self.hparams.train_schedule == "bias_later":
            t = torch.rand(n, 1, device=device)
            return curve(t, 1.0)
        elif self.hparams.train_schedule == "bias_middle":
            return torch.randn(n, 1, device=device).sigmoid()

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch

        # Get conditioning vector
        conditioning = self.encoder(x)
        conditioning = self.vector_field.apply_dropout(
            conditioning, self.hparams.cfg_dropout_rate
        )

        # Sample time-steps
        t = self._sample_time(x.shape[0], x.device)
        p0 = torch.randn_like(y)

        # we sample a point along the trajectory
        pt = p0 * (1 - t) + y * t

        # our target velocity is the vector from the noise to the sample
        target = y - p0

        prediction = self.vector_field(pt, t, conditioning)
        loss = torch.nn.functional.mse_loss(prediction, target)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._train_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def _sample(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        steps: int,
        cfg_strength: float,
    ):
        x, y = batch

        sample = torch.randn_like(y)
        conditioning = self.encoder(x)
        t = torch.zeros(sample.shape[0], 1, device=sample.device)
        dt = 1.0 / steps

        for _ in range(steps):
            warped_t = curve(t, self.hparams.sample_schedule_curve)
            warped_t_plus_dt = curve(t + dt, self.hparams.sample_schedule_curve)
            warped_dt = warped_t_plus_dt - warped_t

            sample = rk4_with_cfg(
                self.vector_field,
                sample,
                warped_t,
                warped_dt,
                conditioning,
                cfg_strength,
            )
            t = t + dt

        return sample, y, x

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds, targets, inputs = self._sample(
            batch,
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
        )

        # update and log metrics
        self.val_lsd(preds, inputs)
        self.val_chamfer(preds, targets)

        self.log("val/lsd", self.val_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/chamfer", self.val_chamfer, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds, targets, inputs = self._sample(
            batch, self.hparams.test_sample_steps, self.hparams.test_cfg_strength
        )

        self.test_lsd(preds, inputs)
        self.test_chamfer(preds, targets)
        self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/chamfer",
            self.test_chamfer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        # TODO: implement metrics
        # self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        # etc...
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/chamfer",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = KSinFeedForwardModule(None, None, None, None, None)

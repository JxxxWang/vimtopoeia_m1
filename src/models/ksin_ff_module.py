from typing import Any, Callable, Dict, Tuple

import torch
from lightning import LightningModule

from src.metrics import ChamferDistance, LogSpectralDistance, SpectralDistance
from src.models.components.loss import ChamferLoss


class KSinFeedForwardModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss_fn: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        if loss_fn == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_fn == "chamfer":
            self.criterion = ChamferLoss()
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_fn}")

        self.train_lsd = LogSpectralDistance()
        self.train_sd = SpectralDistance()
        self.train_chamfer = ChamferDistance()

        self.val_lsd = LogSpectralDistance()
        self.val_sd = SpectralDistance()
        self.val_chamfer = ChamferDistance()

        self.test_lsd = LogSpectralDistance()
        self.test_sd = SpectralDistance()
        self.test_chamfer = ChamferDistance()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_lsd.reset()
        self.val_chamfer.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y, x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        self.train_lsd(preds, inputs)
        self.train_sd(preds, inputs)
        self.train_chamfer(preds, targets)
        self.log(
            "train/lsd", self.train_lsd, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/sd", self.train_sd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/chamfer",
            self.train_chamfer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        # update and log metrics
        self.val_lsd(preds, inputs)
        self.val_sd(preds, inputs)
        self.val_chamfer(preds, targets)

        self.log("val/lsd", self.val_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sd", self.val_sd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/chamfer", self.val_chamfer, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        self.test_lsd(preds, inputs)
        self.test_sd(preds, inputs)
        self.test_chamfer(preds, targets)
        self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/sd", self.test_sd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/chamfer",
            self.test_chamfer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = KSinFeedForwardModule(None, None, None, None, None)

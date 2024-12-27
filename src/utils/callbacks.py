import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback


class PlotLossPerTimestep(Callback):
    """Takes a batch from the validation dataloader, and runs it through the model at
    a number of different values for t. Plots the loss as a function of t.
    """

    def __init__(self, num_timesteps: int = 100):
        super().__init__()
        self.num_timesteps = num_timesteps

    def _get_val_batch(self, trainer):
        val_dl = trainer.val_dataloaders
        return next(iter(val_dl))

    def _compute_losses(self, trainer, pl_module):
        batch = self._get_val_batch(trainer)
        signal, params, _ = batch

        # Get conditioning vector
        conditioning = pl_module.encoder(signal)
        z = pl_module.vector_field.apply_dropout(
            conditioning, pl_module.hparams.cfg_dropout_rate
        )

        x0, x1, z = pl_module._sample_x0_and_x1(params, z)

        losses = []
        for n in range(self.num_timesteps):
            t = torch.full(
                (signal.shape[0], 1), n / (self.num_timesteps - 1), device=signal.device
            )
            x_t = pl_module._sample_probability_path(x0, x1, t)
            target = pl_module._evaluate_target_field(x0, x1, x_t, t)

            prediction = pl_module.vector_field(x_t, t, z)
            loss = (prediction - target).square().mean(dim=-1)
            losses.append(loss)

        return torch.stack(losses, dim=-1)

    def _aggregate_losses(self, losses):
        mean = losses.mean(dim=0)
        std = losses.std(dim=0)
        lower_ci = mean - 2 * std
        upper_ci = mean + 2 * std
        return mean, lower_ci, upper_ci

    def _plot_losses(self, losses):
        t = np.linspace(0, 1, self.num_timesteps)
        mean, lower_ci, upper_ci = self._aggregate_losses(losses)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, mean.cpu().numpy())
        ax.fill_between(t, lower_ci.cpu().numpy(), upper_ci.cpu().numpy(), alpha=0.2)
        ax.set_xlabel("t")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per noise level / timestep")
        return fig

    def _log_plot(self, fig, trainer):
        plot = wandb.Image(fig)
        wandb.log({"plot": plot}, step=trainer.global_step)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        losses = self._compute_losses(trainer, pl_module)
        fig = self._plot_losses(losses)
        self._log_plot(fig, trainer)

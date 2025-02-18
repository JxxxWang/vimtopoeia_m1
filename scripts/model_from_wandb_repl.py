from pathlib import Path
from typing import Literal

import click
import hydra
import torch
from IPython import embed
from omegaconf import DictConfig, OmegaConf


def wandb_dir_to_ckpt_and_hparams(
    wandb_dir: Path, ckpt_type: Literal["best", "last"]
) -> Tuple[Path, Path]:
    log_dir = wandb_dir.parent
    hparam_file = log_dir / "csv" / "hparams.yaml"
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_type == "last":
        ckpt_file = ckpt_dir / "last.ckpt"
    elif ckpt_type == "best":
        ckpt_files = ckpt_dir.glob("epoch*.ckpt")

        # most recent file
        ckpt_files = sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True)
        ckpt_file = ckpt_files[0]

    return ckpt_file, hparam_file


def get_state_dict(ckpt_file: Path, map_location: str = "cuda") -> dict:
    ckpt = torch.load(ckpt_file, map_location=map_location, weight_only=False)
    state_dict = ckpt["state_dict"]
    return state_dict


def instantiate_model(
    model_cfg: DictConfig, ckpt_file: Path, map_location: str = "cuda"
) -> torch.nn.Module:
    state_dict = get_state_dict(ckpt_file, map_location=map_location)

    model = hydra.utils.instantiate(model_cfg)
    model.to(device=map_location)

    model.load_state_dict(state_dict)

    return model


def instantiate_datamodule(data_cfg: DictConfig):
    return hydra.utils.instantiate(data_cfg)


@click.command()
@click.argument("wandb_id", type=str)
@click.option("--log-dir", "-l", type=str)
@click.option("--ckpt_type", "-c", type=str)
@click.option("--device", "-d", type=str)
def main(
    wandb_id: str,
    log_dir: str = "logs/",
    ckpt_type: Literal["best", "last"] = "last",
    device: str = "cuda",
):
    log_dir = Path(log_dir)
    possible_wandb_dirs = log_dir.glob(f"**/*{wandb_id}/")

    ckpts_and_hparams = map(
        lambda x: wandb_dir_to_ckpt_and_hparams(x, ckpt_type), possible_wandb_dirs
    )

    if len(ckpts_and_hparams) > 1:
        # take the one with the most recently updated hparam file
        ckpt_file, hparam_file = max(
            ckpts_and_hparams, key=lambda x: x[1].stat().st_mtime
        )
    elif len(ckpts_and_hparams) == 1:
        ckpt_file, hparam_file = next(ckpts_and_hparams)
    else:
        raise RuntimeError("Could not find wandb id in any of the log directories.")

    cfg = OmegaConf.load(hparam_file)

    model = instantiate_model(cfg.model, ckpt_file, device)
    datamodule = instantiate_datamodule(cfg.data)

    embed(user_ns={"model": model, "datamodule": datamodule, "cfg": cfg})


if __name__ == "__main__":
    main()

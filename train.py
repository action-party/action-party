import argparse
import os

import wandb
from omegaconf import OmegaConf

from trainer import DiffusionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name

    if args.logdir:
        config.logdir = args.logdir
    else:
        config.logdir = f"checkpoints/{config_name}"

    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb or config.get("disable_wandb", False)

    # The open-source branch only ships the diffusion trainer (used by every
    # exp-131/134/135/138/139/141-147 config). The upstream
    # gan / ode / distillation / self_forcing_diffusion trainers were removed
    # during the refactor; if a config asks for them we fail loudly.
    trainer_name = getattr(config, "trainer", "diffusion")
    if trainer_name != "diffusion":
        raise ValueError(
            f"trainer={trainer_name!r} is not available in this build. "
            "Only the 'diffusion' trainer is shipped in the opensource refactor."
        )

    trainer = DiffusionTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

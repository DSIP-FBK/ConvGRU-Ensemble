# train.py
import os
import sys
sys.path.append('../')

import fiddle as fdl
import yaml
import torch
from absl import app, flags
from fiddle import absl_flags, printing
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from losses import PIXEL_LOSSES
from datetime import datetime

from datamodule import RadarDataModule
from lightning_model import RadarLightningModel

seed_everything(42, workers=True)

FLAGS = flags.FLAGS
flags.DEFINE_bool('print_config', False, 'Print configuration and exit.')
flags.DEFINE_string('export_yaml', None, 'Export configuration to YAML file and exit.')


def experiment() -> fdl.Config:
    """Experiment configuration. Override any value with --config config:experiment --config set:path.to.value=X"""
    cfg = fdl.Config(dict)

    # resume from checkpoint
    cfg.checkpoint_path = None

    # enable mixed precision for float32 matmuls if available
    cfg.float32_matmul_precision = None

    # compile model with torch.compile if desired
    cfg.compile_model = False

    # DataModule
    cfg.datamodule = fdl.Config(
        RadarDataModule,
        zarr_path='./data/italian-radar-dpc-sri.zarr',
        csv_path='./importance_sampler/output/sampled_datacubes_2021-01-01-2025-12-11_24x256x256_3x16x16_10000.csv',
        steps=18,
        train_ratio=0.90,
        val_ratio=0.05,
        return_mask=True,
        deterministic=False,
        augment=True,
        # DataLoader params
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        multiprocessing_context='fork',
    )

    # Lightning Model
    cfg.model = fdl.Config(
        RadarLightningModel,
        input_channels=1,
        forecast_steps=12,
        num_blocks=5,
        ensemble_size=2,
        noisy_decoder=False,
        loss_class='crps',
        loss_params={'temporal_lambda': 0.01},
        masked_loss=True,
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 1e-4, 'fused': True},
        lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 10},
    )

    # Trainer
    cfg.trainer = fdl.Config(
        Trainer,
        accelerator='auto',
        # gradient_clip_val=1.0,
        max_epochs=1,
    )

    # Callbacks
    cfg.callbacks = fdl.Config(dict)
    cfg.callbacks.checkpoint_val = fdl.Config(
        ModelCheckpoint,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=None,
        filename=None,  # Set dynamically: best-val-{ckpt_name}
        save_last=False,
    )
    cfg.callbacks.checkpoint_train = fdl.Config(
        ModelCheckpoint,
        monitor='train_loss_epoch',
        save_top_k=1,
        mode='min',
        dirpath=None,
        filename=None,  # Set dynamically: best-train-{ckpt_name}
        save_last=False,
    )
    cfg.callbacks.early_stopping = fdl.Config(
        EarlyStopping,
        monitor='val_loss',
        patience=100,
        mode='min',
    )
    cfg.callbacks.lr_monitor = fdl.Config(
        LearningRateMonitor,
        logging_interval='step',
        log_momentum=False,
        log_weight_decay=False,
    )

    # Loggers
    cfg.loggers = fdl.Config(dict)
    cfg.loggers.tensorboard = fdl.Config(
        TensorBoardLogger,
        save_dir='logs',
        name=None,  # Set dynamically in train()
        version=None,  # Set dynamically in train()
    )

    return cfg


_CONFIG = absl_flags.DEFINE_fiddle_config(
    'config',
    default_module=sys.modules[__name__],
    help_string='Experiment configuration.',
)


def train(cfg: fdl.Config) -> None:
    """Run training with the given fiddle configuration."""
    # enable tensor cores for float32 matmuls if available
    if cfg.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Compute dynamic values for naming
    future_steps = cfg.model.forecast_steps
    past_steps = cfg.datamodule.steps - future_steps

    if cfg.model.loss_class is None:
        loss_name = 'MSELoss'
    elif isinstance(cfg.model.loss_class, type):
        loss_name = cfg.model.loss_class.__name__
    else:
        loss_name = PIXEL_LOSSES[cfg.model.loss_class.lower()].__name__ if cfg.model.loss_class.lower() in PIXEL_LOSSES else str(cfg.model.loss_class)
    lr = cfg.model.optimizer_params["lr"] if cfg.model.optimizer_params is not None and "lr" in cfg.model.optimizer_params else "default"

    noise_str: str = '_noise' if cfg.model.noisy_decoder else ''
    ckpt_base_name: str = f'{past_steps}past-{future_steps}fut{noise_str}_bs{cfg.datamodule.batch_size}_lr{lr}'

    # Set dynamic logger name and version first (checkpoint folder depends on it)
    if cfg.loggers.tensorboard.name is None:
        cfg.loggers.tensorboard.name = f'{loss_name}_{past_steps}past-{future_steps}fut{noise_str}'

    jobid = os.getenv('SLURM_JOB_ID', None)
    tb_version = f'_{cfg.loggers.tensorboard.version}' if cfg.loggers.tensorboard.version is not None else ''

    if jobid is not None:
        cfg.loggers.tensorboard.version = f'job{jobid}_{ckpt_base_name}{tb_version}'
    else:   
        cfg.loggers.tensorboard.version = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{ckpt_base_name}{tb_version}'

    # Set checkpoint paths inside tensorboard experiment folder
    tb_log_dir = f'{cfg.loggers.tensorboard.save_dir}/{cfg.loggers.tensorboard.name}/{cfg.loggers.tensorboard.version}'
    ckpt_dir = f'{tb_log_dir}/checkpoints'

    # Val checkpoint
    if cfg.callbacks.checkpoint_val.dirpath is None:
        cfg.callbacks.checkpoint_val.dirpath = ckpt_dir
    if cfg.callbacks.checkpoint_val.filename is None:
        cfg.callbacks.checkpoint_val.filename = 'best-val-' + ckpt_base_name + '_ep{epoch:03d}_loss{val_loss:.4f}'

    # Train checkpoint
    if cfg.callbacks.checkpoint_train.dirpath is None:
        cfg.callbacks.checkpoint_train.dirpath = ckpt_dir
    if cfg.callbacks.checkpoint_train.filename is None:
        cfg.callbacks.checkpoint_train.filename = 'best-train-' + ckpt_base_name + '_ep{epoch:03d}_loss{train_loss_epoch:.4f}'

    # Build all callbacks and loggers dynamically
    callbacks_dict = fdl.build(cfg.callbacks)
    loggers_dict = fdl.build(cfg.loggers)
    callbacks = list(callbacks_dict.values())
    loggers = list(loggers_dict.values())

    # Add loggers and callbacks to trainer config
    cfg.trainer.logger = loggers
    cfg.trainer.callbacks = callbacks

    print(printing.as_str_flattened(cfg))

    # Save config to tensorboard folder
    os.makedirs(tb_log_dir, exist_ok=True)
    config_path = f'{tb_log_dir}/config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_to_dict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f'Config saved to {config_path}')

    # Build all components
    built = fdl.build(cfg)
    datamodule: RadarDataModule = built['datamodule']

    if cfg.checkpoint_path is not None:
        print(f"Resuming training from checkpoint: {cfg.checkpoint_path}")
        model = RadarLightningModel.load_from_checkpoint(cfg.checkpoint_path, strict=True, weights_only=False)
    else:
        model = built['model']
    trainer: Trainer = built['trainer']

    datamodule.setup()
    print(f"Train: {len(datamodule.train_dataset)}, Val: {len(datamodule.val_dataset)}, Test: {len(datamodule.test_dataset)}")

    if cfg.compile_model:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, dynamic=True)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    print(f"Best val: {callbacks_dict['checkpoint_val'].best_model_path}")
    print(f"Best train: {callbacks_dict['checkpoint_train'].best_model_path}")


def config_to_dict(cfg: fdl.Config) -> dict:
    """Convert fiddle config to nested dict for YAML export."""
    result = {}
    for key, value in fdl.ordered_arguments(cfg).items():
        result[key] = config_to_dict(value) if isinstance(value, fdl.Config) else value
    return result


def main(argv: list[str]) -> None:
    del argv
    cfg = _CONFIG.value
    if FLAGS.print_config:
        print(printing.as_str_flattened(cfg))
        return
    if FLAGS.export_yaml:
        cfg_dict = config_to_dict(cfg)
        with open(FLAGS.export_yaml, 'w') as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
        print(f'Config exported to {FLAGS.export_yaml}')
        return
    train(cfg)

# Example command to run training with custom configuration overrides.
# uv run python train.py \
#     --config config:experiment \
#     --config set:callbacks.checkpoint.save_top_k=3 \
#     --config set:model.num_blocks=5 \
#     --config set:model.forecast_steps=12 \
#     --config set:datamodule.steps=18 \
#     --config set:datamodule.num_workers=32 \
#     --config set:datamodule.batch_size=32
if __name__ == '__main__':
    app.run(main)

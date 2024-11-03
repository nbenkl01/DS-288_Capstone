import os
import torch
from transformers import (
    PatchTSMixerConfig,
    TrainingArguments,
)

def configure_model(config):
    """
    Configures the PatchTSMixer model with specified parameters.
    Args:
        input_columns (list): List of input column names.
        context_length (int): Length of context window.
        patch_length (int): Length of each patch.
        prediction_length (int, optional): Length of prediction window (used in pretraining).
        classifier (bool): If True, configures as classifier; else for masked prediction.
    Returns:
        PatchTSMixerConfig: Configuration for the PatchTSMixer model.
    """
    config_params = {
        "context_length": config.context_length,
        "patch_length": config.patch_length,
        "num_input_channels": len(config.input_columns),
        "patch_stride": config.patch_length,
        "d_model": 2 * config.patch_length,
        "num_layers": 8,
        "expansion_factor": 2,
        "dropout": 0.2,
        "head_dropout": 0.2,
        "mode": "mix_channel",
        "scaling": "std",
    }
    if config.task == 'classifier':
        config_params.update({"num_targets": 2})
    else:
        config_params.update({"prediction_length": config.prediction_length})

    return PatchTSMixerConfig(**config_params)

def configure_masked_transformer(config):
    """
    Configures the PatchTSMixer model with specified parameters.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig
    """
    return PatchTSMixerConfig(
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        patch_length=config.patch_length,
        num_input_channels=len(config.input_columns),
        patch_stride=config.patch_length,
        d_model=2 * config.patch_length,
        num_layers=8,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        mode="mix_channel",
        scaling="std",
    )

def configure_classifier(config):
    """
    Configures the PatchTSMixer model with specified parameters.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig
    """
    return PatchTSMixerConfig(
        context_length=config.context_length,
        patch_length=config.patch_length,
        num_input_channels=len(config.input_columns),
        patch_stride=config.patch_length,
        d_model=2 * config.patch_length,
        num_layers=8,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        num_targets = 2,
        # output_range = [0,1],
        # use_return_dict = True,
        mode="mix_channel",
        scaling="std",
    )

def pretrain_training_args(config):
    """
    Sets up training arguments for the trainer.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    """
    return TrainingArguments(
        output_dir=os.path.join(config.checkpoint_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=0.001,
        num_train_epochs=config.train_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        report_to="wandb",
        run_name = config.run_name,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir=os.path.join(config.checkpoint_dir, "logs"),
        load_best_model_at_end=True
    )

def classify_training_args(config):
    """
    Sets up training arguments for the trainer.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    """
    return TrainingArguments(
        output_dir=os.path.join(config.checkpoint_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=0.001,
        num_train_epochs=config.train_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        report_to="wandb",
        run_name = config.run_name,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir=os.path.join(config.checkpoint_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
    )
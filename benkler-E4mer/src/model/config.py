import os
import torch
from transformers import (
    PatchTSMixerConfig,
    TrainingArguments,
)

def configure_masked_transformer(input_columns, context_length, prediction_length, patch_length):
    """
    Configures the PatchTSMixer model with specified parameters.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig
    """
    return PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        patch_length=patch_length,
        num_input_channels=len(input_columns),
        patch_stride=patch_length,
        d_model=2 * patch_length,
        num_layers=8,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        mode="mix_channel",
        scaling="std",
    )

def configure_classifier(input_columns, context_length, #prediction_length,
                          patch_length):
    """
    Configures the PatchTSMixer model with specified parameters.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig
    """
    return PatchTSMixerConfig(
        context_length=context_length,
        # prediction_length=prediction_length,
        patch_length=patch_length,
        num_input_channels=len(input_columns),
        patch_stride=patch_length,
        d_model=2 * patch_length,
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

def pretrain_training_args(checkpoint_dir, train_epochs, batch_size, num_workers, input_columns, run_name = None):
    """
    Sets up training arguments for the trainer.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    """
    return TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "output"),
        overwrite_output_dir=True,
        # use_mps_device = True if torch.backends.mps.is_available() else False, # Use Mac MPS
        learning_rate=0.001,
        num_train_epochs=train_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="wandb",
        run_name = run_name,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir=os.path.join(checkpoint_dir, "logs"),
        load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # label_names=input_columns,
    )

def classify_training_args(checkpoint_dir, train_epochs, batch_size, num_workers,
                           input_columns, target_columns, run_name = None):
    """
    Sets up training arguments for the trainer.
    See: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    """
    return TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "output"),
        overwrite_output_dir=True,
        # use_mps_device = True if torch.backends.mps.is_available() else False, # Use Mac MPS
        learning_rate=0.001,
        num_train_epochs=train_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="wandb",
        run_name = run_name,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir=os.path.join(checkpoint_dir, "logs"),
        load_best_model_at_end=True,
        # metric_for_best_model="eval_f1",
        # greater_is_better=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # label_names=target_columns,
    )
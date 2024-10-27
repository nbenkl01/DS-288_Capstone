import os
import torch
from transformers import (
    PatchTSMixerForPretraining,
    PatchTSMixerForTimeSeriesClassification,
    Trainer,
)
import requests
from src.model.data import fetch_data#fetch_next_batch, preprocess_batch
from src.data import preprocess
from src.model.config import configure_model, pretrain_training_args, classify_training_args
from src.model.utils import setup_early_stopping, evaluate_and_save_model, compute_metrics
from src.STATIC import ROOT_DIR


def train_model(model, train_dataset, val_dataset, training_args, early_stopping_callback, compute_metrics = None):
    """Initializes and trains the model using the Trainer API."""
    
#     def compute_metrics(eval_pred):
#         # Unpack predictions and labels (if available)
#         logits, labels = eval_pred
#         # Compute the loss or any other metric here
#         loss = trainer.compute_loss(model, torch.tensor(logits), torch.tensor(labels))  # Example
#         return {"eval_loss": loss}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = compute_metrics,
        callbacks=[early_stopping_callback],
    )
    trainer.train()
    return trainer


# Main function to initiate pretraining
def pretrain(dataset_code,
             input_columns, 
             id_columns = [],
             finetune = False,
            pretrained_model_dir = None,
            checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/unlabelled_pretrain"),
             save_dir=os.path.join(ROOT_DIR, "models/unlabelled_pretrain"), 
             run_name = 'unlabelled_pretrain',
             train_epochs=100, 
             context_length=512, prediction_length=96, patch_length=8,
             num_workers=16, batch_size=16):
    
    config = configure_model(input_columns, context_length, prediction_length, patch_length)
    training_args = pretrain_training_args(checkpoint_dir, train_epochs, batch_size, num_workers, input_columns, run_name = run_name)
    early_stopping_callback = setup_early_stopping()
    
    if finetune and os.path.exists(pretrained_model_dir):
        model = PatchTSMixerForPretraining(config).from_pretrained(pretrained_model_dir)
    else:
        model = PatchTSMixerForPretraining(config)

    data_loc = 'local' if os.path.exists(os.path.join(ROOT_DIR, f'./e4data/train_test_split/{dataset_code}')) else 'remote'
    train_data, val_data, test_data = fetch_data(dataset_code, location = data_loc)
    _, train_dataset, val_dataset, test_dataset = preprocess.preprocess_pretraining_datasets(
                                                                                train_data,
                                                                                val_data,
                                                                                test_data,                                              
                                                                                # target_columns = target_columns,
                                                                                id_columns = id_columns)

    # Train model in batches to handle memory and storage constraints
    trainer = train_model(model, train_dataset, val_dataset, training_args, early_stopping_callback)

    # Evaluate and save model
    evaluate_and_save_model(trainer, test_dataset, save_dir)


def train_classifier(dataset_code,
                     input_columns, target_columns, id_columns = [],
                     finetune = True,
                        pretrained_model_dir = os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
                         checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/stress_event_finetune"),
                         save_dir=os.path.join(ROOT_DIR, "models/stress_event_finetune"), 
                         run_name = 'classifier_finetune',
                         train_epochs=100, 
                         context_length=512, prediction_length=96,
                         patch_length=8, num_workers=16, batch_size=16):
    
    config = configure_model(input_columns, context_length, prediction_length, patch_length)
    training_args = classify_training_args(checkpoint_dir, train_epochs, batch_size, num_workers,
                                            input_columns,target_columns,
                                            run_name = run_name
                                          )
    early_stopping_callback = setup_early_stopping()

    if finetune and os.path.exists(pretrained_model_dir):
        model = PatchTSMixerForTimeSeriesClassification(config).from_pretrained(pretrained_model_dir)
    else:
        model = PatchTSMixerForTimeSeriesClassification(config)


    data_loc = 'local' if os.path.exists(os.path.join(ROOT_DIR, f'./e4data/train_test_split/{dataset_code}')) else 'remote'
    train_data, val_data, test_data = fetch_data(dataset_code, location = data_loc)
    _, train_dataset, val_dataset, test_dataset = preprocess.preprocess_finetuning_datasets(
                                                                                train_data,
                                                                                val_data,
                                                                                test_data,                                              
                                                                                target_columns = target_columns,
                                                                                id_columns = id_columns)

    # Train model in batches to handle memory and storage constraints
    trainer = train_model(model, train_dataset, val_dataset, training_args, early_stopping_callback,
                          compute_metrics = compute_metrics)

    # Evaluate and save model
    evaluate_and_save_model(trainer, test_dataset, save_dir)




# def main():
#     # Initiate pretraining process
#     pretrain(input_columns=['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'], train_epochs=1)
#     # train_classifier(WESAD_train_dataset, WESAD_val_dataset, WESAD_test_dataset,
#     #                  input_columns = ['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
#     #                 target_columns = ['binary_stress'],
#     #                  train_epochs = 1)
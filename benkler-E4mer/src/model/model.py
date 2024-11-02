import os
import torch
import numpy as np
from transformers import (
    PatchTSMixerForPretraining,
    PatchTSMixerForTimeSeriesClassification,
    Trainer,
)
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.model.data import fetch_data, clean_data, fetch_next_batch, preprocess_classifier_batch, preprocess_pretrain_batch
from src.data import preprocess
from src.model.config import configure_masked_transformer, configure_classifier, pretrain_training_args, classify_training_args
from src.model.utils import setup_early_stopping, evaluate_and_save_model
from src.model.custom_models import CustomPatchTSMixerForTimeSeriesClassification

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

def masked_trainer(model, train_dataset, val_dataset, training_args, early_stopping_callback):
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
        # compute_metrics = compute_metrics,
        callbacks=[early_stopping_callback],
    )
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    trainer.train()
    return trainer

def classifier_trainer(model, train_dataset, val_dataset, training_args, early_stopping_callback):
    """Initializes and trains the model using the Trainer API."""

    def compute_metrics(eval_pred):
        print("Computing Metrics")
        output, _ = eval_pred

        # Check if logits need reshaping
        if isinstance(output, tuple):
            logits = output[0]  # Unnest if needed
            labels = output[1]
        else:
            logits, labels = output
        print(f"logits: {logits}, labels: {labels}")
        
        if len(logits.shape) > 2:
            logits = logits.squeeze()  # Flatten if necessary

        predictions = np.argmax(logits, axis=-1)
        labels = labels
        print(f"predictions: {predictions}, labels: {labels}")
        
        # Calculate accuracy, precision, recall, F1
        acc = accuracy_score(labels, predictions)
        print(f"acc: {labels}")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        print(f"f1: {f1}")
        
        # Compute loss directly using model’s criterion (cross-entropy loss here as an example)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(torch.tensor(logits), torch.tensor(labels)).item()
        print(f"loss: {loss}")
        
        metrics = {
                    "eval_accuracy": acc,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f1": f1,
                    "eval_loss": loss,
                }
        print(f"Metrics returned by compute_metrics: {metrics}")
        return metrics
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = compute_metrics,
        callbacks=[early_stopping_callback]
    )
    trainer.train()
    return trainer


def batch_train_classifier(model, dataset_code, training_args, early_stopping_callback, input_columns, target_columns, id_columns, context_length, batch_size = 1000):
    """
    Train the model batch-by-batch to avoid loading all data into memory or disk space at once.
    """
    def compute_metrics(eval_pred):
        print("Computing Metrics")
        output, _ = eval_pred

        # Check if logits need reshaping
        if isinstance(output, tuple):
            logits = output[0]  # Unnest if needed
            labels = output[1]
        else:
            logits, labels = output
        print(f"logits: {logits}, labels: {labels}")
        
        if len(logits.shape) > 2:
            logits = logits.squeeze()  # Flatten if necessary

        predictions = np.argmax(logits, axis=-1)
        labels = labels
        print(f"predictions: {predictions}, labels: {labels}")
        
        # Calculate accuracy, precision, recall, F1
        acc = accuracy_score(labels, predictions)
        print(f"acc: {labels}")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        print(f"f1: {f1}")
        
        # Compute loss directly using model’s criterion (cross-entropy loss here as an example)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(torch.tensor(logits), torch.tensor(labels)).item()
        print(f"loss: {loss}")
        
        metrics = {
                    "eval_accuracy": acc,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f1": f1,
                    "eval_loss": loss,
                }
        print(f"Metrics returned by compute_metrics: {metrics}")
        return metrics
    
    trainer = None
    
    # Batch training loop
    batch_index = 0
    while True:
        try:
            train_data, val_data = fetch_next_batch(dataset_code, batch_index,
                                                     columns = ['datetime','subject_id','acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean','binary_stress'],
                                                      batch_size=500)
            train_data = clean_data(train_data, input_columns=input_columns, label_column=target_columns)
            val_data = clean_data(val_data, input_columns=input_columns, label_column=target_columns)
            # tsp, train_dataset, val_dataset = preprocess_classifier_batch(train_data, val_data, input_columns, id_columns, context_length, tsp=None if batch_index == 0 else tsp)
            _, train_dataset, val_dataset = preprocess_classifier_batch(train_data, val_data, input_columns, id_columns, context_length)
            if trainer:
                trainer.train_dataset = train_dataset
                trainer.eval_dataset = val_dataset
                trainer.train()
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset = train_dataset,
                    eval_dataset = val_dataset,
                    compute_metrics=compute_metrics,
                    callbacks=[early_stopping_callback],
                )
            batch_index += 1
        except requests.HTTPError:
            print("No more batches to fetch.")
            break  # Exit loop if no more data is available

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
    
    config = configure_masked_transformer(input_columns, context_length, prediction_length, patch_length)
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
    # trainer = train_model(model, train_dataset, val_dataset, training_args, early_stopping_callback)
    trainer = masked_trainer(model, train_dataset, val_dataset, training_args, early_stopping_callback)

    # Evaluate and save model
    evaluate_and_save_model(trainer, test_dataset, save_dir)


def train_classifier(dataset_code,
                     input_columns, target_columns, id_columns = [],
                     finetune = True,
                        pretrained_model_dir = os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
                         checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/stress_event_finetune"),
                         save_dir=os.path.join(ROOT_DIR, "models/stress_event_finetune"), 
                        run_name = 'classifier_finetune',
                        batch_train = True,
                         train_epochs=100, 
                         context_length=512, #prediction_length=96,
                         patch_length=8, num_workers=16, batch_size=16,
                         data_batch_size=500):
    # Data Preprocessing
    data_loc = 'local' if os.path.exists(os.path.join(ROOT_DIR, f'./e4data/train_test_split/{dataset_code}')) else 'remote'
    if data_loc == 'local':
        train_data, val_data, test_data = fetch_data(dataset_code, location = data_loc)
        train_data = clean_data(train_data, input_columns=input_columns, label_column=target_columns)
        val_data = clean_data(val_data, input_columns=input_columns, label_column=target_columns)
        test_data = clean_data(test_data, input_columns=input_columns, label_column=target_columns)
        _, train_dataset, val_dataset, test_dataset = preprocess.preprocess_finetuning_datasets(
                                                                                    train_data,
                                                                                    val_data,
                                                                                    test_data,
                                                                                    target_columns = target_columns,
                                                                                    id_columns = id_columns
                                                                                    )
        
    # Model Training Config
    config = configure_classifier(input_columns, context_length, #prediction_length,
                                   patch_length)
    training_args = classify_training_args(checkpoint_dir, train_epochs, batch_size, num_workers,
                                            input_columns,
                                            target_columns,
                                            run_name = run_name
                                          )
    early_stopping_callback = setup_early_stopping()

    if finetune and os.path.exists(pretrained_model_dir):
        model = CustomPatchTSMixerForTimeSeriesClassification(config).from_pretrained(pretrained_model_dir)
    else:
        model = CustomPatchTSMixerForTimeSeriesClassification(config)

    if data_loc == 'remote' and batch_train:
        # Train model in batches to handle memory and storage constraints
        trainer = batch_train_classifier(model, dataset_code, training_args, early_stopping_callback, input_columns, target_columns, id_columns, context_length, batch_size=data_batch_size)

    else:
        trainer = classifier_trainer(model, train_dataset, val_dataset, training_args, early_stopping_callback)

    # Evaluate and save model
    evaluate_and_save_model(trainer, test_dataset, save_dir)
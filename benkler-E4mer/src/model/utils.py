import os
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def setup_early_stopping():
    """Creates an early stopping callback to stop training when improvement stalls."""
    return EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.0001,
    )

def evaluate_and_save_model(trainer, test_dataset, config):
    """Evaluates the model on the test dataset and saves it to the specified directory."""
    results = trainer.evaluate(test_dataset)
    print("Test result:")
    print(results)
    os.makedirs(config.save_dir, exist_ok=True)
    trainer.save_model(config.save_dir)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, axis=-1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Calculate accuracy, precision, recall, F1
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # Compute loss directly using model’s criterion (cross-entropy loss here as an example)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(torch.tensor(logits), torch.tensor(labels)).item()
    
    return {
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_loss": loss,
    }
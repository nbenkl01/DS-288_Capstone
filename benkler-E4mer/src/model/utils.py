import os
from transformers import EarlyStoppingCallback

def setup_early_stopping():
    """Creates an early stopping callback to stop training when improvement stalls."""
    return EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.0001,
    )

def evaluate_and_save_model(trainer, test_dataset, save_dir):
    """Evaluates the model on the test dataset and saves it to the specified directory."""
    results = trainer.evaluate(test_dataset)
    print("Test result:")
    print(results)
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
import os
from transformers import (
    PatchTSMixerForPretraining,
    Trainer,
)
from src.model.data import get_data, clean_data, preprocess
from src.model.config import configure_model, configure_training_args
from src.model.utils import setup_early_stopping, evaluate_and_save_model, compute_metrics
from src.model.custom_models import CustomPatchTSMixerForTimeSeriesClassification

def initialize_trainer(model, training_args, train_dataset, val_dataset, early_stopping_callback, config):
    """Initializes the Trainer with common configurations."""
    trainer_params = {
            'model':model,
            'args':training_args,
            'train_dataset':train_dataset,
            'eval_dataset':val_dataset,
            'callbacks':[early_stopping_callback]
        }
    
    if config.task == 'classification':
        trainer_params.update({'compute_metrics':compute_metrics})
    
    trainer = Trainer(**trainer_params)
    
    return trainer

def train_model(model, training_args, early_stopping_callback, config):
    """
    Handles both batch and non-batch training, depending on memory constraints.
    """
    batch_index = 0
    trainer = None
    tsp=None
    
    print(f'BatchTrain: {config.batch_train}')
    while config.batch_train:
        if config.batch_val:
            train_data, val_data = get_data(config, subset = ['train', 'val'], batch_index = batch_index)
            if type(train_data) is type(None) and type(val_data) is type(None):
                print("No more batches to fetch.")
                break
            
            train_data, val_data = map(lambda data: clean_data(data, config), [train_data, val_data])
        else:
            if batch_index == 0:
                config.set_attribute(batch_train = False)
                val_data = get_data(config, subset = ['val'], batch_index = batch_index)
                val_data = clean_data(val_data, config)
                config.set_attribute(batch_train = True)
            
            train_data = get_data(config, subset = ['train'], batch_index = batch_index)
            if type(train_data) is type(None):
                print("No more batches to fetch.")
                break
            
            train_data = clean_data(train_data, config)

        tsp, train_dataset = preprocess(train_data, config, tsp=tsp, fit = True)
        _, val_dataset = preprocess(val_data, config, tsp=tsp, fit = False)
        
        if trainer is None:
            trainer = initialize_trainer(model, training_args, train_dataset, val_dataset, early_stopping_callback, config)
        else:
            trainer.train_dataset, trainer.eval_dataset = train_dataset, val_dataset
        
        trainer.train()
        batch_index += 1
    
    if not config.batch_train:
        # If batch training is disabled, fetch and preprocess all data at once
        # train_data, val_data = get_data(config, subset = ['train', 'val'])
        train_data = get_data(config, subset = ['train'])
        val_data = get_data(config, subset = ['val'])
        train_data, val_data = map(lambda data: clean_data(data, config), [train_data, val_data])
        tsp, train_dataset = preprocess(train_data, config, tsp=None, fit = True)
        _, val_dataset = preprocess(val_data, config, tsp=tsp, fit = False)
        trainer = initialize_trainer(model, training_args, train_dataset, val_dataset, early_stopping_callback, config)
        trainer.train()

    return trainer, tsp

def run_training_task(config):
    """
    High-level function to handle different training tasks, including masked pretraining and classifier fine-tuning.
    """
    model_config = configure_model(config)
    training_args = configure_training_args(config)
    early_stopping_callback = setup_early_stopping()

    model_class = CustomPatchTSMixerForTimeSeriesClassification if config.task == 'classification' else PatchTSMixerForPretraining
    if config.finetune and os.path.exists(config.pretrained_model_dir):
        model = model_class(model_config).from_pretrained(config.pretrained_model_dir)
        if config.freeze:
            for param in model.model.parameters():
                param.requires_grad = False
    else:
        model = model_class(model_config)

    trainer, tsp = train_model(model, training_args, early_stopping_callback, config)

    test_data = get_data(config, subset = ['test'])
    test_data = clean_data(test_data, config)
    _, test_dataset = preprocess(test_data, config, tsp=tsp, fit = False)

    evaluate_and_save_model(trainer, test_dataset, config)
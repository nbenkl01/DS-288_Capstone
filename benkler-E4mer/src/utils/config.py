import argparse
import os
from more_itertools import collapse
from datetime import datetime
from src.STATIC import ROOT_DIR

class Config:
    def __init__(self,
                 dataset_code,
                 task,
                 input_columns,
                 target_columns=None,
                 id_columns=None,
                 timestamp_column='datetime',
                 finetune=False,
                 pretrained_model_dir=None,
                 checkpoint_dir=None,
                 save_dir=None,
                 run_name=None,
                 context_length=512,
                 prediction_length=96,
                 patch_length=8,
                 stride=16,
                 data_batch_size=5000,
                 batch_train=None,
                 batch_val=None,
                 train_epochs=100,
                 num_workers=16,
                 batch_size=16,
                 freeze = False,
                 eval_metric='eval_f1',
                 greater_is_better=True,
                 test_dataset_code = None):

        # Initialize related configuration sections
        self._initialize_task(task, finetune)

        self._initialize_dataset_params(dataset_code, input_columns, target_columns, id_columns, timestamp_column, test_dataset_code)
        self._initialize_data_processing_params(context_length, prediction_length, patch_length, stride, data_batch_size)

        self._initialize_logging_params(pretrained_model_dir, checkpoint_dir, save_dir, run_name)
        self._initialize_training_params(batch_train, batch_val, train_epochs, num_workers, batch_size, freeze)
        self._initialize_evaluation_params(eval_metric, greater_is_better)

    def _initialize_task(self, task, finetune):
        self.finetune = finetune
        self.task = task
        
    def _initialize_dataset_params(self, dataset_code, input_columns, target_columns, id_columns, timestamp_column, test_dataset_code):
        self.dataset_code = dataset_code
        self.test_dataset_code = test_dataset_code or dataset_code
        self.input_columns = input_columns
        self.target_columns = target_columns or []
        self.id_columns = id_columns or []
        self.timestamp_column = timestamp_column
    
    def _initialize_data_processing_params(self, context_length, prediction_length, patch_length, stride, data_batch_size):
        self.data_loc = 'local' if os.path.exists(os.path.join(ROOT_DIR, f'e4data/train_test_split/{self.dataset_code}')) else 'remote'
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride
        self.data_batch_size = data_batch_size

    def _initialize_logging_params(self, pretrained_model_dir, checkpoint_dir, save_dir, run_name):
        self.run_name = run_name or f"{self.dataset_code}_{self.task}_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
        self.pretrained_model_dir = pretrained_model_dir or os.path.join(ROOT_DIR, "models/unlabelled_pretrain")
        self.checkpoint_dir = checkpoint_dir or os.path.join(ROOT_DIR, f"checkpoints/{self.run_name}")
        self.save_dir = save_dir or os.path.join(ROOT_DIR, f"models/{self.run_name}")

    def _initialize_training_params(self, batch_train, batch_val, train_epochs, num_workers, batch_size, freeze):
        self.train_epochs = train_epochs
        self.batch_train = batch_train if batch_train is not None else (True if self.data_loc == 'remote' else False)
        self.batch_val = batch_val if batch_val is not None else (True if self.batch_train and self.task == 'masked_prediction' else False)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.freeze = freeze
    
    def _initialize_evaluation_params(self, eval_metric, greater_is_better):
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better

    @property
    def relevant_columns(self):
        return list(collapse([self.timestamp_column, self.id_columns, self.input_columns, self.target_columns]))
    
    def set_attribute(self, **kwargs):
        """Set attributes based on provided keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Config.")
    
    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser(description="Configuration arguments for the model training")

        # Define all arguments for each parameter in the Config class
        parser.add_argument('--dataset_code', type=str, required=True, help="Dataset code")
        parser.add_argument('--task', type=str, required=True, help="Task type")
        parser.add_argument('--input_columns', type=str, nargs='+', required=True, help="List of input columns")
        parser.add_argument('--target_columns', type=str, nargs='*', default=None, help="List of target columns")
        parser.add_argument('--id_columns', type=str, nargs='*', default=None, help="List of ID columns")
        parser.add_argument('--timestamp_column', type=str, default='datetime', help="Timestamp column name")
        parser.add_argument('--finetune', action='store_true', help="Whether to finetune the model")
        parser.add_argument('--pretrained_model_dir', type=str, default=None, help="Path to pretrained model")
        parser.add_argument('--checkpoint_dir', type=str, default=None, help="Path to checkpoint directory")
        parser.add_argument('--save_dir', type=str, default=None, help="Directory to save model")
        parser.add_argument('--run_name', type=str, default=None, help="Run name")
        parser.add_argument('--context_length', type=int, default=512, help="Context length")
        parser.add_argument('--prediction_length', type=int, default=96, help="Prediction length")
        parser.add_argument('--patch_length', type=int, default=8, help="Patch length")
        parser.add_argument('--stride', type=int, default=16, help="Stride length")
        parser.add_argument('--data_batch_size', type=int, default=5000, help="Batch size for data")
        parser.add_argument('--batch_train', type=bool, default=None, help="Whether to use batch for training")
        parser.add_argument('--batch_val', type=bool, default=None, help="Whether to use batch for validation")
        parser.add_argument('--train_epochs', type=int, default=100, help="Number of epochs for training")
        parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for data loading")
        parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
        parser.add_argument('--eval_metric', type=str, default='eval_f1', help="Evaluation metric")
        parser.add_argument('--greater_is_better', type=bool, default=True, help="Whether higher eval metric is better")

        args = parser.parse_args()

        # Return a Config object initialized with the parsed arguments
        return Config(
            dataset_code=args.dataset_code,
            task=args.task,
            input_columns=args.input_columns,
            target_columns=args.target_columns,
            id_columns=args.id_columns,
            timestamp_column=args.timestamp_column,
            finetune=args.finetune,
            pretrained_model_dir=args.pretrained_model_dir,
            checkpoint_dir=args.checkpoint_dir,
            save_dir=args.save_dir,
            run_name=args.run_name,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            patch_length=args.patch_length,
            stride=args.stride,
            data_batch_size=args.data_batch_size,
            batch_train=args.batch_train,
            batch_val=args.batch_val,
            train_epochs=args.train_epochs,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            eval_metric=args.eval_metric,
            greater_is_better=args.greater_is_better
        )
    
# Default arguments
WESAD_DEFAULT_ARGS = {
    'dataset_code': 'WESAD',
    'task': 'classification',
    'input_columns': ['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
    'target_columns': 'binary_stress',
    'id_columns': ['subject_id','condition'],
    'finetune': False,
    'checkpoint_dir': os.path.join(ROOT_DIR, "checkpoint/stress_event_baseline/WESAD"),
    'save_dir': os.path.join(ROOT_DIR, "models/stress_event_baseline/WESAD"),
    'run_name': f"WESAD_benchmark_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
    'batch_train': False,
    'batch_val': False,
    'train_epochs': 100,
    'num_workers': 16,
    'batch_size': 16,
    'context_length': 512,
    'prediction_length': 96,
    'patch_length': 8,
    'stride': 16,
}

Nurses_CLASS_ARGS = WESAD_DEFAULT_ARGS.copy()
Nurses_CLASS_ARGS.update({
            'dataset_code': 'Nurses/labelled',
            'id_columns': ['subject_id', 'session_id'],
            'checkpoint_dir': os.path.join(ROOT_DIR, "checkpoint/stress_event_baseline/Nurses"),
            'save_dir': os.path.join(ROOT_DIR, "models/stress_event_baseline/Nurses"),
            'run_name': f"Nurses_benchmark_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        })

WESAD_FINETUNE_ARGS = WESAD_DEFAULT_ARGS.copy()
WESAD_FINETUNE_ARGS.update({
            'finetune': True,
            'pretrained_model_dir': os.path.join(ROOT_DIR, "models", "unlabelled_pretrain"),
            'checkpoint_dir': os.path.join(ROOT_DIR, "checkpoint/stress_event_finetune/unlabelled_pretrain"),
            'save_dir': os.path.join(ROOT_DIR, "models/stress_event_finetune/unlabelled_pretrain"),
            'run_name': f"WESAD_unlabelled_finetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        })

UNLABELLED_DEFAULT_ARGS = {
    'dataset_code': 'unlabelled',
    'task': 'masked_prediction',
    'input_columns': ['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
    'id_columns': ['source_id'],
    'finetune': False,
    'checkpoint_dir': os.path.join(ROOT_DIR, "checkpoint/unlabelled_pretrain"),
    'save_dir': os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
    'run_name': f"unlabelled_pretrain_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
    'stride': 1,
    'batch_train': False,
}

Nurses_FINETUNE_ARGS = UNLABELLED_DEFAULT_ARGS.copy()
Nurses_FINETUNE_ARGS.update({
    'dataset_code': f"Nurses/unlabelled",
    'id_columns': ['subject_id', 'session_id'],
    'finetune': True,
    'pretrained_model_dir': os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
    'checkpoint_dir': os.path.join(ROOT_DIR, f"checkpoint/Nurses_SSLFinetune"),
    'save_dir': os.path.join(ROOT_DIR, f"models/Nurses_SSLFinetune"),
    'run_name': f"Nurses_SSLFinetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
    'batch_train': False,
})
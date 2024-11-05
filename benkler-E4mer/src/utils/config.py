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
                 train_epochs=100,
                 num_workers=16,
                 batch_size=16,
                 eval_metric='eval_f1',
                 greater_is_better=True):

        # Initialize related configuration sections
        self._initialize_task(task, finetune)

        self._initialize_dataset_params(dataset_code, input_columns, target_columns, id_columns, timestamp_column)
        self._initialize_data_processing_params(context_length, prediction_length, patch_length, stride, data_batch_size)

        self._initialize_logging_params(pretrained_model_dir, checkpoint_dir, save_dir, run_name)
        self._initialize_training_params(batch_train, train_epochs, num_workers, batch_size)
        self._initialize_evaluation_params(eval_metric, greater_is_better)

    def _initialize_task(self, task, finetune):
        self.finetune = finetune
        self.task = task
        
    def _initialize_dataset_params(self, dataset_code, input_columns, target_columns, id_columns, timestamp_column):
        self.dataset_code = dataset_code
        self.input_columns = input_columns
        self.target_columns = target_columns
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

    def _initialize_training_params(self, batch_train, train_epochs, num_workers, batch_size):
        self.train_epochs = train_epochs
        self.batch_train = batch_train if batch_train is not None else (True if self.data_loc == 'remote' else False)
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def _initialize_evaluation_params(self, eval_metric, greater_is_better):
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better

    @property
    def relevant_columns(self):
        return list(collapse([self.timestamp_column, self.id_columns, self.input_columns, self.target_columns]))
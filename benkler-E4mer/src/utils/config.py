
import os
from datetime import datetime
from src.STATIC import ROOT_DIR

class Config:
    def __init__(self,
                 dataset_code,
                 task,
                 input_columns,
                 target_columns=None,
                 id_columns=None,
                 finetune=False,
                 pretrained_model_dir=None,
                 checkpoint_dir=None,
                 save_dir=None,
                 run_name=None,
                 train_epochs=100,
                 context_length=512,
                 prediction_length=96,
                 patch_length=8,
                 num_workers=16,
                 batch_size=16,
                 data_batch_size=5000):
        
        self.dataset_code = dataset_code
        self.task = task
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.id_columns = id_columns if id_columns else []
        self.finetune = finetune
        self.pretrained_model_dir = pretrained_model_dir or os.path.join(ROOT_DIR, f"models/unlabelled_pretrain")
        self.run_name = run_name or f"{dataset_code}_{task}_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
        self.checkpoint_dir = checkpoint_dir or os.path.join(ROOT_DIR, f"checkpoints/{self.run_name}")
        self.save_dir = save_dir or os.path.join(ROOT_DIR, f"models/{self.run_name}")
        self.train_epochs = train_epochs
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_batch_size = data_batch_size
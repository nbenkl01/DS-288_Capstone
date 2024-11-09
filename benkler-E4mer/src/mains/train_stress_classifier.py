import os
from src.model import model as model
from datetime import datetime
import argparse
from src.utils.config import Config
from src.STATIC import ROOT_DIR

def train_WESAD_benchmark():
    config = Config(dataset_code='WESAD', 
                    task='classification',
                    input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                    target_columns='binary_stress',
                    id_columns=['subject_id','condition'],
                    finetune=False,
                    checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/stress_event_baseline/WESAD"),
                    save_dir=os.path.join(ROOT_DIR, "models/stress_event_baseline/WESAD"),
                    run_name=f"WESAD_benchmark_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                    batch_train=False,
                    )
    model.run_training_task(config)

def train_Nurses_benchmark():
    config = Config(dataset_code='Nurses/labelled', 
                    task='classification',
                    input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                    target_columns='binary_stress',
                    id_columns=['subject_id','session_id'],
                    finetune=False,
                    checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/stress_event_baseline/Nurses"),
                    save_dir=os.path.join(ROOT_DIR, "models/stress_event_baseline/Nurses"),
                    run_name=f"Nurses_benchmark_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                    batch_train=False,
                    )
    model.run_training_task(config)

def finetune_stress_E4mer(pretrained_model='unlabelled_pretrain', test_dataset_code = None):
    config = Config(dataset_code='WESAD', 
                    test_dataset_code = test_dataset_code,
                    task='classification',
                    input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                    target_columns='binary_stress',
                    id_columns=['subject_id','condition'],
                    finetune=True,
                    freeze = False if pretrained_model == 'unlabelled_pretrain' else True,
                    pretrained_model_dir=os.path.join(ROOT_DIR, "models", pretrained_model),
                    checkpoint_dir=os.path.join(ROOT_DIR, f"checkpoint/stress_event_finetune/{pretrained_model}"),
                    save_dir=os.path.join(ROOT_DIR, f"models/stress_event_finetune/{pretrained_model}"),
                    run_name=f"WESAD_{pretrained_model.split('_')[0]}_finetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                    batch_train=False,
                    )
    model.run_training_task(config)

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune E4mer model on WESAD benchmark.")
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], required=True, help="Choose between training or fine-tuning.")
    parser.add_argument("--dataset", type=str, choices=["WESAD", "Nurses"], default='WESAD', help="Choose a dataset.")
    parser.add_argument("--test_Nurses", type=str, default=None, help="Specify the pretrained model directory for fine-tuning.")
    parser.add_argument("--pretrained_model", type=str, default="unlabelled_pretrain", help="Specify the pretrained model directory for fine-tuning.")

    args = parser.parse_args()

    if args.mode == "train":
        if args.dataset == 'WESAD':
            train_WESAD_benchmark()
        else:
            train_Nurses_benchmark()
    elif args.mode == "finetune":
        if args.test_Nurses is not None:
            finetune_stress_E4mer(args.pretrained_model, test_dataset_code=f"Nurses/{args.test_Nurses}/labelled")
        else:
            finetune_stress_E4mer(args.pretrained_model)

if __name__ == "__main__":
    main()
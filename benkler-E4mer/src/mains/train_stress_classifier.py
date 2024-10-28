import os
from src.model import model as model
from datetime import datetime
from src.STATIC import ROOT_DIR
import argparse

def train_WESAD_benchmark():
    model.train_classifier(
        'WESAD',
        input_columns=['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
        target_columns='binary_stress',
        finetune=False,
        checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/stress_event_baseline"),
        save_dir=os.path.join(ROOT_DIR, "models/stress_event_baseline"),
        run_name=f"WESAD_benchmark_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        train_epochs=20
    )

def finetune_stress_E4mer(pretrained_model='unlabelled_pretrain'):
    model.train_classifier(
        'WESAD',
        input_columns=['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
        target_columns='binary_stress',
        finetune=True,
        pretrained_model_dir=os.path.join(ROOT_DIR, "models", pretrained_model),
        checkpoint_dir=os.path.join(ROOT_DIR, f"checkpoint/stress_event_finetune/{pretrained_model}"),
        save_dir=os.path.join(ROOT_DIR, f"models/stress_event_finetune/{pretrained_model}"),
        run_name=f"WESAD_{pretrained_model.split('_')[0]}_finetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        train_epochs=20
    )

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune E4mer model on WESAD benchmark.")
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], required=True, help="Choose between training or fine-tuning.")
    parser.add_argument("--pretrained_model", type=str, default="unlabelled_pretrain", help="Specify the pretrained model directory for fine-tuning.")

    args = parser.parse_args()

    if args.mode == "train":
        train_WESAD_benchmark()
    elif args.mode == "finetune":
        finetune_stress_E4mer(args.pretrained_model)

if __name__ == "__main__":
    main()
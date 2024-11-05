import os
from src.model import model as model
from datetime import datetime
import argparse
from src.utils.config import Config
from src.STATIC import ROOT_DIR

def pretrain_E4mer_base():
    config = Config(dataset_code='unlabelled', 
                    task='masked_prediction',
                    input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                    id_columns=['source_id'],
                    finetune=False,
                    checkpoint_dir=os.path.join(ROOT_DIR, "checkpoint/unlabelled_pretrain"),
                    save_dir=os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
                    run_name=f"unlabelled_pretrain_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                    stride=1,
                    )
    model.run_training_task(config)

def finetune_nurse_SSL(nurse = None):
    config = Config(dataset_code=f"Nurses/{nurse or '.'}/unlabelled",
                    task='masked_prediction',
                    input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                    id_columns=['subject_id','session_id'],
                    finetune=True,
                    pretrained_model_dir=os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
                    checkpoint_dir=os.path.join(ROOT_DIR, f"checkpoint/Nurse{nurse or 's'}_SSLFinetune"),
                    save_dir=os.path.join(ROOT_DIR, f"models/Nurse{nurse or 's'}_SSLFinetune"), 
                    run_name=f"Nurse{nurse or 's'}_SSLFinetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                    batch_train=False,
                    stride=1,
                    )
    model.run_training_task(config)

def main():
    parser = argparse.ArgumentParser(description="Pretrain or fine-tune E4mer model.")
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune"], required=True, help="Choose between pretraining or fine-tuning.")
    parser.add_argument("--nurse", type=str, help="Specify nurse ID for fine-tuning.")

    args = parser.parse_args()

    if args.mode == "pretrain":
        pretrain_E4mer_base()
    elif args.mode == "finetune":
        if args.nurse:
            finetune_nurse_SSL(args.nurse)
        else:
            finetune_nurse_SSL()
            # print("Please provide a nurse ID with --nurse for fine-tuning.")

if __name__ == "__main__":
    main()
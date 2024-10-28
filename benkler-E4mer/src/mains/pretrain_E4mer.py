import os
from src.model import model as model
from datetime import datetime
from src.STATIC import ROOT_DIR
import argparse

def pretrain_E4mer_base():
    model.pretrain(
        'unlabelled',
        input_columns=['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
        # id_columns=['source_id'],
        finetune=False,
        run_name=f"unlabelled_pretrain_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        train_epochs=20
    )

def finetune_nurse_SSL(nurse):
    model.pretrain(
        f'Nurses/{nurse}/unlabelled',
        input_columns=['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
        # id_columns=['source_id'],
        finetune=True,
        pretrained_model_dir=os.path.join(ROOT_DIR, "models/unlabelled_pretrain"),
        checkpoint_dir=os.path.join(ROOT_DIR, f"checkpoint/Nurse{nurse}_SSLFinetune"),
        save_dir=os.path.join(ROOT_DIR, f"models/Nurse{nurse}_SSLFinetune"), 
        run_name=f"Nurse{nurse}_SSLFinetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        train_epochs=20
    )

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
            print("Please provide a nurse ID with --nurse for fine-tuning.")

if __name__ == "__main__":
    main()
import os
from src.model import model as model
from datetime import datetime
from src.STATIC import ROOT_DIR

def pretrain_E4mer_base():
    model.pretrain('unlabelled',
                    input_columns = ['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
                #    id_columns=['source_id'],
                    finetune = False,
                    run_name = f"unlabelled_pretrain_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                    train_epochs = 100)


def finetune_nurse_SSL(nurse): # Move to pretrain_E4mer
    model.pretrain(f'Nurses/{nurse}/unlabelled',
                       input_columns = ['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
                    #    id_columns=['source_id'],
                       finetune = True,
                       pretrained_model_dir=os.path.join(ROOT_DIR, "benkler-E4mer/models/unlabelled_pretrain"),
                       checkpoint_dir=os.path.join(ROOT_DIR, "benkler-E4mer/checkpoint/Nurse{nurse}_SSLFinetune"),
                        save_dir=os.path.join(ROOT_DIR, "benkler-E4mer/models/Nurse{nurse}_SSLFinetune"), 
                       run_name = f"Nurse{nurse}_SSLFinetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                        train_epochs = 100)

# def main():
    
# if __name__ == "__main__":
#     main()
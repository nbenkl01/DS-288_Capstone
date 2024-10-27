import os
from src.model import model as model
from datetime import datetime
from src.STATIC import ROOT_DIR

def train_WESAD_benchmark():
    model.train_classifier('WESAD',
                       input_columns = ['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
                        target_columns = ['binary_stress'],
                       finetune = False,
                        checkpoint_dir=os.path.join(ROOT_DIR, "benkler-E4mer/checkpoint/stress_event_baseline"),
                         save_dir=os.path.join(ROOT_DIR, "benkler-E4mer/models/stress_event_baseline"), 
                       run_name = f"WESAD_benchmark_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                        train_epochs = 100)
    
def finetune_stress_E4mer(pretrained_model = 'unlabelled_pretrain'):
    model.train_classifier('WESAD',
                       input_columns = ['acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean'],
                        target_columns = ['binary_stress'],
                       finetune = True,
                       pretrained_model_dir=os.path.join(ROOT_DIR, "benkler-E4mer/models", pretrained_model),
                       checkpoint_dir=os.path.join(ROOT_DIR, "benkler-E4mer/checkpoint/stress_event_finetune", pretrained_model),
                        save_dir=os.path.join(ROOT_DIR, "benkler-E4mer/models/stress_event_finetune", pretrained_model), 
                       run_name = f"WESAD_{pretrained_model.split('_')[0]}_finetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
                        train_epochs = 100)


# def main():
    
    
# if __name__ == "__main__":
#     main()
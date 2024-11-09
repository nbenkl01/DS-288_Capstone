# import os
from src.model import model as model
# from src.STATIC import ROOT_DIR
from src.mains.pretrain_E4mer import pretrain_E4mer_base, finetune_nurse_SSL
from src.mains.train_stress_classifier import train_WESAD_benchmark, train_Nurses_benchmark, finetune_stress_E4mer

# NURSES = ['94', 'DF', 'F5', 'E4', '7E', 
#           '6B', '5C', '6D', 'BG', 'EG', 'CE',
#           '83', '15', '8B', '7A']

NURSES = ['5C','6B', '7A', '8B', 'DF', 'F5', 'E4',
            'BG']

def main():
    # pretrain_E4mer_base()
    # train_WESAD_benchmark()
    # finetune_stress_E4mer(pretrained_model='unlabelled_pretrain')
    # nurses = os.listdir(os.path.join(ROOT_DIR,'e4data/train_test_split/Nurses'))
    # train_Nurses_benchmark()
    for nurse in NURSES: #listdir will only work on local machine
        finetune_nurse_SSL(nurse)
        finetune_stress_E4mer(pretrained_model = f"Nurse{nurse}_SSLFinetune")

if __name__ == "__main__":
    main()
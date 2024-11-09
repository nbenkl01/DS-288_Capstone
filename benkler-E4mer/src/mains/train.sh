# WESAD
python src/mains/train_stress_classifier.py --mode train #Baseline
python src/mains/train_stress_classifier.py --mode finetune --pretrained_model unlabelled_pretrain #Finetune

# Nurses
python src/mains/train_stress_classifier.py --mode train --dataset Nurses # Baseline
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses . --pretrained_model Nurses_SSLFinetune #Finetune

# Individual
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 5C --pretrained_model Nurse5C_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 6B --pretrained_model Nurse6B_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 7A --pretrained_model Nurse7A_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 8B --pretrained_model Nurse8B_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses DF --pretrained_model NurseDF_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses F5 --pretrained_model NurseF5_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses E4 --pretrained_model NurseE4_SSLFinetune
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses BG --pretrained_model NurseBG_SSLFinetune
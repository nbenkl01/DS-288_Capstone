# Pretrain
python src/mains/pretrain_E4mer.py --mode pretrain

# WESAD
python src/mains/train_stress_classifier.py --mode train
python src/mains/train_stress_classifier.py --mode finetune --pretrained_model unlabelled_pretrain

# Nurses
python src/mains/pretrain_E4mer.py --mode finetune --nurse 5C
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 5C --pretrained_model Nurse5C_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse 6B
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 6B --pretrained_model Nurse6B_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse 7A
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 7A --pretrained_model Nurse7A_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse 8B
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses 8B --pretrained_model Nurse8B_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse DF
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses DF --pretrained_model NurseDF_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse F5
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses F5 --pretrained_model NurseF5_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse E4
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses E4 --test_Nurses BG --pretrained_model NurseE4_SSLFinetune

python src/mains/pretrain_E4mer.py --mode finetune --nurse BG
python src/mains/train_stress_classifier.py --mode finetune --test_Nurses BG --test_Nurses BG --pretrained_model NurseBG_SSLFinetune
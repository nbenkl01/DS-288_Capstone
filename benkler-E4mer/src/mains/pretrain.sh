# Pretrain
python src/mains/pretrain_E4mer.py --mode pretrain

# Nurses Global
python src/mains/pretrain_E4mer.py --mode finetune

#Nurses Individual
python src/mains/pretrain_E4mer.py --mode finetune --nurse 5C
python src/mains/pretrain_E4mer.py --mode finetune --nurse 6B
python src/mains/pretrain_E4mer.py --mode finetune --nurse 7A
python src/mains/pretrain_E4mer.py --mode finetune --nurse 8B
python src/mains/pretrain_E4mer.py --mode finetune --nurse DF
python src/mains/pretrain_E4mer.py --mode finetune --nurse F5
python src/mains/pretrain_E4mer.py --mode finetune --nurse E4
python src/mains/pretrain_E4mer.py --mode finetune --nurse BG
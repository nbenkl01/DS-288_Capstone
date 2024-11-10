### Fails
- Failed to build virtual env necessary to remotely execute code on tufts servers (disk space allocation issue)
- Data streaming & training showed new issue where ssh connection to host drops during data fetching
- Failed to pretrain SSL models due to above failure (See general progress)

### Successes
- Finished modified schedule
- Resolved issues with remote env setup
- Developed batch data fetching & training system to avoid process crashing during pretraining
- Trained baseline WESAD & Nurses stress event detection classifiers

#### General Progress
- Ensured all training code works as hoped for below processes (by testing over samples for 1 epoch) so ready for full training run once data streaming failure is resolved:
  - Pretrain E4Mix base
  - Tune E4Mix nurse (baseline & individual)
  - Finetune all pretrained E4Mix models on:
    - WESAD binary stress state classification
- Began writing evaluation scripts

### Goals
- Resolve data streaming issue
- Finish masked prediction pretraining
- Finish stress event classification finetuning
- Do model evaluation
- Update Mission Statement
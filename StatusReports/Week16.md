### Fails
- Wrestling w/ E4mer codebase prooved significantly more complex to disentangle from barcelona task than anticipated
- Still no modified schedule, need to figure out what is plausible

### Successes
- Successfully reoriented E4mer data preprocessing & segmentation scripts to work w/ labelled Nurses & WESAD
- Successfully oriented existing E4mer code so it executes when passed WESAD labelled dataset (Not sure this is a real succcess)
  - While reworking high-level training scripts means pretraining & finetuning code properly executes over WESAD labelled dataset something seems very wrong about results.

### Goals
- Finish modified schedule
- Need to do more digging deeper into codebase to figure out how internal model scripts handle labelled data
- Figure out EXACTLY how E4mer code works or decide to scrap it & build your own.
- Finetune E4mer on WESAD binary stress state prediction
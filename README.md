# Data Science Capstone Project

Repository for work surrounding Tufts MSDS Capstone project.

## Mission Statement (Project Proposal)

Our mission is to explore the use of self-supervised learning (SSL) in improving transferability of physiological data models from controlled laboratory settings, where precise labels are readily available, to real-world environments, where such labels are scarce. In particular, we employ personalized representation learning to support individualized results. In pursuit of this mission, we study the use of physiological data from non-invasive wearables in detecting acute psychological stress events. Through this work, we aim to better transfer insights from controlled environments to practical scenarios, ultimately fostering greater adaptability, accessibility, and efficacy of personalized physiological modeling.


## Weekly Status Report (Weeks 17 & 18--see ./StatusReports/ for Week 16)
### Week 18
#### Fails
- Failed to build virtual env necessary to remotely execute code on tufts servers (disk space allocation issue)
- Data streaming & training showed new issue where ssh connection to host drops during data fetching
- Failed to pretrain SSL models due to above failure (See general progress)

#### Successes
- Finished modified schedule
- Resolved issues with remote env setup
- Developed batch data fetching & training system to avoid process crashing during pretraining
- Trained baseline WESAD & Nurses stress event detection classifiers

##### General Progress
- Ensured all training code works as hoped for below processes (by testing over samples for 1 epoch) so ready for full training run once data streaming failure is resolved:
  - Pretrain E4Mix base
  - Tune E4Mix nurse (baseline & individual)
  - Finetune all pretrained E4Mix models on:
    - WESAD binary stress state classification
- Began writing evaluation scripts

#### Goals
- Resolve data streaming issue
- Finish masked prediction pretraining
- Finish stress event classification finetuning
- Do model evaluation
- Update Mission Statement

### Week 17
#### Fails
- Failed to properly parse and understand exact nature of training code from E4selflearning to an extent necessary that I would be comfortable using it.
- Failed to build virtual env necessary to remotely execute code on tufts servers (disk space allocation issue)
- Still no modified schedule, need to figure out what is plausible

#### Successes
- Scrapped open-source E4selflearning aproach in favor of building my own (I put this under successes b/c I think it was the correct decision):
  - Weighted oportunity cost of adequately unravelling existing code vs. scrapping that codebase and building mine from the start & decided I would back the second option
- Shifted data processing approach:
  - Returned data preprocessing to flirt program from competency demo:
    - Both unlabelled & labelled corpora
  - Implemented GroupKFold dataset stratified splitting:
    - No single subject is seen in any two of trainig/val/test for WESAD & stratified by condition label
    - No single stress session is seen in any two of trainig/val/test for Nurses & stratified by condition label
    - Unlabelled not grouped, just stratified for even distribution over the source datasets
- Shifted to alternate modeling approach:
  - Shifted to huggingface Trainer to handle model training processes
  - Found granite-tsfm (Time Series Foundation Models) library to use for modeling-related functionality:
    - Integrated TimeSeriesPreProcessor for data segmentation & preparation for use as torch DataLoader after flirt feature extraction
  - Switched to PatchTSMixerModel model architecture:
    - Set up & tested PatchTSMixerForPretraining for pretraining
    - Set up PatchTSMixerForTimeseriesClassification for stress event detection
  - Set up CustomPatchTSMixerForTimeSeriesClassification & CustomPatchTSMixerForTimeSeriesClassificationOutput to allow personalized metric computation
- Wrote modular scripts to efficiently handle data loading, modeling, logging and evaluation
- Integrated code collection w/ DS-288_Capstone
- Successfully Tested modeling functionality
- Set up data streaming code to allow remote training via tufts servers
  - Tested said code

#### General Progress
- Attempted to stage code to run on tufts servers.
- Working w/ tufts eecs staff to help address (disk space allocation issue)

#### Goals
- Finish modified schedule
- Resolve issues with remote env setup
- Pretrain E4Mix base
- Tune E4Mix nurse
  - Do so for all nurses (individualized representation models)
- Finetune all pretrained E4Mix models on:
  - WESAD binary stress state classification
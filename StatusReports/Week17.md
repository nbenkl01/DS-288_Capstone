### Fails
- Failed to properly parse and understand exact nature of training code from E4selflearning to an extent necessary that I would be comfortable using it.
- Failed to build virtual env necessary to remotely execute code on tufts servers (disk space allocation issue)
- Still no modified schedule, need to figure out what is plausible

### Successes
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

### Goals
- Finish modified schedule
- Resolve issues with remote env setup
- Pretrain E4Mix base
- Tune E4Mix nurse
  - Do so for all nurses (individualized representation models)
- Finetune all pretrained E4Mix models on:
  - WESAD binary stress state classification
### Fails
- No DUA Approval yet
- CogWear data turned out to have no timestamp for stroop test so was only useful for simplistic (1-Cognitive Effort, 0-Resting) binary state classification

### Successes
Found Another Alternative Data Source Using (Empatica E4) wristband Data:
Wearable Stress and Affect Detection (WESAD) Dataset:
- 15 participants wore E4 during 4 states: baseline, amusement, stress, meditation
- After state transitions took standardized surveys (unused in study which focused on 3 affect state detection)

### General Progress
Began cleaning WESAD data and finally setting up prediction pipeline:
  1. Clean E4 data using biomarker processing from last week's journal
  2. (In progress) Extract important features from surveys (PANAS, SSSQ, STAI)
  3. (In progress) Link activity-intermitent survey responses to end of wearable timeseries
     - some difficulty here given non-standard timezone allignment in E4 and Respiban, with which survey timing is linked. (Need to align using double-tap signal)
  4. (In progress) Setting up iTransfromer architecture for predicting survey-relevant emotional signals from proceeding wearable timeseries.
  5. (To-Do) Profile emotional states (GaussianMixtureModeling?).
  6. (To-Do) Apply WESAD trained model to Exam Stress Dataset, Predict emotional state profiles.
  7. (To-Do) Train classifier (RF, XGBoost, ...,? decide on alg) to predict exam performance from emotional state profile.
     - Train on first two tests.
     - Test on second two tests.
  8. (To-Do) Create integrated pipeline to predict individual's cognitive state profile from wearable datastream and signal individual when profile trajectory likely to lead to performance decrease.
(In progress) Begun working on design document.

### Goals
- Finish technical competency demo.
  - Align surveys with E4 data
  - Train simplistic iTransformer
  - Report simple results
- Continue work on design document.
  - Finish rough, signposted, outline/bullet-draft
- (backup plan) Continue trying to finally Access TILES 2018 & TILES 2019
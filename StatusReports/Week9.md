### Fails
- No DUA Approval yet
- Failed to construct more complex model architectures (iTransformer/Transformer/LSTM) in time for competency demo submission.

### Successes
- Completed Competency Demo.
- Completed first-draft, structured data loading and modeling pipeline for use during next project phase.
- Completed progress points from last week:
  1. ~Clean E4 data using biomarker processing from last week's journal~
  2. ~Extract important features from surveys (PANAS, SSSQ, STAI)~
  3. ~Link activity-intermitent survey responses to end of wearable timeseries~
- Completed lit-review for Design Document.
- Parameterized "fallback" and "stretch" goals

### General Progress
- (In progress) Continue working on design document.
- Continued work on prediction pipeline:
    1. (In progress) Setting up iTransfromer architecture for predicting survey-relevant emotional signals from proceeding wearable timeseries.
    2. (To-Do) Profile emotional states (GaussianMixtureModeling?).
    3. (To-Do) Apply WESAD trained model to Exam Stress Dataset, Predict emotional state profiles.
    4. (To-Do) Train classifier (RF, XGBoost, ...,? decide on alg) to predict exam performance from emotional state profile.
       - Train on first two tests.
       - Test on second two tests.
    5. (To-Do) Create integrated pipeline to predict individual's cognitive state profile from wearable datastream and signal individual when profile trajectory likely to lead to performance decrease.

### Goals
- Continue work on design document.
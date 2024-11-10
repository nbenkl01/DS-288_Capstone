### Fails
- No DUA Approval yet

### Successes
Found 3 Alternative Sources Using (Empatica E4) wristband Data:
1. Exam Stress Dataset: 
   - Students wore E4 during 3 midterm tests
   - Use: Measure Cognitive Performance
   - Target: Exam score
2. CogWear: detect cognitive effort & attention with consumer-grade wearables? 
   - Individuals wore E4 wristbands under 3 conditions:
     - Resting
     - Stroop Test (Name text color BLUE--written in red ink)
     - 4 Surveys (2 “gamified” 2 Standard)
   - Use: Measure changes in individual's attention.
   - Target: Stroop test mistakes/minute, survey response/minute
3. Stress Detection in Nurses:
   - 15 nurses wore E4 during covid (1,250 hours collected in 2 sessions: Apr-May, Nov-Dec)
   - Use: Large db. for unsupervised learning of biomarker "shapelets"
Found biomarker preprocessing libraries and implemented feature extraction:
   - Electrodermal Activity (EDA): Tonic & Phasic means
     - Phasic component: fast-changing skin conductance response (SCR) -- stimulus-dependent fast changing signal (more correlated to brain activity)
     - Tonic component: slow-changing skin conductance level (SCL) --  continuous and slow-changing (signals throughout activity)
   - Heart Rate Variability (HRV):
     - Root mean square of successive differences between normal heartbeats (RMSSD).
   - Accelerometer (ACC):
     - Min, mean, & max L2 -- Euclidean distance traveled from origin during sliding window.

### Goals
- Try to finally Access TILES 2018 & TILES 2019
- Determine final question (Transfer learning of does attention influence performance? -- train to detect attention fails on CogWear, test on Exam Stress, do pred attention decreces in students correlate with poor performance)
- Figure out modeling approach (iTransformers for forcasting, Event DTW for similarity scoring, etc...)
- Put together technical competency demo.
- Start sketching out design document.
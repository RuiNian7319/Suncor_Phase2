# Suncor Pipeline Optimization Project
Given a flow rate, how should I operate to get there?

# Data files in data/Optimization_Data
Comments about data:  Flow rates are all shifted up by 1 minute to reflect time delay (at speed of sound)

Changables: Flow, dra readbacks, on/off pumps, pressure at Cheyenne, vfd currents

Opti_flow_dra_current: Data with Fort Lupton flow rate, all DRA ppm, and both VFD current and booster pump at Cheyenne

Opti_withAllPressure: Includes data with all pressures from along the pipeline

Opti_withAllChangable: Includes only changable data

Opti_withAllChangableDen: Includes only changable data and density at Cheyenne, CIG, Ault

Opti_withAllChangableDenNoFL: Above without discharge pressure at FL

Opti_withAllChangableDenCur: Includes all changable, density, and vfd currents
Opti_withAllChangableDenCurv2: Includes all changable, density, and Fort Lupton vfd current
Opti_withAllChangableDenCurv3: Includes all changable (no pressures), density, and vfd currents

Opti_withAllChangable: Includes all changable (density, ppm merged & smoothed), and vfd currents
Opti_withAllChangablev2: Includes all changable (density, ppm merged, non-smoothed), and vfd currents

Opti_1stPrinc: Dataset for 1st principle modelling.  2x dra, 2x vfd, 4x pump

Opti_withAllChangable: [Flow, DRA x4, On/off pumps x4, VFDs x2, Density x3]
Opti_withAllChangableHalf: [Flow, DRA x4, On/off pumps x4, VFDs x2, Density x3] except half

# Important findings
1. Fort Lupton pressure reading is 90% correlated with pressure.
2. Density increases prediction accuracy
3. VFD Current at Cheyenne matters a lot
4. Unnormalized data increases training time significantly
5. Inlet pressure has nothing to do with outlet pressure when predicting pressure for constraints
6. Inlet and outlet pressures might both matter for current-to-pressure
  - Some deltaP are 0, but the VFDs are still on


# Updates
### March 6th
Change all error metrics to RMSE

Linear Regression with all pressure (Opti_withAllPressure):
RMSE: 101
MAE: 71

Adding Ault pumps & not filtering when Cheyenne booster pumps are not on.


Linear Regression with all pressure (Opti_withAllPressure):
RMSE: 101
MAE: 69

Neural Network with all pressure (Opti_withAllPressure):
RMSE: 49
MAE: 33

### March 7th
Add seaborn variance plot
Optimization: PPM is more expensive to raise given the desired flow rate

Linear Regression (Opti_withAllChangable):
RMSE: 135
MAE: 92

Neural Network (Opti_withAllChangable):
RMSE: 96
MAE: 65

Quadratic Model (Opti_withAllChangable) - Incorrect data probably:
RMSE: 145
MAE: 100

### March 8th
Added Fort Lupton ON/OFF pump

Linear Regression (Opti_withAllChangable):
RMSE: 135
MAE: 92

Neural Network (Opti_withAllChangable):
RMSE: 96
MAE: 66

Linear Regression (Opti_withAllChangableDen):
RMSE: 131
MAE: 90

Fort Lupton pump didn't add much..

### March 9th
Added density filtering

Linear Regression (Opti_withAllChangableDen):
RMSE: 124
MAE: 84

Neural Network (Opti_withAllChangableDen):
RMSE: 87
MAE: 60

Sorted validation plots without randomizing.
Error can be very high, random spikes in the plot.

The VFD at Fort Lupton is messing with results.

### March 10th
Remove discharge pressure at Fort Lupton to remove spikes.

Linear Regression (Opti_withAllChangableDenNoFL):
RMSE: 207
MAE: ~140

Neural Network (Opti_withAllChangableDenNoFL):
RMSE: 140
MAE: 103

Building quadratic model

Quadratic model (Opti_withAllChangableDenNoFL):
RMSE`: 200
MAE:  ~135

Build sqrt model


### March 11th
Add currents at Cheyenne and FL

Linear Regression (Opti_withAllChangableDenCur):
RMSE: 183
MAE: 137

Neural Network (Opti_withAllChangableDenCur):
RMSE: 74
MAE: 50

Remove VFD Current at Cheyenne, because we have pressure there

Linear Regression (Opti_withAllChangableDenCurv2):
RMSE: 204
MAE: 157

Neural Network (Opti_withAllChangableDenCurv2):
RMSE: 132
MAE: 96

Add VFD Current at Cheyenne, remove all pressure readings

Linear Regression (Opti_withAllChangableDenCurv3):
RMSE: 184
MAE: 136

Neural Network (Opti_withAllChangableDenCurv3):
RMSE: 112
MAE: 82

Quadratic Model (Opti_withAllChangableDenCurv3):
RMSE: 167
MAE: 124

Sqrt Model (Opti_withAllChangableDenCurv3):
RMSE: 162
MAE: 122

Sqrt Model w/ intercept (Opti_withAllChangableDenCurv3):
RMSE: 162
MAE: 122

### March 12th
Fixed error bug, it was because of no data reshaping.
Adding exponentionally smoothing makes the plots look better and reduces error
Separated modules into smaller sub-components

Try linear / quadratic model with intercept

Linear Regression w/ intercept(Opti_withAllChangableDenCurv3):
RMSE: 181
MAE: 136

Quadratic Model w/ intercept (Opti_withAllChangableDenCurv3):
RMSE: 167
MAE: 124

### March 13th
Removing normalization

Linear Regression w/ intercept non-normalized (Opti_withAllChangableDenCurv3):
Ran for 4500 epochs vs. around 3 (requires a hr).
RMSE: 181
MAE: 135

Building pressure constraint models

Linear Regression w/ intercept non-normalized (Chey_Pres):
Ran for 1000 epochs.
RMSE: 130
MAE: 88

### March 14th
Build 5 models:
1. LP DRA $/ppm as a function of flow rate
2. EP DRA $/ppm as a function of flow rate
3. Average DRA $/ppm as a function of flow rate
4. Pressure from current & inlet pressure at Cheyenne
5. Pressure from current at Fort Lupton

The three DRA $/ppm are finished, done using Excel, accuracy is high.

Linear Regression w/ intercept non-normalized (Cur2Pres_Chey):
RMSE: 96
MAE: 76

Linear Regression w/ intercept non-normalized (Cur2Pres_FL):
RMSE: 30
MAE: 22

### March 15th
Constrained model built

Linear Regression w/ intercept normalized, with Constraint (Opti_withAllChangableDenCurv3):
RMSE: 181
MAE: 136

Linear Regression w/ intercept un-normalized, with Constraint (Opti_withAllChangableDenCurv3):
RMSE: 183
MAE: 136

Model with dra ppm merged and smoothed

Linear Regression w/ intercept normalized (Opti_withAllChangable):
RMSE: 188
MAE: 140

Linear Regression w/ intercept normalized and constrained (Opti_withAllChangable):
RMSE: 195
MAE: 96

Linear Regression w/ intercept normalized (Opti_withAllChangablev2):
RMSE: 190
MAE: 140

Linear Regression w/ intercept normalized and constrained (Opti_withAllChangablev2):
RMSE: 195
MAE: 142

### March 18th
Build 1st principle model
- Requires a lot longer to train

First principles performs a lot worse

Linear Regression 1st principle (Opti_1stPrinc):
RMSE: 250
MAE: 200

Linear Regression 1st principle, no normalization (Opti_1stPrinc):
RMSE: 254
MAE: 203

### March 19th
Objectizing LS

Done

### March 20th
Objectizing NN

### March 21st
Building new data set: [Flow, DRA x4, On/off pumps, VFDs, Density]
Taking only half the data to build a more accurate model.

Linear Regression (Opti_withAllChangable):
RMSE: 181
MAE: 135

Linear Regression (Opti_withAllChangableHalf):
RMSE: 182
MAE: 130

Linear Regression - no Regularization (Opti_withAllChangableHalf):
RMSE: 182
MAE: 130

Linear Regression Constrained (Opti_withAllChangable):
RMSE: 181
MAE: 136

Linear Regression Constrained (Opti_withAllChangableHalf):
RMSE: 181
MAE: 131

Linear Regression Constrained and Non-normalized (Opti_withAllChangable):
RMSE: 182
MAE: 136

### March 22nd

NN - 30 node layers (Opti_withAllChangable):
RMSE: 126
MAE: 95

NN - 30 node layers (Opti_withAllChangableHalf):
RMSE: 106
MAE: 77

Building pressure constraint models:  Seems that Ault is in the most danger of exceeding pressure.

Cheyenne (Safety Sensitive: Medium)

Linear Regression (Chey_Pres):
RMSE: 134
MAE: 98

NN - 30 node layers (Chey_Pres):
RMSE: low
MAE: low

Ault (Safety Sensitive: Severe)

Linear Regression (Ault_Pres):
RMSE: 200
MAE: 161

NN - 30 node layers (Ault_Pres):
RMSE: low
MAE: low

Fort Lupton (Safety Sensitive: Mild)

Linear Regression (FL_Pres):
RMSE: 42
MAE: 33

NN - 30 node layers (FL_Pres):
RMSE: low
MAE: low

**Remove DRA readings**

Ault (Safety Sensitive: Severe)

Linear Regression (Ault_Pres):
RMSE: 220
MAE: 178

Fort Lupton (Safety Sensitive: Mild)

Linear Regression (FL_Pres):
RMSE: Failed
MAE: Failed

### March 22nd

Linear Regression (Chey_Pres):
RMSE: 134
MAE: 98
**NOTES: A lot of actual pressure are much higher than predicted... can be easily seen in predictions vs actual**

Linear Regression (Ault_Pres):
RMSE: 196
MAE: 156
**NOTES: Fits are kinda bad, like really bad**

Linear Regression (FL_Pres):
RMSE: 41
MAE: 32
**NOTES: Real reading is insanely noisy**

Linear Regression - Pressure smoothed (FL_Pres):
RMSE: 39
MAE: 31

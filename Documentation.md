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

# Important findings
1. Fort Lupton pressure reading is 90% correlated with pressure.
2. Density increases prediction accuracy
3. VFD Current at Cheyenne matters a lot


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


### March 10th
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



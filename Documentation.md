# Suncor Pipeline Optimization Project
Given a flow rate, how should I operate to get there?

# Data files in data/Optimization_Data
Comments about data:  Flow rates are all shifted up by 1 minute to reflect time delay (at speed of sound)

Opti_flow_dra_current: Data with Fort Lupton flow rate, all DRA ppm, and both VFD current and booster pump at Cheyenne

Opti_withAllPressure: Includes data with all pressures from along the pipeline

Opti_withAllChangable: Includes only changable data



# Updates
### March 6th
Change all error metrics to RMSE

Linear Regression with all pressure (Opti_withAllPressure):
RMSE: 101
MAE: 71

Adding Ault pumps & not filtering when Ault booster pumps are not on.


Linear Regression with all pressure (Opti_withAllPressure):
RMSE: 101
MAE: 69

Neural Network with all pressure (Opti_withAllPressure):
RMSE: 49
MAE: 33

### March 6th

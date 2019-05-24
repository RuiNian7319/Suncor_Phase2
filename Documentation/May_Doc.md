# May data doc


**Original**
2019 Train data
RMSE: 106
MAE: 76

2019 May data
RMSE: 173
MAE: 143

1. Swapped densities to binary values (0, 1) and smoothed temperatures

2019 Train data
RMSE: 116
MAE: 89

2019 May data
RMSE: 147
MAE: 112

2. Swapped densities to binary values (1, 2) and smoothed temperatures

2019 Train data
RMSE: 118
MAE: 91

2019 May data
RMSE: 148
MAE: 112

Comment: No change.

3. Build model on only May data
Comments: Fit is good, can't capture peaks well.

4. Delete weird low flow rates in historical data and build model on old data, then validate on new data.

2019 Train data
RMSE: 98
MAE: 73

2019 May data
RMSE: 127
MAE: 93

5. Built models for the train and test splits

Training data
RMSE: 102
MAE: 75

Test data
RMSE: 75
MAE: 54



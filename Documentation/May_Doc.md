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


# Pressure instead of Current Models
1. Build models using pressure instead of current

2019 Train data
RMSE: 97
MAE: 75

2019 May data
RMSE: 152
MAE: 120

2. Build model for the train and test splits

Training data
RMSE: 102
MAE: 75

Test data
RMSE: 84
MAE: 62


# Current2Pressure Models
1. Build current to pressure models

2019 Train data
RMSE: 72
MAE: 56

2019 May data
RMSE: 47
MAE: 39

# Current2Pressure Models split

**Sweet data**

2019 Train data
RMSE: 65
MAE: 46

2019 Test data
RMSE: Good after intercept correction
MAE: Good after intercept correction

**Sour data**

2019 Train data
RMSE: 63
MAE: 49

2019 Test data
RMSE: Good after intercept correction
MAE: Good after intercept correction

# Current2Pressure Ault pump split

Good sweet:
RMSE: 138
MAE: 116

Bad sweet:
RMSE: 174
MAE: 165

Good sour:
RMSE: 86
MAE: 71

Bad sour:
RMSE: 172
MAE: 167

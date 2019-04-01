# Folder Layout

## Sub-folders

**linear_regression_models**: Linear models for overall optimization. Regressors: X <br>
**2019_models: 2019 Models**
**sqrt_models**: Sqrt models for overall optimization.  Regressor: sqrt(X), X <br>
**quadratic_models**: Quadratic models for overall optimization.  Regressors: X^2, X <br>
**neural_network_models**: Neural networks for overall optimization.  Regressors: X <br>
**pressure_models**: Pressure constraint models.  Built for discharge at Cheyenne, Ault, and Fort Lupton. <br>
**pressure_current_models**:  Transforming currents to an expected pressure, so the operators can put that as a set-point. <br>

## Sub-modules

**EWMA**: Exponentially weighted moving average code.  Used for exponentially smoothing data. <br>
**MinMaxNorm**: Min-Max normalization. $x_{norm} = \frac{x_i - x_{min}}{x_{max} - x_{min]}$ <br>
**Seq_plot**: Sequential plot definition to validate the machine learning models in production <br>
**DRA_Curves**: DRA Cost curves <br>

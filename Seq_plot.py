"""
Sequential plot definition to validate the machine learning models in production

Rui Nian
"""

import numpy as np
import tensorflow as tf
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def seq_pred(session, model, features, data, normalizer, time_start, time_end, xlabel='Time', ylabel='Flow Rate',
             adv_plot=True, savefig=False):
    """
    Description
        ----
        Plots the time-series data, used for validation in production. Currently, data is being shuffled; therefore,
        the validation plots data from random time steps, causing difficulty in visualizing the true accuracy of
        the model.

    Inputs
        ----
             session: Tensorflow session. Usually: sess
               model: Tensorflow regression model. Ex: z = tf.matmul(W, x) + b
            features: Tensorflow placeholder for the features.
                data: Raw data to test.  Data in form of [labels, features]
          normalizer: Normalization object.  If un-normalized data is preferred, pass normalizer=None
          time_start: Start index in the data
            time_end: End index in the data
              xlabel: Label of the x-axis
              ylabel: Label of the y-axis
            adv_plot: Plot with fancy error bars

    Outputs (No returns)
        ----
               RMSE: Root mean squared error of the data between time_start and time_end
                MAE: Mean absolute error of the data between time_start and time_end
               Plot: Plot of the actual vs. predicted
    """
    # Normalize data
    if normalizer is not None:
        data = normalizer(data)

    # Label should be first column, features follows
    plot_x = data[time_start:time_end, 1:]
    plot_y = data[time_start:time_end, 0]

    # Reshape the data for tensorflow
    plot_x = plot_x.reshape(-1, data.shape[1] - 1)
    plot_y = plot_y.reshape(-1, 1)

    # Run tensorflow model to get predicted y
    preds = session.run(model, feed_dict={features: plot_x})

    # Unnormalize data
    if normalizer is not None:
        preds = np.multiply(preds, normalizer.denominator[0, 0])
        preds = preds + normalizer.col_min[0, 0]

        plot_y = np.multiply(plot_y, normalizer.denominator[0, 0])
        plot_y = plot_y + normalizer.col_min[0, 0]

    # RMSE & MAE Calc
    rmse_loss = np.sqrt(np.mean(np.square(np.subtract(plot_y, preds))))
    mae_loss = np.mean(np.abs(np.subtract(plot_y, preds)))

    print('Seq. RMSE: {:2f} | Seq. MAE: {:2f}'.format(rmse_loss, mae_loss))

    if adv_plot:
        # Visualization of model in production with fancy error bars
        stderr = np.std(np.abs(np.subtract(plot_y, preds)))

        group1 = np.concatenate([np.linspace(0, time_end - time_start - 1, time_end - time_start).reshape(-1, 1),
                                 preds[0:time_end - time_start]], axis=1)
        group2 = np.concatenate([np.linspace(0, time_end - time_start - 1, time_end - time_start).reshape(-1, 1),
                                 preds[0:time_end - time_start] + stderr], axis=1)
        group3 = np.concatenate([np.linspace(0, time_end - time_start - 1, time_end - time_start).reshape(-1, 1),
                                 preds[0:time_end - time_start] - stderr], axis=1)

        group = np.concatenate([group1, group2, group3])

        df = pd.DataFrame(group, columns=['time', 'predictions'])

        sns.lineplot(x='time', y='predictions', data=df)
        plt.plot(plot_y[time_start:time_end], label='Actual')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=0, frameon=False)

        plt.show()

    else:
        # Visualization of model in production without fancy error bars
        plt.plot(preds[0:time_end - time_start], label='Predicted')
        plt.plot(plot_y[0:time_end - time_start], label='Actual')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=0, frameon=False)

        if savefig:
            plt.savefig('figure.eps', format='eps', dpi=1000)

        plt.show()

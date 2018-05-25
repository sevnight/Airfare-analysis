# In[0] load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import sys

# load passenger data set and save to DataFrame
dateparse = lambda dates: pd.datetime.strptime(dates,"%Y-%m-%d")
df = pd.read_csv('airdata.csv', header=0, index_col='Date', parse_dates=[0],date_parser=dateparse)

df.sort_index(inplace=True)
print(df.head())
 
# create Series object
y = df['Price']

# split into training and test sets
y = df['Price']
y_train = y[:'2018-07-01']
y_test = y['2018-07-01':]

# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 4)
 
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
 
# generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None
 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = sm.tsa.statespace.SARIMAX(y_train,
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True)
            res = tmp_mdl.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))

# define SARIMAX model and fit it to the data
mdl = sm.tsa.statespace.SARIMAX(y_train,
                                order=(1, 1, 0),
                                seasonal_order=(1, 2, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
res = mdl.fit()

# print statistics
print(res.aic)
print(res.summary())


from scipy.stats import chi2
chi = chi2.isf(q=0.05, df=116)
chi

res.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.show()

# fit model to data
res = sm.tsa.statespace.SARIMAX(y_train,
                                order=(1, 1, 0),
                                seasonal_order=(1, 2, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()
 
# in-sample-prediction and confidence bounds
pred = res.get_prediction(start=pd.to_datetime('2018-08-30'), 
                          end=pd.to_datetime('2018-11-17'),
                          dynamic=True)
pred_ci = pred.conf_int()
 
# plot in-sample-prediction
ax = y['2018-07-01':].plot(label='Observed',color='#006699')
pred.predicted_mean.plot(ax=ax, label='One-step Ahead Prediction', alpha=.7, color='#ff0066')
 
# draw confidence bound (gray)
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], color='#ff0066', alpha=.25)
 
# style the plot
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2018-08-30'), y.index[-1], alpha=.15, zorder=-1, color='grey')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend(loc='upper left')
plt.show()

y_hat = pred.predicted_mean
y_true = y['2018-08-30':]
 
# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))

# predict out of sample and find confidence bounds
pred_out = res.get_prediction(start=pd.to_datetime('2018-08-30'), 
                              end=pd.to_datetime('2018-11-17'), 
                              dynamic=False, full_results=True)
pred_out_ci = pred_out.conf_int()
 
# plot time series and out of sample prediction
ax = y['2018-07-01':].plot(label='Observed', color='#006699')
pred_out.predicted_mean.plot(ax=ax, label='Out-of-Sample Forecast', color='#ff0066')
ax.fill_between(pred_out_ci.index,
                pred_out_ci.iloc[:, 0],
                pred_out_ci.iloc[:, 1], color='#ff0066', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2018-08-30'), y.index[-1], alpha=.15, zorder=-1, color='grey')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend()
#plt.savefig('./img/out_of_sample_pred.png')
plt.show()

# extract the predicted and true values of our time series
y_hat = pred_out.predicted_mean
y_true = y['2018-08-30':]
 
# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))

plt.plot(y_true, label='Observed', color='#006699')
plt.plot(y_hat, label='Out-of-Sample Forecast', color='#ff0066')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend(loc='upper left')
#plt.savefig('./img/out_of_sample_comparison.png')
plt.show()

# build model and fit
res = sm.tsa.statespace.SARIMAX(y,
                                order=(2, 1, 3),
                                seasonal_order=(1, 2, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()
 
# get forecast 120 steps ahead in future
pred_uc = res.get_forecast(steps=120)
 
# get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
 
# plot time series and long-term forecast
ax = y.plot(label='Observed', figsize=(16, 8), color='#006699')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='#ff0066')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='#ff0066', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend(loc='upper left')
plt.show()


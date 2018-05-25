
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

# Функция загрузки файла
def loadFile(filename, dateCol):
    dateparse = lambda dates: pd.datetime.strptime(dates,"%d.%m.%y")
    data = pd.read_csv(filename,index_col=dateCol, parse_dates=[0],date_parser=dateparse)
    return data

# In[2]
# airfare.csv содержит цены на билет по направлению Москва - Санкт-Питербург
# airfare2.csv содержит цены на билет по направлению Москва - Симферополь
data=loadFile('airfare.csv',"Date")
ts=data['Price']

ts.head(10)


# In[8]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling mean')
    std = plt.plot(rolstd, color='black', label = 'Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    if dftest[0]> dftest[4]['5%']: 
        print('есть единичные корни, ряд не стационарен')
    else:
        print('единичных корней нет, ряд стационарен')


plt.plot(ts)
test_stationarity(ts)


# In[9]:


# Берем натуральный логарифм
def remove_log(ts):
    return np.log(ts)

def rolling_mean(ts):
    return ts.rolling(12).mean()

ts_log = remove_log(ts)
moving_avg = rolling_mean(ts_log)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log.dropna(inplace=True)
test_stationarity(ts_log)


# In[10]:


def remove_moving_avg(ts, moving_avg):
    return ts_log - moving_avg

ts_log_moving_avg_diff = remove_moving_avg(ts_log, moving_avg)
ts_log_moving_avg_diff.head(12)
plt.plot(ts_log_moving_avg_diff)
plt.show()

# In[6]:

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# In[21]:


ts_state = ts_log_moving_avg_diff
ts_state = ts_state.asfreq('D').fillna(method='pad')
ts_state.head()


# In[22]:


# Раскладываем модель на составляющие, анализируем остатки
from statsmodels.tsa.seasonal import seasonal_decompose
def split_series(ts_log):
    decomposition = seasonal_decompose(ts_log)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    return [trend,seasonal,residual]

def split_analysis(ts_log):
    splits = split_series(ts_log)
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(splits[0], label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(splits[1] ,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(splits[2], label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return splits
    
residuals = split_analysis(ts_state)[2]


# In[23]:


ts_log_decompose = residuals
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[24]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_correlation_funcs(ts_log_diff):
    plot_acf(ts_log_diff.values.squeeze(), lags=25)
    plot_pacf(ts_log_diff, lags=25)
    plt.show()
    
plot_correlation_funcs(ts_state)


# In[25]:


from statsmodels.tsa.arima_model import ARIMA

def find_ARIMA(ts, order):
    model = ARIMA(ts, order)  
    results = model.fit(disp=-1)  
    plt.plot(ts)
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results.fittedvalues-ts)**2))
    plt.show()
    return results

print('AR-модель')
results_AR = find_ARIMA(ts_state, (2,1,0))

print('MA-модель')
results_MA = find_ARIMA(ts_state, (0,1,4))

print('ARIMA-модель')
results_ARIMA = find_ARIMA(ts_state, (2,1,4))


# In[51]:


def make_prediction(ts, ts_log, results_ARIMA, moving_avg):
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff.head()
    future = results_ARIMA.predict('23.4.19','23.6.19' )
    future.head()
    pred = predictions_ARIMA_diff.append(future)
    
    predictions_ARIMA_diff_cumsum = pred + moving_avg #predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_diff_cumsum.head()
    
    predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=pred.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    predictions_ARIMA_log.head()
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    
    plt.plot(ts)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: %.4f'% np.sqrt(sum(((predictions_ARIMA-ts).fillna(0))**2)/len(predictions_ARIMA)))
    plt.show()

    predictionRange = predictions_ARIMA.loc['18.4.19':'22.4.19']
    std = predictionRange.std()
    print('std2: %.4f'% np.sqrt(std))

moving_avg = moving_avg.fillna(8)
make_prediction(ts, ts_state, results_ARIMA, moving_avg)



# In[1]
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import keras
import math
import sklearn
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Функция загрузки файла
def loadFile(filename, dateCol):
    dateparse = lambda dates: pd.datetime.strptime(dates,"%Y-%m-%d")
    data = pd.read_csv(filename,index_col=dateCol, parse_dates=[0],date_parser=dateparse)
    data = data.sort_index()
    return data

# Функция установки freq (не исп)
def appendFreq(ts,freq):
    print(ts.index)
    tsn = pd.Series(ts.values,ts.index)
    tsn = tsn.asfreq(freq)
    print(tsn.index)
    return tsn

# Функция для проверки ряда на стационарность методом Дикки-Фуллера
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
    plt.savefig('./img/arima/1_RollingMean&StandardDeviation.png')
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

# Взятие логарифма для приведения ряда к стационарному виду
def remove_log(ts):
    ts_log = np.log(ts)
    plt.plot(ts_log, color='red')
    plt.savefig('./img/arima/2_log.png')
    plt.show()
    return ts_log

# Удаление скользящего среднего для приведения к стационарности
def remove_moving_avg(ts):
    moving_avg = ts - ts.rolling(12).mean()
    plt.plot(moving_avg)
    plt.savefig('./img/arima/3_moving.png')
    plt.show()
    return moving_avg

# Экспоненциально взвешенное скользящее среднее для приведения к стационарности
def exp_wighted_avg(ts):
    expwighted_avg = ts.ewm(halflife=12).mean()
    #plt.plot(ts)
    plt.plot(expwighted_avg, color='red')
    plt.savefig('./img/arima/4_exp.png')
    plt.show()
    return expwighted_avg

# Разница между y(t) и y(t+1)) для приведения к стационарности
def  series_diff(ts):
    ts_diff = ts - ts.shift()
    plt.plot(ts_diff)
    plt.savefig('./img/arima/5_yt-yt+1.png')
    plt.show()
    return ts_diff

# Разложение модели на составляющие
from statsmodels.tsa.seasonal import seasonal_decompose
def split_series(ts_log):
    decomposition = seasonal_decompose(ts_log)    
    return [decomposition.trend,decomposition.seasonal,decomposition.resid]
# Анализ остатков
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

# Строим АКФ и ЧАКФ
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_correlation_funcs(ts_log_diff):
    plot_acf(ts_log_diff.values.squeeze(), lags=25)
    plt.savefig('./img/arima/7_1_acf.png')
    plot_pacf(ts_log_diff, lags=25)
    plt.savefig('./img/arima/7_2_pacf.png')
    plt.show()

# Построение ARIMA модели
from statsmodels.tsa.arima_model import ARIMA
def find_ARIMA(ts_log, order):
    model = ARIMA(ts_log, order)  
    results = model.fit(disp=-1)  
    plt.plot(ts_log)
    plt.plot(results.fittedvalues, color='red')
    rowToSum = (results.fittedvalues-ts_log)
    rowToSum.dropna(inplace=True)
    plt.title('RSS: %.4f'% sum(rowToSum**2))
    plt.savefig('./img/arima/8_arima.png')
    plt.show()
    return results

# Построение прогноза по модели
def make_prediction(ts, ts_log, results_ARIMA,moving_avg):
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    future = results_ARIMA.predict('2018-11-17', '2018-11-25')
    pred = predictions_ARIMA_diff.append(future)    
    predictions_ARIMA_diff_cumsum = pred.cumsum() #predictions_ARIMA_diff.cumsum()    
    predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=pred.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)    
    plt.plot(ts)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: %.4f'% np.sqrt(sum(((predictions_ARIMA-ts).fillna(0))**2)/len(predictions_ARIMA)))
    plt.show()
    predictionRange = predictions_ARIMA.loc['2018-11-17':'2018-11-25']
    std = predictionRange.std()
    print('std2: %.4f'% np.sqrt(std))

# In[2] 
data=loadFile('airdata.csv',"Date")
ts=data['Price']
ts.head(10)

# In[3]
# Проверка стационарности
test_stationarity(ts)

# Далее преведены методы приведения к стационарности
# In[4]
# 1)Берем натуральный логарифм
ts_log = remove_log(ts)
ts_log.dropna(inplace=True)
test_stationarity(ts_log)

# In[5]
# 2)Убираем скользящее среднее
ts_log_moving_avg_diff = remove_moving_avg(ts_log)
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

# In[6]
# 3)Экспоненциально взвешенное скользящее среднее
ts_log_moving_avg_exp_diff = exp_wighted_avg(ts_log_moving_avg_diff)
ts_log_moving_avg_exp_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_exp_diff)

# In[7]
# 4)Разница между y(t) и y(t+1))
ts_dif = series_diff(ts_log_moving_avg_exp_diff)
ts_dif.dropna(inplace=True)
test_stationarity(ts_dif)

# In[8]
# Раскладываем модель на составляющие, анализируем остатки
ts_dif = ts_dif.dropna()
ts_dif = ts_dif.asfreq('D').fillna(method='pad')
residuals = split_analysis(ts_dif)[2]
residuals = residuals.dropna()
test_stationarity(residuals)

# In[9]
# Строим АКФ и ЧАКФ
plot_correlation_funcs(ts_dif)

# In[10]
# Строим различные модели ARIMA, пытаемся подобрать правильную
print('AR-модель')
results_AR = find_ARIMA(ts_dif, (2,1,0))
print('MA-модель')
results_MA = find_ARIMA(ts_dif, (0,1,4))
print('ARIMA-модель')
results_ARIMA = find_ARIMA(ts_dif, (2,1,4))

# In[11]
# По модели ARIMA пытаемся сделать прогноз
make_prediction(ts,ts_dif,results_AR,ts_log_moving_avg_diff)
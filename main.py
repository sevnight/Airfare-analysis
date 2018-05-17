
# In[4]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# Функция загрузки файла
def loadFile(filename, dateCol):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    data = pd.read_csv(filename,index_col=dateCol, parse_dates=[0],date_parser=dateparse)
    return data

# Функция подготовки временного ряда. 
# Ubuntu лежит в основе самых популярных OS - Linux Mint, Steam, Ubuntu. Просуммируем их
def prepareData(data):
    data['Buntu']=data['Linux Mint']+data['Steam OS']+data['Ubuntu']
    ts = data['Buntu']
    return ts

data = loadFile('data.csv', 'Date')
ts= prepareData(data)
ts.head(10) #выведем первые 5 элементов ряда

# In[6]:

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

# In[8]:

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

# In[13]:

# Убираем стационарность (скользящее среднее)
def remove_moving_avg(ts, moving_avg):
    return ts_log - moving_avg

ts_log_moving_avg_diff = remove_moving_avg(ts_log, moving_avg)
ts_log_moving_avg_diff.head(12)
plt.plot(ts_log_moving_avg_diff)
plt.show()

# In[16]:

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

# In[17]:

# Убираем стационарность (экспоненциально взвешенное скользящее среднее)
def exp_wighted_avg(ts_log):
    return ts_log.ewm(halflife=12).mean()

expwighted_avg =exp_wighted_avg(ts_log)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
plt.show()

# In[18]:

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

# In[19]:

# Убираем стационарность (разница между y(t) и y(t+1))
def  series_diff(ts_log):
    return ts_log - ts_log.shift()

ts_log_diff = series_diff(ts_log)
plt.plot(ts_log_diff)
plt.show()

# In[20]:

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# In[21]:

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
    
residuals = split_analysis(ts_log)[2]

# In[22]:

# Рассматриваем ряд остатков, проверяем его на стационарность
ts_log_decompose = residuals
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

# In[24]:

# Строим АКФ и ЧАКФ
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_correlation_funcs(ts_log_diff):
    plot_acf(ts_log_diff.values.squeeze(), lags=25)
    plot_pacf(ts_log_diff, lags=25)
    plt.show()
    
plot_correlation_funcs(ts_log_diff)

# In[26]:

# Строим различные модели ARIMA, пытаемся подобрать правильную
from statsmodels.tsa.arima_model import ARIMA

def find_ARIMA(ts_log, ts_log_diff, order):
    model = ARIMA(ts_log, order)  
    results = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results.fittedvalues-ts_log_diff)**2))
    plt.show()
    return results

print('AR-модель')
results_AR = find_ARIMA(ts_log, ts_log_diff, (2,1,0))

print('MA-модель')
results_MA = find_ARIMA(ts_log, ts_log_diff, (0,1,2))

print('ARIMA-модель')
results_ARIMA = find_ARIMA(ts_log, ts_log_diff, (2,1,2))

# In[39]:

# По модели ARIMA пытаемся сделать прогноз
def make_prediction(ts, ts_log, results_ARIMA):
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff.head()
    future = results_ARIMA.predict('2017-09-01', '2018-01-01')
    future.head()
    pred = predictions_ARIMA_diff.append(future)
    
    predictions_ARIMA_diff_cumsum = pred.cumsum() #predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_diff_cumsum.head()
    
    predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=pred.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    predictions_ARIMA_log.head()
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    
    plt.plot(ts)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: %.4f'% np.sqrt(sum(((predictions_ARIMA-ts).fillna(0))**2)/len(predictions_ARIMA)))
    plt.show()

    predictionRange = predictions_ARIMA.loc['2017-10-01':'2018-01-01']
    std = predictionRange.std()
    print('std2: %.4f'% np.sqrt(std))

make_prediction(ts, ts_log, results_ARIMA)

# In[40]
# Начало работы с Нейронкой
import keras
#import sklearn
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# initialize figur and axes
fig, axes = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(16, 6))
fig.suptitle('XOR Problem', fontsize=24, fontweight='bold')
 
# classifiable plot
axes[0].plot([0,0,1], [0,1,0], 'o', color='grey')
axes[0].plot([1], [1], 'X', color='#ff0066')
axes[0].plot([0.25, 1.25], [1.25, 0.25], color='black')
axes[0].set_xlim((-0.25, 1.25))
axes[0].set_ylim((-0.25, 1.25))
 
# unclassifiable plot
axes[1].plot([0,1], [1,0], 'o', color='grey')
axes[1].plot([0, 1], [0, 1], 'X', color='#ff0066')
axes[1].set_xlim((-0.25, 1.25))
axes[1].set_ylim((-0.25, 1.25))
plt.savefig('./img/xor_problem.png')
plt.show()

# set seed
np.random.seed(7)
 
# import data set
#df = pd.read_csv('data.csv', sep=';', parse_dates=True, index_col=0)
data = loadFile('data.csv', 'Date')
df= prepareData(data)
data = df.values
 
# using keras often requires the data type float32
data = data.astype('float32')
 
# slice the data
#train = data[0:120, :]   # length 120
#test = data[120:, :]     # length 24
train = df[0:100 :]   # length 100
test = df[100: :]     # length 6

def prepare_data(data, lags=1):
    """
    Create lagged data from an input time series
    """
    X, y = [], []
    for row in range(len(data) - lags - 1):
        a = data[row:(row + lags)]
        X.append(a)
        y.append(data[row + lags])
    return np.array(X), np.array(y)
 
# prepare the data
lags = 1
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test     # due to naming convention

# plot the created data
plt.plot(y_test, label='Original Data | y or t+1', color='#006699')
plt.plot(X_test, label='Lagged Data | X or t', color='orange')
plt.legend(loc='upper left')
plt.title('One Period Lagged Data')
plt.show()

# In[41]
# Прогнозирование временных рядов с помощью многослойной сети персептрона
# create and fit Multilayer Perceptron model
mdl = Sequential()
mdl.add(Dense(3, input_dim=lags, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=400, batch_size=2, verbose=2)

import math
# estimate model performance
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))

# generate predictions for training
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)
 
# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[: :] = np.nan
train_predict_plot[lags: len(train_predict) + lags :] = train_predict[1]
 
# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[: :] = np.nan
test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1 :] = test_predict[1]
 
# plot baseline and predictions
plt.plot(data, label='Observed', color='#006699')
plt.plot(train_predict_plot, label='Prediction for Train Set', color='#006699', alpha=0.5)
plt.plot(test_predict_plot, label='Prediction for Test Set', color='#ff0066')
plt.legend(loc='best')
plt.title('Artificial Neural Network')
plt.show()

mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
plt.legend(loc='best')
plt.show()


# In[42]
# Многослойный перцептрон с окном
# reshape and lag shift the dataset
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
 
# plot the created data
plt.plot(y_train, label='Original Data | y or t+1', color='#006699')
plt.plot(X_train, label='Lagged Data', color='orange')
plt.legend(loc='best')
plt.title('Three Period Lagged Data')
plt.savefig('./img/ann3_training.png')
plt.show()

# create and fit Multilayer Perceptron model
mdl = Sequential()
mdl.add(Dense(4, input_dim=lags, activation='relu'))
mdl.add(Dense(8, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=400, batch_size=2, verbose=2)
 
# Estimate model performance
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))

# generate predictions for training
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)
 
# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[: :] = np.nan
train_predict_plot[lags: len(train_predict) + lags :] = train_predict[1]
 
# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[: :] = np.nan
test_predict_plot[len(train_predict)+(lags * 2)+1:len(data)-1 :] = test_predict
 
# plot observation and predictions
plt.plot(data, label='Observed', color='#006699');
plt.plot(train_predict_plot, label='Prediction for train', color='#006699', alpha=0.5);
plt.plot(test_predict_plot, label='Prediction for test', color='#ff0066');
plt.legend(loc='best')
plt.title('Multilayer Perceptron with Window')
plt.show()

mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
plt.legend(loc='upper left')
plt.show()


# In[43]
# Прогнозирование временных рядов с повторяющейся нейронной сетью LSTM
# fix random seed for reproducibility
np.random.seed(1)
 
# load the dataset
data = loadFile('data.csv', 'Date')
df= prepareData(data)
data = df.values
data = data.astype('float32')
 
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data[1])
 
# split into train and test sets
train = dataset[0:100, :]
test = dataset[100:, :]
 
# reshape into X=t and Y=t+1
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
 
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# create and fit the LSTM network
mdl = Sequential()
mdl.add(Dense(3, input_shape=(1, lags), activation='relu'))
mdl.add(LSTM(6, activation='relu'))
mdl.add(Dense(1, activation='relu'))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# make predictions
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)
 
# invert transformation
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])
 
# calculate root mean squared error
train_score = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
print('Train Score: {:.2f} RMSE'.format(train_score))
test_score = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print('Test Score: {:.2f} RMSE'.format(test_score))

# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lags:len(train_predict)+lags, :] = train_predict
 
# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (lags * 2)+1:len(data)-1, :] = test_predict
 
# plot observation and predictions
plt.plot(data, label='Observed', color='#006699');
plt.plot(train_predict_plot, label='Prediction for Train Set', color='#006699', alpha=0.5);
plt.plot(test_predict_plot, label='Prediction for Test Set', color='#ff0066');
plt.legend(loc='upper left')
plt.title('LSTM Recurrent Neural Net')
plt.show()

mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
plt.legend(loc='upper left');
plt.savefig('./img/lstm_close.png')
plt.show()


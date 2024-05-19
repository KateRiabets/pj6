import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# Завантаження даних з CSV файлу
file_path = "MSFT(2000-2023).csv"
data = pd.read_csv(file_path, usecols=['Date', 'Adj Close'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Агрегація денних даних у місячні середні значення
data_monthly = data['Adj Close'].resample('MS').mean()

# Візуалізація агрегованих даних
plt.figure(figsize=(10, 5))
plt.plot(data_monthly, label='Ціна закриття (щомісячно)')
plt.title('Щомісячна скоригована ціна закриття акцій MSFT')
plt.xlabel('Дата')
plt.ylabel('Скоригована ціна закриття')
plt.legend()
plt.show()

# Перевірка стаціонарності ряду за допомогою тесту Дікі-Фуллера
result = adfuller(data_monthly.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
if result[1] > 0.05:
    print("Часовий ряд не стаціонарний")
    # Диференціація даних для досягнення стаціонарності
    data_monthly_diff = data_monthly.diff().dropna()
    # Повторна перевірка стаціонарності після диференціації
    result_diff = adfuller(data_monthly_diff)
    print('Після диференціації:')
    print('ADF Statistic: %f' % result_diff[0])
    print('p-value: %f' % result_diff[1])

# Розклад часового ряду на трендову, сезонну і випадкову компоненти
decomposition = seasonal_decompose(data_monthly.dropna(), model='additive', period=12)
fig = decomposition.plot()
plt.show()

# Автоматичний підбір параметрів моделі SARIMA
model = auto_arima(data_monthly_diff, seasonal=True, m=12, stepwise=True,
                   suppress_warnings=True, D=1, start_p=0, max_p=2, start_q=0, max_q=2)
print(model.summary())

# Навчання моделі на даних, окрім останніх 12 місяців
train = data_monthly_diff.iloc[:-12]  # використання даних до останнього року
test = data_monthly_diff.iloc[-12:]  # останній рік даних
model.fit(train)

# Прогнозування на наступні 12 місяців
forecast_diff = model.predict(n_periods=12)
forecast_diff = pd.Series(forecast_diff, index=test.index)

# Зворотна диференціація для повернення до початкового масштабу
forecast = data_monthly.iloc[-12-1] + forecast_diff.cumsum()

# Візуалізація результату
plt.figure(figsize=(10, 5))
plt.plot(data_monthly, label='Фактичні дані')
plt.plot(forecast, label='Прогнозована ціна')
plt.title('Прогноз щомісячної ціни закриття акцій Microsoft')
plt.xlabel('Дата')
plt.ylabel('Скоригована ціна закриття')
plt.legend()
plt.show()

# Розрахунок середньоквадратичної помилки прогнозу
mse = mean_squared_error(data_monthly.iloc[-12:], forecast)
print('MSE: ', mse)

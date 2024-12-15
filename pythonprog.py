pip install arch

import  yfinance as yf #для загрузки финансовых данных.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import arch  #для статистического анализа и моделирования
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind

# Загрузка исторических данных о валютных курсах
data = yf.download("USDRUB=X EURRUB=X", start="2013-01-01", end="2023-04-13")
usd_rates = data['Close']['USDRUB=X'].dropna()
eur_rates = data['Close']['EURRUB=X'].dropna()

def simulate_exchange_rates(historical_rates, days, simulations, drifts, volatilities):
    dt = 1 / days
    simulated_rates = {}
    for currency, rates in historical_rates.items():
        drift = drifts[currency]
        volatility = volatilities[currency]
        daily_returns = np.exp((drift - 0.5 * volatility**2) * dt +
                               volatility * np.random.normal(0, np.sqrt(dt), (days, simulations)))
        simulation = rates.iloc[-1] * np.cumprod(daily_returns, axis=0)
        simulated_rates[currency] = simulation
    return simulated_rates

def portfolio_value(simulated_rates, portfolio):
    simulations = next(iter(simulated_rates.values())).shape[1]
    values = np.full(simulations, portfolio.get('RUB', 0), dtype=float)
    for currency, amount in portfolio.items():
        if currency != 'RUB':
            values += amount * simulated_rates[currency][-1, :]
    return values

def calculate_var(portfolio_values, confidence_level=0.95):
    sorted_values = np.sort(portfolio_values)
    index = int((1 - confidence_level) * len(sorted_values))
    return sorted_values[index]

def calculate_cvar(portfolio_values, confidence_level=0.95):
    var = calculate_var(portfolio_values, confidence_level)
    cvar = portfolio_values[portfolio_values <= var].mean()
    return cvar

def calculate_confidence_interval(simulated_data, confidence_level=0.95):
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100
    lower_bound = np.percentile(simulated_data, lower_percentile)
    upper_bound = np.percentile(simulated_data, upper_percentile)
    return lower_bound, upper_bound

def calculate_drift_volatility(rates, lookback_period=252):
    log_returns = np.log(rates / rates.shift(1)).dropna()
    rolling_mean = log_returns.rolling(window=lookback_period).mean()
    rolling_std = log_returns.rolling(window=lookback_period).std()
    drift = rolling_mean.iloc[-1]
    volatility = rolling_std.iloc[-1]
    return drift, volatility

historical_rates = {'USD': usd_rates, 'EUR': eur_rates}
days = 252
simulations = 1000000
# Расчет дрейфа и волатильности для Моделирования
# через calculate_drift_volatility
drifts = {}
volatilities = {}
for currency, rates in historical_rates.items():
    drifts[currency], volatilities[currency] = calculate_drift_volatility(rates)
'''
# Расчет дрейфа и волатильности для Моделирования via garch
usd_log_returns = np.log(usd_rates / usd_rates.shift(1)).dropna()
eur_log_returns = np.log(eur_rates / eur_rates.shift(1)).dropna()

usd_drift_garch, usd_volatility_garch = fit_garch_model(usd_log_returns)
eur_drift_garch, eur_volatility_garch = fit_garch_model(eur_log_returns)

# Обновление словарей дрейфов и волатильности для использования в симуляции Монте-Карло
drifts['USD'] = usd_drift_garch
drifts['EUR'] = eur_drift_garch
volatilities['USD'] = usd_volatility_garch
volatilities['EUR'] = eur_volatility_garch
'''

# Моделирование курсов валют
simulated_rates = simulate_exchange_rates(historical_rates, days, simulations, drifts, volatilities)

# Ввод данных пользователем
rub_amount = float(input("Введите количество рублей: "))
usd_amount = float(input("Введите количество долларов: "))
eur_amount = float(input("Введите количество евро: "))

# Получение данных пользователя
portfolio = {'RUB': rub_amount, 'USD': usd_amount, 'EUR': eur_amount}
# Расчет стоимости портфеля
portfolio_values = portfolio_value(simulated_rates, portfolio)

cvar = calculate_cvar(portfolio_values)
print(f"Максимальный риск портфеля (95% довер): {cvar}")

data_recent = yf.download("USDRUB=X EURRUB=X", start="2023-04-12", end="2024-04-13")['Close'].dropna()
data_recent_usd = data_recent['USDRUB=X'].dropna()
data_recent_eur = data_recent['EURRUB=X'].dropna()

real_data_last_year = data_recent_usd[-days:]
simulated_data_first_path = simulated_rates['USD'][:, 0]

ks_stat, p_value_ks = ks_2samp(real_data_last_year, simulated_data_first_path)
t_stat, p_value_t = ttest_ind(real_data_last_year, simulated_data_first_path, equal_var=False)

print(f"KS Statistic: {ks_stat}, P-value (KS-test): {p_value_ks}")
print(f"T-statistic: {t_stat}, P-value (T-test): {p_value_t}")

usd_confidence_interval = calculate_confidence_interval(simulated_rates['USD'][-1, :])
eur_confidence_interval = calculate_confidence_interval(simulated_rates['EUR'][-1, :])

print(f"Доверительный интервал 95% для курса доллара: {usd_confidence_interval}")
print(f"Доверительный интервал 95% для курса евро: {eur_confidence_interval}")

# Визуализация исторических и симулированных данных
plt.figure(figsize=(14, 7))
plt.plot(usd_rates.index, usd_rates,
         label='Реальные данные(2013-01-01 - 2023-04-12)')
plt.plot(data_recent_usd.index, data_recent_usd,
         label='Реальные данные (2023-04-13 - 2024-04-13)', color='green', alpha=0.7)
plt.plot(pd.date_range(start=usd_rates.index[-1], periods=days, freq='B'),
         simulated_rates['USD'][:, 0], label='Смоделированные данные', color='orange', alpha=0.7)
plt.legend()
plt.title('Реальные vs Смоделированные USD/RUB курсы')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.savefig('RUBUSD.png')
plt.show()
# Визуализация исторических и симулированных данных
plt.figure(figsize=(14, 7))
plt.plot(eur_rates.index, eur_rates,
         label='Реальные данные(2013-01-01 - 2023-04-12)')
plt.plot(data_recent_eur.index, data_recent_eur,
         label='Реальные данные (2023-04-13 - 2023-04-13)', color='green', alpha=0.7)
plt.plot(pd.date_range(start=eur_rates.index[-1], periods=days, freq='B'),
         simulated_rates['EUR'][:, 0], label='Смоделированные данные', color='orange', alpha=0.7)
plt.legend()
plt.title('Реальные vs Смоделированные EUR/RUB курсы')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.savefig('RUBEUR.png')
plt.show()

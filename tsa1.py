import pandas as pd
import numpy as np

data = pd.read_excel("aluminium.xlsx")

data['Date'] = pd.to_datetime(data['Date'])

data = data.set_index(data['Date'])

data = data.sort_index()

#df = data['Price']
# series = data['Price'].astype(float)   # ensure numeric
series = data['Price'].copy()

# price_series = data['Price'].astype(float)
# series = series.resample('D').interpolate(method='time')

# series = series.asfreq('D')
series = series.interpolate(method='time')

# print(df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(series, label = "Aluminium Price")
plt.title("Aluminium Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(series, model='multiplicative', period = 365, extrapolate_trend='freq')
result.plot()
plt.suptitle('Seasonal Decomposition of Aluminium Prices over Time Series')
plt.tight_layout()
plt.show()

values = data['Price'].astype(float).to_numpy()
# values = data.values

parts = int(len(values)/3)
part_1, part_2, part_3 = values[0:parts], values[parts:(parts*2)], values[(parts*2):(parts*3)]
mean_1, mean_2, mean_3 = part_1.mean(), part_2.mean(), part_3.mean()
var_1, var_2, var_3 = part_1.var(), part_2.var(), part_3.var()
print('mean1=%f, mean2=%f, mean2=%f' % (mean_1, mean_2, mean_3))
print('variance1=%f, variance2=%f, variance2=%f' % (var_1, var_2, var_3))

log_values = np.log(data['Price'])

parts = int(len(log_values)/3)
part_1, part_2, part_3 = log_values[0:parts], log_values[parts:(parts*2)], log_values[(parts*2):(parts*3)]
mean_1, mean_2, mean_3 = part_1.mean(), part_2.mean(), part_3.mean()
var_1, var_2, var_3 = part_1.var(), part_2.var(), part_3.var()
print('mean1=%f, mean2=%f, mean2=%f' % (mean_1, mean_2, mean_3))
print('variance1=%f, variance2=%f, variance2=%f' % (var_1, var_2, var_3))

diff1 = log_values.diff().dropna()
parts = int(len(log_values)/3)
part_1, part_2, part_3 = diff1[0:parts], diff1[parts:(parts*2)], diff1[(parts*2):(parts*3)]
mean_1, mean_2, mean_3 = part_1.mean(), part_2.mean(), part_3.mean()
var_1, var_2, var_3 = part_1.var(), part_2.var(), part_3.var()
print('mean1=%f, mean2=%f, mean2=%f' % (mean_1, mean_2, mean_3))
print('variance1=%f, variance2=%f, variance2=%f' % (var_1, var_2, var_3))

from statsmodels.tsa.stattools import adfuller
print(adfuller(diff1))

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

log_diff = np.log(data['Price']).diff().dropna()

plt.figure(figsize=(10,4))
plot_acf(log_diff, lags=60, ax=plt.gca()); plt.title('ACF of log-differenced')
plt.show()

plt.figure(figsize=(10,4))
plot_pacf(log_diff, lags=60, ax=plt.gca(), method='ywm'); plt.title('PACF of log-differenced')
plt.show()


# ARIMA
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(log_values, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

forecast_log = model_fit.forecast(steps=30)
forecast = np.exp(forecast_log)
print(forecast)

plt.figure(figsize=(12,5))
plt.plot(data['Price'], label="Original")
plt.plot(forecast.index, label="Forecast")
plt.legend()
plt.show()


series = data.set_index('Date')['Price'].sort_index()

if series.index.freq is None:
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
    series = series.reindex(full_idx)            
    series = series.interpolate(method='time')   
    series.index.name = 'Date'

print("Series index freq:", series.index.freq)

model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

n_forecast = 300
pred = model_fit.get_forecast(steps=n_forecast)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

start_date = series.index[-1] + pd.Timedelta(days=1)
future_index = pd.date_range(start=start_date, periods=n_forecast, freq='D')

pred_mean = pd.Series(pred_mean, index=future_index, name='predicted_mean')
pred_ci.index = future_index

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(series, label='Original')
plt.plot(pred_mean, label='Forecast', color='orange')
plt.fill_between(future_index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1],
                 color='orange', alpha=0.2)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Original and Forecast (ARIMA(1,1,1))')
plt.show()

# corrected_arima_forecast.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ---------- SETTINGS ----------
n_forecast = 30   # horizon
order = (1, 1, 1) # ARIMA order (p,d,q)
# ------------------------------

# assume `data` DataFrame exists with columns ['Date', 'Price']
# Example: data = pd.read_csv("your_prices.csv")

# 1) Prepare series, datetime index, sort
data['Date'] = pd.to_datetime(data['Date'])
series = data.set_index('Date')['Price'].sort_index()

# 2) Handle duplicate timestamps (aggregate duplicates by mean)
if not series.index.is_unique:
    print("Found duplicate timestamps — aggregating by mean.")
    series = series.groupby(series.index).mean()

# 3) If index has no daily freq, reindex to daily and interpolate small gaps
if pd.infer_freq(series.index) is None:
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
    series = series.reindex(full_idx)
    # interpolate (time) for small gaps, if you prefer forward-fill use .ffill()
    series = series.interpolate(method='time')
series.index.name = 'Date'

print("Series prepared: start =", series.index[0], "end =", series.index[-1], "len =", len(series))

# 4) Log-transform (ARIMA on logs)
if (series <= 0).any():
    raise ValueError("Series contains non-positive values; log transform not possible. Fix data first.")
log_series = np.log(series)

# 5) Fit ARIMA on log-series
model = ARIMA(log_series, order=order)
model_fit = model.fit()
print(model_fit.summary())

# 6) Forecast on log-scale (get mean and standard error)
pred = model_fit.get_forecast(steps=n_forecast)
mu_log = pred.predicted_mean           # pandas Series (may have RangeIndex)
se_log = pred.se_mean                  # standard error of predicted mean (numpy / pd.Series)
var_log = se_log ** 2                  # variance of the predicted mean

# 7) Build a proper future datetime index and align results
start_date = series.index[-1] + pd.Timedelta(days=1)
future_index = pd.date_range(start=start_date, periods=n_forecast, freq='D')

# ensure mu_log and se_log are Series indexed by future_index
mu_log = pd.Series(mu_log.values, index=future_index, name='mu_log')
se_log = pd.Series(se_log, index=future_index, name='se_log')
var_log = se_log ** 2

# 8) Bias-corrected forecast (expected value on original scale)
# E[Y] ≈ exp(mu + 0.5 * var)
forecast_expected = np.exp(mu_log + 0.5 * var_log)

# 9) Analytic 95% prediction interval on original scale (approx)
# lower = exp(mu - 1.96*se + 0.5*var), upper = exp(mu + 1.96*se + 0.5*var)
lower = np.exp(mu_log - 1.96 * se_log + 0.5 * var_log)
upper = np.exp(mu_log + 1.96 * se_log + 0.5 * var_log)

# 10) Simple prints / checks
print("Forecast start:", future_index[0], "end:", future_index[-1])
print("mu_log head:\n", mu_log.head())
print("forecast_expected head:\n", forecast_expected.head())

# 11) Plot actuals + forecast + CI
plt.figure(figsize=(12,5))
plt.plot(series.index, series.values, label='Actual', linewidth=1.4)
plt.plot(forecast_expected.index, forecast_expected.values, label='Forecast (bias-corrected)', color='orange', linewidth=1.6)
plt.fill_between(future_index, lower, upper, color='orange', alpha=0.25, label='95% CI (approx)')
plt.axvline(series.index[-1], color='gray', linestyle='--', linewidth=0.8)  # last observed marker

plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Original and Forecast (ARIMA{order} on log scale, horizon={n_forecast})')
plt.xlim(series.index[0], future_index[-1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(series, 
                order=(1,1,1),
                seasonal_order=(1,1,1,7))   # 7-day seasonality (you may change)
model_fit = model.fit()
print(model_fit.summary())

n_forecast = 300
pred = model_fit.get_forecast(steps=n_forecast)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# Build future index
start_date = series.index[-1] + pd.Timedelta(days=1)
future_index = pd.date_range(start=start_date, periods=n_forecast, freq='D')

pred_mean.index = future_index
pred_ci.index = future_index

plt.figure(figsize=(12,5))
plt.plot(series, label='Original')
plt.plot(pred_mean, label='Forecast', color='orange')
plt.fill_between(future_index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1],
                 color='orange', alpha=0.2)
plt.legend()
plt.show()

# assuming model_fit is your SARIMAX/SARIMA fitted object
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

resid = model_fit.resid.dropna()

sm.graphics.tsa.plot_acf(resid, lags=40); plt.show()
sm.graphics.tsa.plot_acf(resid**2, lags=40); plt.show()
print("Ljung-Box:", acorr_ljungbox(resid, lags=[10,20], return_df=True))
print("Jarque-Bera:", sm.stats.jarque_bera(resid))
print("ARCH LM p-value:", het_arch(resid)[1])

df = series.to_frame(name="y")
df['jump'] = (df.index >= "2024-01-01").astype(int)

def fourier_series(index, period, K):
    t = np.arange(len(index))
    data = {}
    for k in range(1, K+1):
        data[f'sin_{k}'] = np.sin(2*np.pi*k*t/period)
        data[f'cos_{k}'] = np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(data, index=index)

fourier = fourier_series(df.index, period=20, K=3)

exog = pd.concat([df['jump'], fourier], axis=1)

model = SARIMAX(
    np.log(df['y']),
    exog=exog,
    order=(2,1,2),
    seasonal_order=(1,1,1,7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

# third time

model = SARIMAX(np.log(df['y']),
                # exog=exog,
                order=(2,1,2)).fit()
                # seasonal_order=(1,1,1,7),
                # enforce_stationarity=False,
                # enforce_invertibility=False).fit()

resid = model.resid.dropna()

sm.graphics.tsa.plot_acf(resid, lags=40); plt.show()
sm.graphics.tsa.plot_acf(resid**2, lags=40); plt.show()
print("Ljung-Box:", acorr_ljungbox(resid, lags=[10,20], return_df=True))
print("Jarque-Bera:", sm.stats.jarque_bera(resid))
print("ARCH LM p-value:", het_arch(resid)[1])

print(model.summary())
print(df.index.equals(exog.index))

from arch import arch_model

resid = model.resid.dropna()

garch = arch_model(
    resid,
    vol='Garch',
    p=1,
    q=1
).fit()

print(garch.summary())
sarima_fc = model.get_forecast(steps=60)
mu = sarima_fc.predicted_mean

# GARCH variance forecast
garch_fc = garch.forecast(horizon=60)
var = garch_fc.variance.values[-1]

# Bias-corrected back-transform
final_pred = np.exp(mu + 0.5*var)

# Now Garch is been included for volatility related issues


# # New 1

# ---- settings ----
h = 60  # forecast horizon days
y = df['y']          # original series (values on original scale)
log_y = np.log(y)    # you fitted ARIMA on log(y)

# 1) ensure index is datetime
df.index = pd.to_datetime(df.index)

# 2) try to get a freq; if None, set to daily (adjust if your data is weekly/monthly)
freq = pd.infer_freq(df.index)
if freq is None:
    # if you know frequency is daily, set 'D'; for weekly use 'W', monthly 'MS', etc.
    freq = 'D'
    # alternative: use median diff to define freq
    # freq = pd.Timedelta(pd.Series(df.index).diff().median()).components.days and map to 'D' etc
print("Using freq =", freq)

# 3) produce arima mean forecast (on log-scale)
arima_fc = model.get_forecast(steps=h)
mu = arima_fc.predicted_mean  # may have integer index

# 4) create proper forecast index starting after last observation
last_date = df.index[-1]
forecast_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                               periods=h, freq=freq)

# 5) ensure mu has the correct index
mu = pd.Series(mu.values, index=forecast_index)

# 6) produce GARCH variance forecast (elementwise)
garch_fc = garch.forecast(horizon=h)
# garch_fc.variance returns array shape (n_periods, horizon). take last row
var = garch_fc.variance.values[-1]              # numpy array length h
var = pd.Series(var, index=forecast_index)      # align with forecast_index

# 7) bias-correct and back-transform to original scale
final_pred = np.exp(mu + 0.5 * var)             # elementwise

# 8) plot actual + forecast (with simple shaded interval using var)
plt.figure(figsize=(12,5))
plt.plot(df.index, df['y'], label='Actual')

plt.plot(forecast_index, final_pred, color='red', label='Forecast')
plt.xlim(df.index[0], forecast_index[-1])
plt.legend()
plt.title('Actual vs Forecast (ARIMA mean + GARCH variance)')
plt.show()

# 1) confirm indexes
print("last actual date:", df.index[-1])
print("forecast start:", mu.index[0], "forecast end:", mu.index[-1])
print("mu.index.equals(var.index):", mu.index.equals(var.index))
print("mu.index.equals(final_pred.index):", mu.index.equals(final_pred.index))

# 2) quick peek at mu & var
print(mu.head())
print(var.head())

N = 30
train = df['y'][:-N]
test  = df['y'][-N:]

arima_train = ARIMA(np.log(train), order=(2,1,2)).fit()
resid_train = arima_train.resid.dropna()
garch_train = arch_model(resid_train, vol='Garch', p=1, q=1).fit(disp='off')

h = N
arima_fc = arima_train.get_forecast(steps=h)
mu_t = arima_fc.predicted_mean

garch_fc = garch_train.forecast(horizon=h)
var_t = garch_fc.variance.values[-1]

mu_t.index = test.index
var_t = pd.Series(var_t, index=test.index)

pred_t = np.exp(mu_t + 0.5*var_t)

# # Performance results
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------- ensure test and pred_t align -------
# `test` is your actual series (index = dates)
# `pred_t` should be a pandas Series indexed by the same dates (we built it earlier)
print("test.index equals pred_t.index ?", test.index.equals(pred_t.index))
# If False, align like this:
if not test.index.equals(pred_t.index):
    pred_t = pd.Series(pred_t.values, index=test.index)  # force same index

# ------- RMSE / MAE / MAPE -------
mse = mean_squared_error(test.values, pred_t.values)   # plain MSE
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test.values, pred_t.values)
mape = (np.abs((test.values - pred_t.values) / test.values)).mean() * 100

print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# ------- Baseline (naive last-value) -------
naive = np.repeat(train.iloc[-1], len(test))
mse_naive = mean_squared_error(test.values, naive)
rmse_naive = np.sqrt(mse_naive)
print(f"Naive RMSE: {rmse_naive:.4f}")

# ------- Coverage of analytic 95% interval -------
# if you built lower_t and upper_t earlier (from mu_t and var_t), compute coverage:
lower_t = np.exp(mu_t - 1.96 * np.sqrt(var_t) + 0.5 * var_t)
upper_t = np.exp(mu_t + 1.96 * np.sqrt(var_t) + 0.5 * var_t)
try:
    inside = ((test >= lower_t) & (test <= upper_t)).sum()
    coverage = inside / len(test)
    print(f"Analytic 95% interval coverage: {coverage:.2%}")
except NameError:
    print("lower_t/upper_t not available — compute them from mu_t and var_t as shown earlier.")

# ------- Quick diagnostic plot -------
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(train.index[-200:], train[-200:], label='Train (last 200)')
plt.plot(test.index, test, label='Test', color='blue')
plt.plot(pred_t.index, pred_t, label='Forecast', color='red')
plt.fill_between(pred_t.index, lower_t, upper_t, color='orange', alpha=0.2)  # if available
plt.legend()
plt.show()

# # Seaborn Graph

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Apply seaborn theme
# sns.set_theme(style="whitegrid")

# # ---------- Build Forecast DataFrame ----------
# forecast_index = mu.index
# forecast_df = pd.DataFrame({
#     'mu_log': mu,
#     'var_log': var,
#     'pred': final_pred
# }, index=forecast_index)

# # 95% CI on original scale (bias corrected)
# forecast_df['lower'] = np.exp(mu - 1.96 * np.sqrt(var) + 0.5 * var)
# forecast_df['upper'] = np.exp(mu + 1.96 * np.sqrt(var) + 0.5 * var)

# # ---------------------------------------------
# # Select last 120 days of actuals for clean view
# # ---------------------------------------------
# start_plot = df.index[-120] if len(df) > 120 else df.index[0]
# actual_recent = df.loc[start_plot:, 'y']

# # ---------- Seaborn Plot ----------
# plt.figure(figsize=(14,6))

# # Actual prices
# sns.lineplot(x=actual_recent.index, y=actual_recent.values, label="Actual", linewidth=1.5)

# # Forecasted mean
# sns.lineplot(x=forecast_df.index, y=forecast_df['pred'], color='red',
#              marker='o', markersize=5, label='Forecast')

# # Shaded CI
# plt.fill_between(forecast_df.index,
#                  forecast_df['lower'],
#                  forecast_df['upper'],
#                  color='orange', alpha=0.25, label='95% CI')

# # Titles & labels
# plt.title("Forecasted Prices (Seaborn Style) — ARIMA + GARCH", fontsize=15)
# plt.xlabel("Date", fontsize=12)
# plt.ylabel("Price", fontsize=12)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.legend()
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(style="whitegrid")

# # choose context window for actuals (last 120 days)
# start_plot = df.index[-120] if len(df) > 120 else df.index[0]
# actual_recent = df.loc[start_plot:, 'y']

# plt.figure(figsize=(14,6))
# sns.lineplot(x=actual_recent.index, y=actual_recent.values, label='Actual', linewidth=1.5)
# sns.lineplot(x=forecast_df.index, y=forecast_df['pred'], color='red',
#              marker='o', markersize=6, label='Forecast')
# plt.fill_between(forecast_df.index, forecast_df['lower'], forecast_df['upper'],
#                  color='orange', alpha=0.25, label='95% CI')

# plt.xticks(rotation=45)
# plt.title("Forecast (ARIMA+GARCH) — day-by-day markers")
# plt.xlabel("Date"); plt.ylabel("Price")
# plt.legend(); plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import mplcursors

# sns.set_theme(style="whitegrid")
# fig, ax = plt.subplots(figsize=(14,6))

# # plot actuals
# ax.plot(actual_recent.index, actual_recent.values, label='Actual', linewidth=1.5)

# # forecast line with markers
# line_forecast, = ax.plot(forecast_df.index, forecast_df['pred'], '-o', color='red', label='Forecast')

# # CI
# ax.fill_between(forecast_df.index, forecast_df['lower'], forecast_df['upper'], color='orange', alpha=0.25)

# ax.set_title("Hover over forecast points to see date & value")
# ax.tick_params(axis='x', rotation=45)
# ax.legend()
# plt.tight_layout()

# # enable hover tooltips
# crs = mplcursors.cursor(line_forecast, hover=True)
# @crs.connect("add")
# def on_add(sel):
#     x, y = sel.target
#     # convert numeric to datetime if needed
#     import matplotlib.dates as mdates
#     dt = mdates.num2date(x)
#     sel.annotation.set_text(f"{pd.to_datetime(dt).strftime('%Y-%m-%d')}\nValue: {y:.2f}")
#     sel.annotation.get_bbox_patch().set(alpha=0.9)

# plt.show()

# import matplotlib.pyplot as plt
# import mplcursors

# sns.set_theme(style="whitegrid")
# fig, ax = plt.subplots(figsize=(14,6))

# # plot actuals
# ax.plot(actual_recent.index, actual_recent.values, label='Actual', linewidth=1.5)

# # forecast line with markers
# line_forecast, = ax.plot(forecast_df.index, forecast_df['pred'], '-o', color='red', label='Forecast')

# # CI
# ax.fill_between(forecast_df.index, forecast_df['lower'], forecast_df['upper'], color='orange', alpha=0.25)

# ax.set_title("Hover over forecast points to see date & value")
# ax.tick_params(axis='x', rotation=45)
# ax.legend()
# plt.tight_layout()

# # enable hover tooltips
# crs = mplcursors.cursor(line_forecast, hover=True)
# @crs.connect("add")
# def on_add(sel):
#     x, y = sel.target
#     # convert numeric to datetime if needed
#     import matplotlib.dates as mdates
#     dt = mdates.num2date(x)
#     sel.annotation.set_text(f"{pd.to_datetime(dt).strftime('%Y-%m-%d')}\nValue: {y:.2f}")
#     sel.annotation.get_bbox_patch().set(alpha=0.9)

# plt.show()

# make_plotly_forecast.py
import pandas as pd
import plotly.graph_objects as go

# load files (created earlier)
f = pd.read_csv("forecast_daily_table.csv", index_col=0, parse_dates=True)  # must have columns: pred, lower, upper
try:
    h = pd.read_csv("historical.csv", index_col=0, parse_dates=True)         # optional, column 'y'
except Exception:
    h = None

fig = go.Figure()

# historical (last 120 days for context)
if h is not None:
    start = h.index[-120] if len(h) > 120 else h.index[0]
    hist = h.loc[start:]
    fig.add_trace(go.Scatter(x=hist.index, y=hist['y'], mode='lines', name='Actual'))

# forecast line (pred)
fig.add_trace(go.Scatter(
    x=f.index, y=f['pred'],
    mode='lines+markers',
    name='Forecast',
    hovertemplate='Date: %{x|%Y-%m-%d}<br>Pred: %{y:.4f}<extra></extra>'
))

# CI as filled polygon
fig.add_trace(go.Scatter(
    x=list(f.index) + list(f.index[::-1]),
    y=list(f['upper']) + list(f['lower'][::-1]),
    fill='toself',
    fillcolor='rgba(255,165,0,0.2)',
    line=dict(color='rgba(255,165,0,0)'),
    hoverinfo='skip',
    showlegend=True,
    name='95% CI'
))

fig.update_layout(title="Forecast (ARIMA + GARCH)",
                  xaxis_title="Date", yaxis_title="Price",
                  template='plotly_white', hovermode='x unified')

# write standalone HTML you can open locally
fig.write_html("forecast_plot.html", include_plotlyjs='cdn')
print("Saved forecast_plot.html — open this file in your browser.")

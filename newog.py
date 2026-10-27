import pandas as pd
import numpy as np

data = pd.read_excel("realprice.xlsx")

data['Date'] = pd.to_datetime(data['Date'])

data = data.set_index(data['Date'])

data = data.sort_index()

#df = data['Price']
# series = data['Price'].astype(float)   # ensure numeric
series = data['Price'].copy()

# price_series = data['Price'].astype(float)
# series = series.resample('D').interpolate(method='time')

series = series.asfreq('D')
series = series.interpolate(method='time')

#print(df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(series, label = "Aluminium Price")
plt.title("Aluminium Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

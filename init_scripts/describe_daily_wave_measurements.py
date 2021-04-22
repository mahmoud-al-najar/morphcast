import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_all_days = pd.read_csv('waves_dates_and_counts.csv')
plt.plot(df_all_days.date.astype(np.datetime64), df_all_days.n)
plt.show()

df_days_with_8 = df_all_days[df_all_days.n == 8]
df_days_with_24 = df_all_days[df_all_days.n == 24]
df_days_with_48 = df_all_days[df_all_days.n == 48]

print(len(df_all_days) - (len(df_days_with_8) + len(df_days_with_24) + len(df_days_with_48)))

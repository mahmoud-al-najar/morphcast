import pandas as pd
df = pd.read_csv('init_scripts/raw_master_waves.csv')
start = '1996-11-01'
end = '2021-04-20'
date_range = pd.date_range(start=start, end=end)

dates = []
counts = []
for d in date_range.values:
    x = pd.to_datetime(d)
    year = x.year
    month = x.month
    day = x.day
    print(f'{year}-{month:02d}-{day:02d}', ', ',
          len(df[(df['date'] >= f'{year}-{month:02d}-{day:02d}') & (df['date'] < f'{year}-{month:02d}-{day + 1:02d}')]))
    n = len(df[(df['date'] >= f'{year}-{month:02d}-{day:02d}') & (df['date'] < f'{year}-{month:02d}-{day + 1:02d}')])
    dates.append(x)
    counts.append(n)
df = pd.DataFrame({'date': dates, 'n': counts})
df.to_csv('init_scripts/waves_dates_and_counts.csv', index=False)

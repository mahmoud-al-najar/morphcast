import os
import json
import copy
import netCDF4
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import matplotlib.dates as mdates
from scipy import interpolate
from calendar import monthrange
import calendar
from utilities.common import datetime_to_timestamp, get_datetime_from_ymd_string, timestamp_to_datetime


def read_measurement_metadata(file_path):
    df = pd.read_csv(file_path)
    with open(file_path.replace('.csv', '.json')) as f:
        json_dict = json.load(f)
    time_start = json_dict['time_start']
    time_end = json_dict['time_end']
    lat_min = json_dict['lat_min']
    lat_max = json_dict['lat_max']
    lng_min = json_dict['lng_min']
    lng_max = json_dict['lng_max']
    return df, time_start, time_end, lat_min, lat_max, lng_min, lng_max


# start_time = 852213960.000002  # Jan '97 # 846828000.0  # Nov '96
# end_time = 1136062800.0  # Dec '05

target_start_ymd = '19961201'
start_time = datetime_to_timestamp(get_datetime_from_ymd_string(target_start_ymd))

target_end_ymd = '20210201'
end_time = datetime_to_timestamp(get_datetime_from_ymd_string(target_end_ymd))

print(timestamp_to_datetime(start_time), timestamp_to_datetime(end_time))

all_hours = np.arange(start_time, end_time, 3600)
all_dts = [timestamp_to_datetime(ts) for ts in all_hours]
full_range = end_time - start_time

years = np.arange(1997, 2020+1, 1)
months = np.arange(1, 13, 1)

file_ids = []
file_ids.append('199611')
for year in years:
    for month in months:
        file_ids.append(f'{year}{str(month).zfill(2)}')
file_ids.append('202101')
file_ids.append('202102')
file_ids.append('202103')
file_ids.append('202104')

data_dir = '/media/mn/WD4TB/topo/waves/waverider-17m/processed_data/'

raw_df = None
missing_ids = []
for f_id in file_ids:
    file_path = os.path.join(data_dir, f'{f_id}.csv')
    if os.path.isfile(file_path):
        fdf, time_start, time_end, lat_min, lat_max, lng_min, lng_max = \
            read_measurement_metadata(file_path)
        if raw_df is None:
            raw_df = fdf
        else:
            raw_df = raw_df.append(fdf, ignore_index=True)
            raw_df = raw_df.sort_values(by='t')
    else:
        missing_ids.append(f_id)
        print('Unavailable: ', f_id)

print(raw_df.head())
raw_t = raw_df.t
raw_tp = raw_df.tp
raw_hs = raw_df.hs
raw_direction = raw_df.dir

t_new = all_hours

f_tp = interpolate.interp1d(raw_t, raw_tp)
tp_new = f_tp(t_new)

f_hs = interpolate.interp1d(raw_t, raw_hs)
hs_new = f_hs(t_new)

f_direction = interpolate.interp1d(raw_t, raw_direction)
direction_new = f_direction(t_new)

raw_df['date'] = raw_df.t.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
raw_df.date = raw_df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

df = pd.DataFrame({'t': t_new, 'tp': tp_new, 'hs': hs_new, 'dir': direction_new})
df['date'] = df.t.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
df.date = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

print(df.head())
print(len(df))
df.to_csv('interpolated_master_waves.csv', index=False)
raw_df.to_csv('raw_master_waves.csv', index=False)
exit()

for year in years:
    for month in months:
        days = monthrange(year, month)[1]

        dt_start = datetime(year=year, month=month, day=1, hour=0, minute=0)
        dt_end = datetime(year=year, month=month, day=days, hour=23, minute=59)
        in_range_df = df[
            df.t.gt(calendar.timegm(dt_start.timetuple())) & df.t.lt(calendar.timegm(dt_end.timetuple()))]
        in_range_raw_df = raw_df[
            raw_df.t.gt(calendar.timegm(dt_start.timetuple())) & raw_df.t.lt(calendar.timegm(dt_end.timetuple()))]

        plt.figure(figsize=(13, 7.5))
        plt.plot(in_range_df.date, in_range_df.tp + 1)
        plt.plot(in_range_raw_df.date, in_range_raw_df.tp)
        plt.legend(['interpolated', 'raw'])
        plt.suptitle(f'{year}-{month}')
        plt.xticks(rotation=-45)
        plt.show()

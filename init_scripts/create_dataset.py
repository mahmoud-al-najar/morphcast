import os
import copy
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import groupby
from utilities.wrappers import Topo
from utilities.common import get_datetime_from_ymd_string, datetime_to_timestamp


s_start_ymd = '19961104'
dt_start_ymd = get_datetime_from_ymd_string(s_start_ymd)
ts_start_ymd = datetime_to_timestamp(dt_start_ymd)

s_end_ymd = '20210330'
dt_end_ymd = get_datetime_from_ymd_string(s_end_ymd)
ts_end_ymd = datetime_to_timestamp(dt_end_ymd)

print(s_start_ymd, s_end_ymd)
print(dt_start_ymd, dt_end_ymd)
print(ts_start_ymd, ts_end_ymd)

target_dates = np.arange(dt_start_ymd, dt_end_ymd, np.timedelta64(1, 'M'),
                                 dtype='datetime64[M]')  # Y-M only
target_dates = target_dates.astype('datetime64[D]')  # Y-M-D
target_dates = target_dates.astype('datetime64[s]')  # add seconds
print(target_dates)

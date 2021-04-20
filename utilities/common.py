import numpy as np
import datetime as dt


def get_datetime_from_ymd_string(s):
    return np.datetime64(dt.datetime.strptime(s, "%Y%m%d"))


def datetime_to_timestamp(d):
    return (d - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def timestamp_to_datetime(ts):
    return np.datetime64(dt.datetime.utcfromtimestamp(ts))

import datetime as dt


def get_datetime_from_ymd_string(s):
    return dt.datetime.strptime(s, "%Y%m%d")

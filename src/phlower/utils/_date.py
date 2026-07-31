import datetime as dt


def now_datetime_string() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

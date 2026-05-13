import numpy as np
import pandas as pd

def safe_div(a, b):
    try:
        a = 0.0 if a is None or pd.isna(a) else float(a)
        b = 0.0 if b is None or pd.isna(b) else float(b)
        if b == 0:

            if a > 0:
                return 999999

            return 0

        return a / b

    except:
        return None

def is_invalid(v):

    return (
        v is None or
        pd.isna(v) or
        v == float("inf") or
        v == float("-inf")
    )


def calc_current_ratio(kv):
    return safe_div(kv.get("Оборотные активы"), kv.get("Краткосрочные обязательства"))

def calc_leverage(kv):
    total_liab = (kv.get("Краткосрочные обязательства") or 0) + (kv.get("Долгосрочные обязательства") or 0)
    return safe_div(total_liab, kv.get("Собственный капитал"))

def compute_all_basic(kv):
    return {
        "current_ratio": calc_current_ratio(kv),
        "leverage": calc_leverage(kv)
    }
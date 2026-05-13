import pandas as pd
import re

def extract_number(value):
    """Преобразует строку или число в чистый float, учитывая минусы и точки."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    # Убираем пробелы (в том числе неразрывные)
    text = str(value).replace('\xa0', '').replace(' ', '')
    # Заменяем запятую на точку
    text = text.replace(',', '.')
    try:
        return float(text)
    except ValueError:
        # Оставляем только цифры, первую точку и первый минус
        cleaned = "".join([c for c in text if c.isdigit() or c in '.-'])
        try:
            return float(cleaned)
        except:
            return None

def normalize_dataframe(df):
    """
    Приводит все значения в DataFrame к числовому формату.
    Эту функцию ищет ваш app.py
    """
    # Создаем копию, чтобы не портить оригинал
    new_df = df.copy()

    # Проходим по всем колонкам и применяем extract_number
    for col in new_df.columns:
        new_df[col] = new_df[col].apply(extract_number)

    return new_df
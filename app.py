from flask import Flask, render_template, request, flash
import pandas as pd
import os
import math

from src.normalize import normalize_dataframe
from src.features import compute_all_basic
from src.models_classic import compute_all_classic
from src.normalize import extract_number

app = Flask(__name__)
app.secret_key = "secret_key"

UPLOAD_DIR = "data/uploads"
RESULTS_DIR = "data/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FIELDS_WITH_CODES = {
    "Оборотные активы":"1200",
    "Краткосрочные обязательства":"1500",
    "Долгосрочные обязательства":"1400",
    "Собственный капитал":"1300",
    "Баланс":"1600/1700",
    "Выручка":"2110",
    "Чистая прибыль":"2400",
    "Кредиторская задолженность":"1520",
    "Дебиторская задолженность":"1230",
    "Совокупные расходы":"2120+2210+2220",
    "Прибыль до налогообложения":"2300",
    "Наиболее ликвидные активы":"1240+1250",
    "Баланс за предыдущий год":"1600(прошлый год)"
}
FIELDS = list(FIELDS_WITH_CODES.keys())

def is_data_sufficient(kv):
    """Проверяет наличие минимально необходимых данных для анализа."""
    # Считаем, сколько полей из списка критических заполнены
    critical_fields = ["Баланс", "Чистая прибыль", "Выручка", "Собственный капитал"]
    count = sum(1 for f in critical_fields if kv.get(f) is not None)
    return count >= 3

def num(v):
    return extract_number(v)

def interpret_altman(z):

    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "нет данных"
    if z > 0:
        return "высокая вероятность банкротства"
    return "финансово устойчиво"

def interpret_taffler(z):
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "нет данных"
    if z < 0.3:
        return "высокая вероятность банкротства"
    return "финансово устойчиво"

def interpret_belikov(r):

    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "нет данных"
    if r < 0:
        return "высокая вероятность банкротства"
    if r < 1:
        return "зона риска"
    return "устойчивое состояние"

def interpret_zaitseva(res):

    if res is None:
        return "нет данных"

    if res["K_fact"] > res["K_norm"]:
        return "высокий риск банкротства"
    return "низкий риск банкротства"


def overall_conclusion(models_table):
    valid_models = [row for row in models_table if row["Интерпретация"] != "нет данных"]
    if len(valid_models) < 2:
        return "Недостаточно данных для формирования итогового заключения."
    
    high_risk_count = 0
    risk_count = 0
    stable_count = 0

    for row in models_table:
        s = str(row["Интерпретация"]).lower()
        if "высокая" in s or "высокий" in s:
            high_risk_count += 1
        if "банкрот" in s or "риск" in s:
            risk_count += 1
        if "устойчив" in s:
            stable_count += 1

    # Если хотя бы 2 модели показывают высокий риск — это уже критично
    if high_risk_count >= 2:
        return "Критическое состояние: высокая вероятность банкротства по большинству методик."

    # Если 3 или более модели (даже с низким риском) указывают на проблемы
    if risk_count >= 3:
        return "Неустойчивое финансовое состояние: выявлены признаки риска банкротства."

    if stable_count >= 3:
        return "Финансово устойчивое состояние."

    return "Пограничное состояние: результаты моделей противоречивы."

def build_advice(models_table, kv):
    advice = []

    # 1. Анализ прибыли и капитала (Критический уровень)
    profit = kv.get("Чистая прибыль")
    equity = kv.get("Собственный капитал")
    if profit is not None and profit < 0:
        advice.append("Фиксируется отрицательная чистая прибыль — рекомендуется пересмотреть структуру расходов и проанализировать порог рентабельности.")

    if equity is not None and equity < 0:
        advice.append("Критический риск: Собственный капитал отрицательный. Предприятие утратило финансовую независимость и находится в стадии технического банкротства.")

    # 2. Анализ ликвидности и оборачиваемости
    assets = kv.get("Баланс")
    current_assets = kv.get("Оборотные активы")
    short_liab = kv.get("Краткосрочные обязательства")
    revenue = kv.get("Выручка")

    if current_assets is not None and short_liab is not None:
        if current_assets < short_liab:
            advice.append("Дефицит ликвидности: оборотных активов недостаточно для покрытия краткосрочных обязательств. Риск потери платежеспособности.")

    if revenue is not None and assets not in [None, 0]:
        if revenue / assets < 0.5:
            advice.append("Низкая оборачиваемость активов — стоит повысить эффективность использования ресурсов и деловую активность.")

    # 3. Анализ динамики активов (Сравнение с прошлым годом)
    prev_balance = kv.get("Баланс за предыдущий год")
    if assets is not None and prev_balance is not None and prev_balance > 0:
        if assets < prev_balance:
            decline = ((prev_balance - assets) / prev_balance) * 100
            advice.append(f"Активы компании сократились на {decline:.1f}% по сравнению с предыдущим годом.")
            if decline > 20:
                advice.append("Наблюдается существенное сокращение активов — возможна деградация производственного потенциала.")

    # 4. Резюме на основе комплексной оценки моделей
    risk_count = 0
    stable_count = 0
    for row in models_table:
        s = str(row["Интерпретация"]).lower()
        if "банкрот" in s or "риск" in s:
            risk_count += 1
        if "устойчив" in s:
            stable_count += 1

    if risk_count >= 2:
        advice.append("Необходимо разработать стратегию финансового оздоровления, снизить долговую нагрузку и усилить контроль денежных потоков.")
    elif stable_count >= 3:
        advice.append("Компания демонстрирует устойчивость. Рекомендуется поддерживать текущий уровень ликвидности и инвестировать в развитие.")

    # 5. Если всё в порядке
    if not advice:
        advice.append("Существенных финансовых рисков по основным показателям не выявлено. Рекомендуется регулярный мониторинг.")

    return advice

def is_data_sufficient(kv):
    """Проверяет, достаточно ли данных для минимального анализа."""
    # Список критически важных полей
    required_fields = ["Баланс", "Чистая прибыль", "Выручка", "Собственный капитал"]

    filled_count = 0
    for field in required_fields:
        val = kv.get(field)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            filled_count += 1
    # Если заполнено меньше 3 ключевых полей, анализ нецелесообразен
    return filled_count >= 3


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        use_file = file and file.filename != ""

        manual_data = {field: request.form.get(field) for field in FIELDS}
        manual_filled = any(v not in [None, ""] for v in manual_data.values())
        if use_file and manual_filled:
            flash("Выберите только один способ ввода: либо файл, либо ручной ввод.")
            return render_template("index.html", fields=FIELDS, fields_with_codes = FIELDS_WITH_CODES)

        if not use_file and not manual_filled:
            flash("Загрузите файл или заполните таблицу вручную.")
            return render_template("index.html", fields=FIELDS, fields_with_codes = FIELDS_WITH_CODES)

         # --- ЛОГИКА ДЛЯ ФАЙЛА ---
        if use_file:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            else:
                flash("Поддерживаются только CSV и Excel файлы.")
                return render_template("index.html", fields=FIELDS, fields_with_codes = FIELDS_WITH_CODES)
            df = normalize_dataframe(df)
            rows = []

            for _, row in df.iterrows():
                kv = row.to_dict()
                # Для файла проверка может быть мягче, но основные расчеты всё равно пройдут
                compute_all_basic(kv)
                classic = compute_all_classic(kv)

                models_table = [
                    {"Модель": "Альтман", "Значение": classic["Altman_2factor"], "Интерпретация": interpret_altman(classic["Altman_2factor"])},
                    {"Модель": "Таффлер–Тишоу", "Значение": classic["Taffler_Tishaw"], "Интерпретация": interpret_taffler(classic["Taffler_Tishaw"])},
                    {"Модель": "Беликов–Давыдов", "Значение": classic["Belikov_Davydov"], "Интерпретация": interpret_belikov(classic["Belikov_Davydov"])},
                    {"Модель": "Зайцева", "Значение": classic["Zaitseva"]["K_fact"] if classic["Zaitseva"] else None, "Интерпретация": interpret_zaitseva(classic["Zaitseva"])}
                ]

                rows.append({
                    "company": kv.get("company", "Неизвестно"),
                    "features_table": [{"Показатель": col, "Значение": kv.get(col)} for col in FIELDS],
                    "models_table": models_table,
                    "advice": build_advice(models_table, kv),
                    "overall": overall_conclusion(models_table)
                })

            return render_template("results.html", rows=rows)

        # --- ЛОГИКА ДЛЯ РУЧНОГО ВВОДА (С ПРОВЕРКОЙ) ---
        kv = {field: num(manual_data.get(field)) for field in FIELDS}


        if not is_data_sufficient(kv):
            flash("Недостаточно данных! Заполните хотя бы 3 основных показателя: Баланс, Прибыль, Выручка или Капитал.")
            return render_template("index.html", fields=FIELDS, fields_with_codes = FIELDS_WITH_CODES)

        compute_all_basic(kv)
        classic = compute_all_classic(kv)

        models_table = [
            {"Модель": "Альтман", "Значение": classic["Altman_2factor"], "Интерпретация": interpret_altman(classic["Altman_2factor"])},
            {"Модель": "Таффлер–Тишоу", "Значение": classic["Taffler_Tishaw"], "Интерпретация": interpret_taffler(classic["Taffler_Tishaw"])},
            {"Модель": "Беликов–Давыдов", "Значение": classic["Belikov_Davydov"], "Интерпретация": interpret_belikov(classic["Belikov_Davydov"])},
            {"Модель": "Зайцева", "Значение": classic["Zaitseva"]["K_fact"] if classic["Zaitseva"] else None, "Интерпретация": interpret_zaitseva(classic["Zaitseva"])}
        ]

        return render_template("results.html", rows=[{
            "company": "Ручной ввод",
            "features_table": [{"Показатель": col, "Значение": kv.get(col)} for col in FIELDS],
            "models_table": models_table,
            "advice": build_advice(models_table, kv),
            "overall": overall_conclusion(models_table)
        }])

    return render_template("index.html", fields=FIELDS, fields_with_codes = FIELDS_WITH_CODES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
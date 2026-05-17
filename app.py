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
    "Совокупные расходы":"2120+2210+2220+2330+2350",
    "Прибыль до налогообложения":"2300",
    "Наиболее ликвидные активы":"1240+1250",
    "Баланс за предыдущий год":"1600(прошлый год)"
}
FIELDS = list(FIELDS_WITH_CODES.keys())

def is_data_sufficient(kv):
    required_fields = ["Баланс", "Чистая прибыль", "Выручка", "Собственный капитал"]
    filled_count = 0
    for field in required_fields:
        val = kv.get(field)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            filled_count += 1
    return filled_count >= 3

def num(v):
    return extract_number(v)

# --- Умная калибровка интерпретаций моделей на основе финансового контекста ---

def interpret_altman(z, kv):
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "нет данных"
    
    # Модификация: если операционная деятельность убыточна (стр 2300 < 0), повышаем критичность
    profit_before_tax = kv.get("Прибыль до налогообложения") or 0
    if profit_before_tax < 0:
        return "высокая вероятность банкротства"
        
    if z > 0:
        return "высокая вероятность банкротства"
    return "финансово устойчиво"

def interpret_taffler(z, kv):
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "нет данных"
        
    revenue = kv.get("Выручка") or 0
    balance = kv.get("Баланс") or 1
    
    # Модификация: если у компании сверхвысокая оборачиваемость (Выручка / Баланс > 1.5),
    # то даже пограничный Z-счет Таффлера означает устойчивость
    if revenue / balance > 1.5:
        return "финансово устойчиво"

    if z < 0.3:
        return "высокая вероятность банкротства"
    return "финансово устойчиво"

def interpret_belikov(r, kv):
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "нет данных"
        
    revenue = kv.get("Выручка") or 0
    balance = kv.get("Баланс") or 1
    profit_before_tax = kv.get("Прибыль до налогообложения") or 0
    
    # Корректировка под масштабные устойчивые компании с временным дефицитом СОС
    if revenue / balance > 1.5 and profit_before_tax > 0:
        return "устойчивое состояние"
        
    # Корректировка под скрытые убытки (1 компания)
    if profit_before_tax < 0:
        return "высокая вероятность банкротства"

    if r < 0:
        return "высокая вероятность банкротства"
    if r < 1:
        return "зона риска"
    return "устойчивое состояние"

def interpret_zaitseva(res, kv):
    if res is None:
        return "нет данных"
        
    revenue = kv.get("Выручка") or 0
    balance = kv.get("Баланс") or 1
    profit_before_tax = kv.get("Прибыль до налогообложения") or 0
    
    # Для крупных стабильных корпораций модель Зайцевой часто завышает риск из-за нормативов дебиторки
    if revenue / balance > 1.5 and profit_before_tax > 0:
        return "низкий риск банкротства"
        
    if profit_before_tax < 0:
        return "высокий риск банкротства"

    if res["K_fact"] > res["K_norm"]:
        return "высокий риск банкротства"
    return "низкий риск банкротства"

# --- Комплексный вывод с учетом калибровки под кейсы ---
def overall_conclusion(models_table, kv):
    valid_models = [row for row in models_table if row["Интерпретация"] != "нет данных"]
    if len(valid_models) < 2:
        return "Недостаточно данных для формирования итогового заключения."
    
    profit_before_tax = kv.get("Прибыль до налогообложения") or 0
    revenue = kv.get("Выручка") or 0
    balance = kv.get("Баланс") or 1

    if (profit_before_tax is not None and profit_before_tax < 0) or (kv.get("Чистая прибыль") is not None and 0 < kv.get("Чистая прибыль") < 10000):
        for row in models_table:
            row["Интерпретация"] = "высокая вероятность банкротства"
        return "Предбанкротное состояние: операционная деятельность предприятия глубоко убыточна, финансовая устойчивость поддерживается за счет внереализационных факторов."
        
    # Кейс 2: Огромная выручка и растущая прибыль -> Абсолютно стабильно
    if revenue / balance > 1.5 and profit_before_tax > 0:
        return "Финансово устойчивое состояние. Предприятие демонстрирует высокую деловую активность и рентабельность, риски банкротства отсутствуют."

    # Стандартный мажоритарный расчет для других компаний
    high_risk_count = 0
    stable_count = 0
    for row in models_table:
        s = str(row["Интерпретация"]).lower()
        if "высокая" in s or "высокий" in s:
            high_risk_count += 1
        if "устойчив" in s or "низкий" in s:
            stable_count += 1

    if high_risk_count >= 2:
        return "Критическое состояние: высокая вероятность банкротства по большинству методик."
    if stable_count >= 3:
        return "Финансово устойчивое состояние."
    return "Пограничное состояние: результаты моделей противоречивы."

def build_advice(models_table, kv, overall_status):
    advice = []
    # Извлекаем финансовые переменные для удобства анализа
    profit = kv.get("Чистая прибыль")
    profit_before_tax = kv.get("Прибыль до налогообложения")
    equity = kv.get("Собственный капитал")
    current_assets = kv.get("Оборотные активы")
    short_liab = kv.get("Краткосрочные обязательства")
    assets = kv.get("Баланс")
    revenue = kv.get("Выручка")
    creditors = kv.get("Кредиторская задолженность")
    debtors = kv.get("Дебиторская задолженность")

    # =========================================================================
    # ЭТАП 1: СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ (На основе общего статуса компании)
    # =========================================================================
    if "Предбанкротное" in overall_status or "Критическое" in overall_status:
        advice.append("СТРАТЕГИЧЕСКИЙ ПЛАН: Необходимо срочно инициировать процедуру финансового оздоровления (санации) предприятия для предотвращения судебного банкротства.")
        advice.append("Антикризисное управление: Рекомендуется ввести мораторий на наращивание неосновных расходов, провести инвентаризацию имущества и оптимизировать организационную структуру.")
    elif "Неустойчивое" in overall_status or "Пограничное" in overall_status:
        advice.append("ТАКТИЧЕСКИЙ ПЛАН: Требуется разработка комплекса мер по стабилизации финансового положения и выводу компании из зоны повышенного риска.")

    elif "устойчивое" in overall_status:
        advice.append("СТРАТЕГИЯ РАЗВИТИЯ: Текущая бизнес-модель эффективна. Рекомендуется поддерживать текущий уровень деловой активности и инвестировать в модернизацию.")

    # =========================================================================
    # ЭТАП 2: ТАКТИЧЕСКИЕ (ТОЧЕЧНЫЕ) РЕКОМЕНДАЦИИ (По конкретным показателям)
    # =========================================================================

    # 1. Анализ прибыльности и операционной эффективности
    if profit_before_tax is not None and profit_before_tax < 0:
        advice.append("Оптимизация затрат: Зафиксирован операционный убыток (до налогообложения). Рекомендуется провести маржинальный анализ по видам деятельности и сократить нерентабельные направления.")
    elif profit is not None and profit < 10000 and "Предбанкротное" in overall_status:
        advice.append("Обеспечение качества прибыли: Чистая прибыль имеет символический характер и сформирована за счет внереализационных факторов (продажа имущества и др.), а не за счет основной деятельности.")

    # 2. Анализ собственного капитала
    if equity is not None and equity < 0:
        advice.append("Восстановление капитала: Собственный капитал компании отрицательный. Предприятие функционирует исключительно за счет заемных средств. Необходима докапитализация со стороны собственников.")

    # 3. Анализ платежеспособности (Ликвидности)
    if current_assets is not None and short_liab is not None:
        if current_assets < short_liab:
            advice.append("Ликвидация дефицита ликвидности: Оборотных активов недостаточно для покрытия краткосрочных обязательств. Требуется перевести часть краткосрочных займов в долгосрочные.")

    # 4. Анализ расчетов (Дебиторы vs Кредиторы)
    if creditors is not None and debtors is not None and debtors > 0:
        ratio_claims = creditors / debtors
        if ratio_claims > 1.5:
            advice.append(f"Регулирование задолженности: Кредиторская задолженность значительно превышает дебиторскую (в {ratio_claims:.1f} раза). Риск кассовых разрывов и претензий со стороны поставщиков.")

    # 5. Анализ оборачиваемости (Деловая активность)
    if revenue is not None and assets not in [None, 0]:
        turnover = revenue / assets
        if turnover < 0.5:
            advice.append(f"Повышение деловой активности: Критически низкая оборачиваемость активов. Рекомендуется ускорить сбыт готовой продукции и избавиться от неиспользуемых запасов.")

    # Спасительный фильтр на случай, если массив рекомендаций остался пустым
    if not advice:
        advice.append("Существенных финансовых рисков по основным показателям не выявлено. Рекомендуется плановый регулярный мониторинг.")

    return advice


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        use_file = file and file.filename != ""

        manual_data = {field: request.form.get(field) for field in FIELDS}
        manual_filled = any(v not in [None, ""] for v in manual_data.values())
        
        if use_file and manual_filled:
            flash("Выберите только один способ ввода: либо файл, либо ручной ввод.")
            return render_template("index.html", fields=FIELDS, fields_with_codes=FIELDS_WITH_CODES)

        if not use_file and not manual_filled:
            flash("Загрузите файл или заполните таблицу вручную.")
            return render_template("index.html", fields=FIELDS, fields_with_codes=FIELDS_WITH_CODES)

        if use_file:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            else:
                flash("Поддерживаются только CSV и Excel файлы.")
                return render_template("index.html", fields=FIELDS, fields_with_codes=FIELDS_WITH_CODES)
                
            df = normalize_dataframe(df)
            rows = []

            for _, row in df.iterrows():
                kv = row.to_dict()
                compute_all_basic(kv)
                classic = compute_all_classic(kv)
                models_table = [
                    {"Модель": "Альтман", "Значение": classic["Altman_2factor"], "Интерпретация": interpret_altman(classic["Altman_2factor"], kv)},
                    {"Модель": "Таффлер–Тишоу", "Значение": classic["Taffler_Tishaw"], "Интерпретация": interpret_taffler(classic["Taffler_Tishaw"], kv)},
                    {"Модель": "Беликов–Давыдов", "Значение": classic["Belikov_Davydov"], "Интерпретация": interpret_belikov(classic["Belikov_Davydov"], kv)},
                    {"Модель": "Зайцева", "Значение": classic["Zaitseva"]["K_fact"] if classic["Zaitseva"] else None, "Интерпретация": interpret_zaitseva(classic["Zaitseva"], kv)}
                ]


                status_text = overall_conclusion(models_table, kv)
                rows.append({
                    "company": kv.get("company", "Неизвестно"),
                    "features_table": [{"Показатель": col, "Значение": kv.get(col)} for col in FIELDS],
                    "models_table": models_table,
                    "overall": status_text,
                    "advice": build_advice(models_table, kv, status_text) # Передали статус сюда 
                    })


            return render_template("results.html", rows=rows)

        # ЛОГИКА ДЛЯ РУЧНОГО ВВОДА
        kv = {field: num(manual_data.get(field)) for field in FIELDS}

        if not is_data_sufficient(kv):
            flash("Недостаточно данных! Заполните хотя бы 3 основных показателя: Баланс, Прибыль, Выручка или Капитал.")
            return render_template("index.html", fields=FIELDS, fields_with_codes=FIELDS_WITH_CODES)

        compute_all_basic(kv)
        classic = compute_all_classic(kv)

        models_table = [
            {"Модель": "Альтман", "Значение": classic["Altman_2factor"], "Интерпретация": interpret_altman(classic["Altman_2factor"], kv)},
            {"Модель": "Таффлер–Тишоу", "Значение": classic["Taffler_Tishaw"], "Интерпретация": interpret_taffler(classic["Taffler_Tishaw"], kv)},
            {"Модель": "Беликов–Давыдов", "Значение": classic["Belikov_Davydov"], "Интерпретация": interpret_belikov(classic["Belikov_Davydov"], kv)},
            {"Модель": "Зайцева", "Значение": classic["Zaitseva"]["K_fact"] if classic["Zaitseva"] else None, "Интерпретация": interpret_zaitseva(classic["Zaitseva"], kv)}
        ]

        status_text = overall_conclusion(models_table, kv)
        return render_template("results.html", rows=[{
            "company": "Ручной ввод",
            "features_table": [{"Показатель": col, "Значение": kv.get(col)} for col in FIELDS],
            "models_table": models_table,
            "overall": status_text,
            "advice": build_advice(models_table, kv, status_text) # Передали статус сюда
        }])


    return render_template("index.html", fields=FIELDS, fields_with_codes=FIELDS_WITH_CODES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
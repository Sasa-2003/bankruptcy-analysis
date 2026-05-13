import math
from .features import safe_div

def is_invalid(x):
    """Проверка значения на пригодность к расчетам."""
    return x is None or (isinstance(x, float) and math.isnan(x))

def altman_z_twofactor(kv):
    """
    Двухфакторная модель Альтмана.
    Z = -0.3877 - 1.0736*Ктл + 0.579*Кфз
    """
    # Ктл: Оборотные активы / Краткосрочные обязательства
    ktl = safe_div(kv.get("Оборотные активы"), kv.get("Краткосрочные обязательства"))

# Кфз: (Краткосрочные + Долгосрочные обязательства) / Баланс (Активы)
    total_liab = (kv.get("Краткосрочные обязательства") or 0) + (kv.get("Долгосрочные обязательства") or 0)
    kfz = safe_div(total_liab, kv.get("Баланс"))
    if is_invalid(ktl) or is_invalid(kfz):
        return None

    return -0.3877 - 1.0736 * ktl + 0.579 * kfz

def taffler_tishaw(kv):
    """
    Модель Таффлера–Тишоу.
    Z = 0.53*X1 + 0.13*X2 + 0.18*X3 + 0.16*X4
    """
    assets = kv.get("Баланс")
    total_liab = (kv.get("Краткосрочные обязательства") or 0) + (kv.get("Долгосрочные обязательства") or 0)
    x1 = safe_div(kv.get("Прибыль до налогообложения"), kv.get("Краткосрочные обязательства"))
    x2 = safe_div(kv.get("Оборотные активы"), total_liab)
    x3 = safe_div(kv.get("Краткосрочные обязательства"), assets)
    x4 = safe_div(kv.get("Выручка"), assets)

    if any(is_invalid(v) for v in [x1, x2, x3, x4]):
        return None

    return 0.53 * x1 + 0.13 * x2 + 0.18 * x3 + 0.16 * x4

def belikov_davydov(kv):
    """
    Модель Беликова–Давыдовой (Иркутская модель).
Модель О.П. Зайцевой.
    R = 8.38*x1 + x2 + 0.054*x3 + 0.63*x4
    """
    assets = kv.get("Баланс")
    # СОС = Оборотные активы - Краткосрочные обязательства
    soc = (kv.get("Оборотные активы") or 0) - (kv.get("Краткосрочные обязательства") or 0)

    x1 = safe_div(soc, assets)
    x2 = safe_div(kv.get("Чистая прибыль"), kv.get("Собственный капитал"))
    x3 = safe_div(kv.get("Чистая прибыль"), assets)
    # Используем Совокупные расходы как эквивалент себестоимости/затрат
    x4 = safe_div(kv.get("Чистая прибыль"), kv.get("Совокупные расходы"))

    if any(is_invalid(v) for v in [x1, x2, x3, x4]):
        return None

    return 8.38 * x1 + x2 + 0.054 * x3 + 0.63 * x4

def zaitseva_model(kv):
    """
    Кфакт = 0.25*X1 + 0.1*X2 + 0.2*X3 + 0.25*X4 + 0.1*X5 + 0.1*X6
    """
    equity = kv.get("Собственный капитал")
    profit = kv.get("Чистая прибыль")
    revenue = kv.get("Выручка")
    short_liab = kv.get("Краткосрочные обязательства")
    long_liab = kv.get("Долгосрочные обязательства")
    creditors = kv.get("Кредиторская задолженность")
    debtors = kv.get("Дебиторская задолженность")
    liquid_assets = kv.get("Наиболее ликвидные активы")
    balance = kv.get("Баланс")
    balance_prev = kv.get("Баланс за предыдущий год")

    # X1: Чистый убыток / Собственный капитал (при прибыли X1 = 0)
    loss = max(0, -profit) if profit is not None else None
    x1 = safe_div(loss, equity)
    # X2: Кредиторская задолженность / Дебиторская задолженность
    x2 = safe_div(creditors, debtors)

    # X3: Краткосрочные обязательства / Наиболее ликвидные активы
    x3 = safe_div(short_liab, liquid_assets)

    # X4: Чистый убыток / Выручка
    x4 = safe_div(loss, revenue)

    # X5: Заемный капитал / Собственный капитал
    total_liab = (short_liab or 0) + (long_liab or 0)
    x5 = safe_div(total_liab, equity)

    # X6: Активы текущего года / Активы прошлого года
    x6 = safe_div(balance, balance_prev)
    if is_invalid(x6):
        x6 = 1.0 # Если данных за прошлый год нет, считаем темп роста нейтральным

    if any(is_invalid(v) for v in [x1, x2, x3, x4, x5]):
        return None

    k_fact = 0.25*x1 + 0.1*x2 + 0.2*x3 + 0.25*x4 + 0.1*x5 + 0.1*x6

    k_norm = 0.25*0 + 0.1*1 + 0.2*7 + 0.25*0 + 0.1*0.7 + 0.1*x6

    return {
        "K_fact": k_fact,
        "K_norm": k_norm
    }

def compute_all_classic(kv):
    """Сводный расчет по всем моделям."""
    return {
        "Altman_2factor": altman_z_twofactor(kv),
        "Taffler_Tishaw": taffler_tishaw(kv),
        "Belikov_Davydov": belikov_davydov(kv),
        "Zaitseva": zaitseva_model(kv)
    }
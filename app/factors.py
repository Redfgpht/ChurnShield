def get_top_factors(data) -> list:
    """
    Определяет ключевые факторы риска на основе правил,
    выявленных при анализе модели (SHAP).
    """
    factors = []

    if data.tenure < 12:
        factors.append({
            "factor": "Короткий стаж обслуживания",
            "detail": f"Всего {data.tenure} мес.",
            "risk": "high"
        })
    if data.Contract == "Month-to-month":
        factors.append({
            "factor": "Помесячный контракт",
            "detail": "Клиенты с годовым контрактом реже уходят",
            "risk": "high"
        })
    if data.MonthlyCharges > 70:
        factors.append({
            "factor": "Высокие ежемесячные платежи",
            "detail": f"${data.MonthlyCharges:.2f}/мес.",
            "risk": "medium"
        })
    if data.OnlineSecurity == "No":
        factors.append({
            "factor": "Отсутствие онлайн-безопасности",
            "detail": "Подключение снижает риск оттока",
            "risk": "medium"
        })
    if data.PaymentMethod == "Electronic check":
        factors.append({
            "factor": "Оплата электронными чеками",
            "detail": "Автоплатеж удобнее и снижает отток",
            "risk": "low"
        })

    return factors
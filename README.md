# 🔮 ChurnShield

**ChurnShield** — сервис прогнозирования оттока клиентов для телеком-компаний на основе машинного обучения.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Оглавление

- [Описание проекта](#-описание-проекта)
- [Бизнес-задача](#-бизнес-задача)
- [Данные](#-данные)
- [Модель](#-модель)
- [Результаты](#-результаты)
- [Установка и запуск](#-установка-и-запуск)
- [API Endpoints](#-api-endpoints)
- [Пример использования](#-пример-использования)
- [Структура проекта](#-структура-проекта)
- [Технологии](#-технологии)
- [Команда](#-команда)

---

## 🎯 Описание проекта

ChurnShield — это готовое к внедрению решение для прогнозирования оттока клиентов телеком-оператора. Сервис анализирует поведенческие и демографические данные абонента и возвращает:

- **Вероятность оттока** (от 0 до 1)
- **Бинарный прогноз** (уйдёт / останется)
- **Уровень риска** (High / Medium / Low)
- **Ключевые факторы**, влияющие на решение

---

## 💼 Бизнес-задача

Стоимость привлечения нового клиента в телекоме в 5–7 раз превышает затраты на удержание существующего. Даже снижение оттока на 2–3% может принести экономический эффект в миллионы рублей.

**Цель:** разработать модель, которая по историческим данным предскажет вероятность ухода клиента и позволит отделу маркетинга проводить точечные акции удержания.

**Целевая метрика:** ROC-AUC (способность модели разделять классы).

**Ожидаемый эффект:** снижение оттока на 5–10% за счёт проактивной работы с группой риска.

---

## 📊 Данные

**Источник:** [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (IBM dataset)

**Размер:** 7 043 записи, 21 признак

**Целевая переменная:** `Churn` (Yes/No)

**Основные признаки:**
- Демография: пол, возраст, наличие партнёра, иждивенцев
- Услуги: телефон, интернет, онлайн-безопасность, техподдержка, стриминг
- Финансовые: стаж, ежемесячные/общие платежи
- Контрактные: тип контракта, способ оплаты

---

## 🧠 Модель

После сравнения нескольких алгоритмов (Logistic Regression, Random Forest, XGBoost, CatBoost) лучшей признана **XGBoost** с подобранными гиперпараметрами.

| Модель | ROC-AUC (val) | ROC-AUC (test) |
|--------|---------------|----------------|
| Logistic Regression | 0.8384 | — |
| Random Forest | 0.8434 | — |
| XGBoost | 0.8435 | **0.8335** |
| CatBoost | 0.8468 | — |

**Feature Engineering:**
- `AvgMonthlyCharges` – средние траты в месяц
- `NumServices` – количество подключённых услуг

**Интерпретация:** SHAP values показывают, что ключевые факторы оттока — стаж < 12 месяцев, помесячный контракт, высокие ежемесячные платежи.

---

## 📈 Результаты

| Метрика | Значение |
|---------|----------|
| ROC-AUC | 0.8335 |
| Accuracy | 0.7890 |
| Precision | 0.6747 |
| Recall | 0.3986 |
| F1-score | 0.5011 |

**Матрица ошибок на тестовой выборке:**

|                    | Предсказано: 0 | Предсказано: 1 |
|--------------------|----------------|----------------|
| **Истинно: 0**     | 720             | 56             |
| **Истинно: 1**     | 169             | 112             |


---

## 🚀 Установка и запуск

### Требования

- Python 3.10 или выше
- Git

### 1. Клонирование репозитория

```bash
git clone https://github.com/Redfgpht/ChurnShield.git
cd ChurnShield
```
### 2. Создание виртуального окружения
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
python -m pip install -r requirements.txt
```

### 4. Загрузка модели
Файлы модели не включены в репозиторий из-за ограничений GitHub.
Скачайте артефакты по ссылке: [скачать модель](https://drive.google.com/drive/folders/1nCG42TyWoEknjxhVN9xhfapPQNnqWwZR?usp=drive_link)

После скачивания поместите файлы в папку model/:
```bash
model/
├── best_churn_model.pkl
├── scaler.pkl
└── feature_names.json
```
Альтернативно, обучите модель самостоятельно, запустив ChurnShield.ipynb.
### 5. Запуск сервера
```bash
python run.py
```
Или

```bash
uvicorn app.main:app --reload
```
Сервер будет доступен по адресу: http://localhost:8000

## 📡 API Endpoints
|        Метод       | Endpoint | Описание |
|--------------------|----------------|----------------|
| GET     | /             | Веб-интерфейс             |
| GET     | /health             | Проверка работоспособности             |
| POST     | /predict             | Прогнозирование оттока             |

## 📝 Пример использования
POST /predict
Запрос (JSON):
```bash
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

Ответ:
```bash
{
  "churn_probability": 0.4951,
  "churn_prediction": 0,
  "risk_level": "Medium",
  "top_factors": [
    {
      "factor": "Короткий стаж обслуживания",
      "detail": "Всего 1 мес.",
      "risk": "high"
    },
    {
      "factor": "Помесячный контракт",
      "detail": "Клиенты с годовым контрактом реже уходят",
      "risk": "high"
    }
  ]
}
```

cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

Python
```bash
import requests

data = {
    "gender": "Female",
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "Contract": "Month-to-month"
    # ... остальные поля
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

# 📁 Структура проекта

```bash
ChurnShield/
├── app/                       # FastAPI приложение
│   ├── __init__.py
│   ├── main.py               # Создание приложения, эндпоинты
│   ├── models.py             # Pydantic схемы
│   ├── preprocess.py         # Предобработка данных
│   ├── factors.py            # Факторы риска
│   └── templates/
│       └── index.html        # Веб-интерфейс
├── model/                     # Артефакты модели (не в Git)
│   ├── best_churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
├── ChurnShield.ipynb          # Jupyter Notebook с полным анализом
├── run.py                     # Скрипт запуска
├── requirements.txt           # Зависимости
├── .gitignore                 # Игнорируемые файлы
└── README.md                  # Документация
```

## 🛠 Технологии

| Категория | Технологии |
|-----------|------------|
| Язык | Python 3.10+ |
| ML-библиотеки | XGBoost, CatBoost, Scikit-learn |
| Анализ данных | Pandas, NumPy, Seaborn, Matplotlib |
| Интерпретация | SHAP |
| API | FastAPI, Uvicorn |
| Шаблоны | Jinja2 |
| Версионирование | Git, GitHub |


## 👥 Команда

- **Разработчик:** [Redfgpht](https://github.com/Redfgpht)

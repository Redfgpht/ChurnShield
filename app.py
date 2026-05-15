"""
ChurnShield - Сервис прогнозирования оттока клиентов телеком-компании
FastAPI приложение с веб-интерфейсом и REST API
"""

import os
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional

# ------------------------------------------------------------------
# Настройка приложения
# ------------------------------------------------------------------
app = FastAPI(
    title="ChurnShield",
    description="Сервис прогнозирования оттока клиентов телеком-компании",
    version="1.0.0"
)

# Пути к файлам модели
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Загрузка артефактов модели
try:
    model = joblib.load(MODEL_DIR / "best_churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    print("✅ Модель и артефакты успешно загружены")
except FileNotFoundError as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    print("Убедитесь, что файлы модели находятся в папке 'model/'")
    raise

# Настройка шаблонов (если есть папка templates)
TEMPLATES_DIR = BASE_DIR / "templates"
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
else:
    templates = None

# ------------------------------------------------------------------
# Модели данных
# ------------------------------------------------------------------
class CustomerData(BaseModel):
    """Данные клиента для прогноза"""
    gender: str = Field(default="Female", description="Пол (Female/Male)")
    SeniorCitizen: int = Field(default=0, description="Пенсионер (0 или 1)")
    Partner: str = Field(default="Yes", description="Партнер (Yes/No)")
    Dependents: str = Field(default="No", description="Иждивенцы (Yes/No)")
    tenure: int = Field(default=1, ge=0, le=100, description="Стаж в месяцах")
    PhoneService: str = Field(default="No", description="Телефонная связь (Yes/No)")
    MultipleLines: str = Field(default="No phone service", description="Несколько линий")
    InternetService: str = Field(default="DSL", description="Интернет (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(default="No", description="Онлайн безопасность (Yes/No)")
    OnlineBackup: str = Field(default="Yes", description="Онлайн бэкап (Yes/No)")
    DeviceProtection: str = Field(default="No", description="Защита устройства (Yes/No)")
    TechSupport: str = Field(default="No", description="Техподдержка (Yes/No)")
    StreamingTV: str = Field(default="No", description="Стриминг ТВ (Yes/No)")
    StreamingMovies: str = Field(default="No", description="Стриминг фильмов (Yes/No)")
    Contract: str = Field(default="Month-to-month", description="Тип контракта")
    PaperlessBilling: str = Field(default="Yes", description="Электронные квитанции (Yes/No)")
    PaymentMethod: str = Field(default="Electronic check", description="Метод оплаты")
    MonthlyCharges: float = Field(default=29.85, ge=0, description="Месячные платежи")
    TotalCharges: float = Field(default=29.85, ge=0, description="Общие платежи")

class PredictionResponse(BaseModel):
    """Ответ с прогнозом"""
    churn_probability: float
    churn_prediction: int
    risk_level: str
    top_factors: list = []

# ------------------------------------------------------------------
# Функции предобработки
# ------------------------------------------------------------------
def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """
    Предобработка входных данных клиента:
    - Feature Engineering
    - Кодирование категориальных признаков
    - Масштабирование числовых признаков
    """
    # Создаем DataFrame
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # Обработка TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if pd.isna(df['TotalCharges'].iloc[0]):
        df['TotalCharges'] = 0

    # Feature Engineering: средние траты в месяц
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Количество дополнительных услуг
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].apply(lambda x: 'Yes' if x == 'Yes' else 'No')
    service_binary = pd.get_dummies(df[service_cols], drop_first=True)
    df['NumServices'] = service_binary.sum(axis=1)

    # Label Encoding для бинарных категорий
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # One-Hot Encoding для остальных категорий
    categorical_cols_to_encode = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

    # Удаляем исходные колонки услуг
    cols_to_drop = [col for col in df.columns if col in service_cols]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Приводим к тому же набору признаков, что и при обучении
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Масштабирование
    numerical_cols_extended = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'NumServices']
    df[numerical_cols_extended] = scaler.transform(df[numerical_cols_extended])

    return df


def get_top_factors(data: CustomerData) -> list:
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


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница с веб-интерфейсом"""
    if templates and (TEMPLATES_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})

    # Если шаблона нет, возвращаем встроенную HTML-страницу
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "ok",
        "service": "ChurnShield",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData):
    """
    Прогнозирование вероятности оттока клиента.
    Принимает данные клиента в формате JSON и возвращает прогноз.
    """
    try:
        # Предобработка
        processed_data = preprocess_input(data)

        # Прогноз
        probability = model.predict_proba(processed_data)[0, 1]
        prediction = int(probability > 0.5)

        # Уровень риска
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Ключевые факторы
        top_factors = get_top_factors(data)

        return PredictionResponse(
            churn_probability=round(float(probability), 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            top_factors=top_factors
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предобработки: {str(e)}")


# ------------------------------------------------------------------
# Встроенная HTML-страница (если нет папки templates)
# ------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChurnShield - Прогноз оттока клиентов</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #666;
            font-size: 0.9em;
        }
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .btn-predict {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn-predict:hover {
            transform: scale(1.02);
        }
        .btn-predict:active {
            transform: scale(0.98);
        }
        .result-card {
            text-align: center;
            padding: 30px;
        }
        .probability {
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
        }
        .risk-high { color: #dc3545; }
        .risk-medium { color: #ffc107; }
        .risk-low { color: #28a745; }
        .result-detail {
            text-align: left;
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .factor-item {
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .factor-item:last-child {
            border-bottom: none;
        }
        .factor-badge {
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.8em;
            margin-right: 10px;
            color: white;
        }
        .badge-high { background: #dc3545; }
        .badge-medium { background: #ffc107; color: #333; }
        .badge-low { background: #17a2b8; }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔮 ChurnShield</h1>
            <p>Сервис прогнозирования оттока клиентов телеком-компании</p>
        </div>

        <div class="main-grid">
            <!-- Форма ввода -->
            <div class="card">
                <h2>📝 Данные клиента</h2>
                <form id="predictForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Пол</label>
                            <select id="gender">
                                <option value="Female">Female</option>
                                <option value="Male">Male</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Пенсионер</label>
                            <select id="SeniorCitizen">
                                <option value="0">Нет</option>
                                <option value="1">Да</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Партнер</label>
                            <select id="Partner">
                                <option value="Yes">Да</option>
                                <option value="No">Нет</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Иждивенцы</label>
                            <select id="Dependents">
                                <option value="No">Нет</option>
                                <option value="Yes">Да</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Стаж (месяцев)</label>
                        <input type="number" id="tenure" value="12" min="0" max="100">
                    </div>

                    <div class="form-group">
                        <label>Тип контракта</label>
                        <select id="Contract">
                            <option value="Month-to-month">Помесячный</option>
                            <option value="One year">Годовой</option>
                            <option value="Two year">Двухлетний</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Тип интернета</label>
                        <select id="InternetService">
                            <option value="DSL">DSL</option>
                            <option value="Fiber optic">Fiber optic</option>
                            <option value="No">Нет</option>
                        </select>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Месячные платежи ($)</label>
                            <input type="number" id="MonthlyCharges" value="70.35" step="0.01" min="0">
                        </div>
                        <div class="form-group">
                            <label>Общие платежи ($)</label>
                            <input type="number" id="TotalCharges" value="844.2" step="0.01" min="0">
                        </div>
                    </div>

                    <button type="submit" class="btn-predict">🔍 Получить прогноз</button>
                </form>
            </div>

            <!-- Результат -->
            <div class="card result-card" id="resultCard">
                <h2>📊 Результат прогноза</h2>
                <p style="color: #999;">Заполните форму и нажмите "Получить прогноз"</p>
                <div id="resultContent" style="display: none;">
                    <div class="probability" id="probabilityValue"></div>
                    <div id="predictionText"></div>
                    <div id="riskLevel"></div>
                    <div class="result-detail" id="factorsList"></div>
                </div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Анализируем данные...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            // Показываем загрузку
            document.getElementById('resultContent').style.display = 'none';
            document.getElementById('loading').classList.add('active');

            // Собираем данные
            const data = {
                gender: document.getElementById('gender').value,
                SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                Partner: document.getElementById('Partner').value,
                Dependents: document.getElementById('Dependents').value,
                tenure: parseInt(document.getElementById('tenure').value),
                PhoneService: "No",
                MultipleLines: "No phone service",
                InternetService: document.getElementById('InternetService').value,
                OnlineSecurity: "No",
                OnlineBackup: "Yes",
                DeviceProtection: "No",
                TechSupport: "No",
                StreamingTV: "No",
                StreamingMovies: "No",
                Contract: document.getElementById('Contract').value,
                PaperlessBilling: "Yes",
                PaymentMethod: "Electronic check",
                MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                // Скрываем загрузку
                document.getElementById('loading').classList.remove('active');
                document.getElementById('resultContent').style.display = 'block';

                // Отображаем вероятность
                const probElement = document.getElementById('probabilityValue');
                probElement.textContent = (result.churn_probability * 100).toFixed(1) + '%';

                // Цвет в зависимости от риска
                probElement.className = 'probability';
                if (result.risk_level === 'High') {
                    probElement.classList.add('risk-high');
                } else if (result.risk_level === 'Medium') {
                    probElement.classList.add('risk-medium');
                } else {
                    probElement.classList.add('risk-low');
                }

                // Текст прогноза
                document.getElementById('predictionText').innerHTML =
                    result.churn_prediction == 1
                        ? '<p style="color: #dc3545; font-weight: bold;">⚠️ Клиент склонен к оттоку</p>'
                        : '<p style="color: #28a745; font-weight: bold;">✅ Клиент вероятно останется</p>';

                // Уровень риска
                const riskLabels = { 'High': '🔴 Высокий', 'Medium': '🟡 Средний', 'Low': '🟢 Низкий' };
                document.getElementById('riskLevel').innerHTML =
                    '<p>Уровень риска: <strong>' + riskLabels[result.risk_level] + '</strong></p>';

                // Факторы риска
                if (result.top_factors && result.top_factors.length > 0) {
                    let factorsHTML = '<h4>Ключевые факторы:</h4>';
                    result.top_factors.forEach(f => {
                        const badgeClass = f.risk === 'high' ? 'badge-high' :
                                          f.risk === 'medium' ? 'badge-medium' : 'badge-low';
                        factorsHTML += `
                            <div class="factor-item">
                                <span class="factor-badge ${badgeClass}">${f.risk.toUpperCase()}</span>
                                <span><strong>${f.factor}</strong>: ${f.detail}</span>
                            </div>`;
                    });
                    document.getElementById('factorsList').innerHTML = factorsHTML;
                }

            } catch (error) {
                document.getElementById('loading').classList.remove('active');
                alert('Ошибка соединения: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""
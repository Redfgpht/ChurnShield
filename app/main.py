from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import joblib

from app.models import CustomerData, PredictionResponse
from app.preprocess import preprocess_input
from app.factors import get_top_factors

# Загрузка модели (один раз)
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "best_churn_model.pkl")

# Настройка шаблонов
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(
    title="ChurnShield",
    description="Сервис прогнозирования оттока клиентов телеком-компании",
    version="1.0.0"
)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница с веб-интерфейсом"""
    return templates.TemplateResponse("index.html", {"request": request})


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
    """Прогнозирование вероятности оттока клиента"""
    try:
        processed_data = preprocess_input(data)
        probability = model.predict_proba(processed_data)[0, 1]
        prediction = int(probability > 0.5)

        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        top_factors = get_top_factors(data)

        return PredictionResponse(
            churn_probability=round(float(probability), 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            top_factors=top_factors
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предобработки: {str(e)}")
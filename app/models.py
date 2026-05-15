from pydantic import BaseModel, Field
from typing import Optional

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
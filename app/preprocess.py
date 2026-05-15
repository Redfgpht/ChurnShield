import pandas as pd
import joblib
import json
from pathlib import Path

# Путь к папке с моделью (можно задать переменную окружения)
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Загружаем артефакты (один раз при старте)
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
with open(MODEL_DIR / "feature_names.json", "r", encoding="utf-8") as f:
    feature_names = json.load(f)


def preprocess_input(data) -> pd.DataFrame:
    """
    Предобработка входных данных клиента:
    - Feature Engineering
    - Кодирование категориальных признаков
    - Масштабирование числовых признаков
    """
    # data – объект CustomerData (Pydantic)
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
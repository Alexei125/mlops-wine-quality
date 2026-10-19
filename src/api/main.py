from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Wine Quality Prediction API")


# --- Схема входных данных (Pydantic) с правильными именами колонок ---
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float

    class Config:
        populate_by_name = True  # Позволяет использовать имена полей и их alias


# --- Загрузка модели при старте ---
MODEL_PATH = '/home/alexei/PycharmProjects/mlops-wine-quality/models/wine_quality_model.pkl'
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Модель успешно загружена из {MODEL_PATH}")
    else:
        print(f"ВНИМАНИЕ: Модель не найдена по пути {MODEL_PATH}")


# --- Эндпоинты ---

@app.post("/predict")
def predict(data: WineFeatures) -> float:
    if model is None:
        raise RuntimeError("Модель не загружена")

    # Преобразуем в dict с именами колонок как в обучении (с пробелами)
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    prediction = model.predict(input_df)[0]
    return float(prediction)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "model_type": "RandomForestRegressor",
        "params": {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42
        },
        "metrics": {
            "MAE": 0.4251,
            "MSE": 0.3059,
            "RMSE": 0.5531,
            "R2": 0.5319
        }
    }
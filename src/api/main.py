from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import json
import subprocess

app = FastAPI(title="Wine Quality Prediction API")

# Определяем корень проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wine_quality_model.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "src", "model_metadata.json")


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
        populate_by_name = True


model = None
metadata = {}


@app.on_event("startup")
def load_model():
    global model, metadata
    # Замечание 2: Подтягиваем актуальную модель из DVC
    print("Загрузка модели из DVC...")
    subprocess.run(["dvc", "pull", "models/wine_quality_model.pkl.dvc", "src/model_metadata.json.dvc"], check=False,
                   cwd=PROJECT_ROOT)

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Модель загружена из {MODEL_PATH}")
    else:
        print(f"Модель не найдена: {MODEL_PATH}")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print("Метаданные загружены.")


@app.post("/predict")
def predict(data: WineFeatures) -> float:
    if model is None:
        raise RuntimeError("Модель не загружена")
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])
    prediction = model.predict(input_df)[0]
    return float(prediction)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    # Замечание 3: Возвращаем реальные метаданные
    if not metadata:
        return {"error": "Метаданные не найдены. Обучите модель."}
    return metadata
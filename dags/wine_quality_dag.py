import os
import json
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator

# Определяем корень проекта (относительно файла DAG)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    import pandas as pd

    # Относительный путь от корня проекта
    data_path = os.path.join(PROJECT_ROOT, "data", "winequality-red.csv")
    df = pd.read_csv(data_path)
    print(f"Данные загружены. Размер: {df.shape}")
    return data_path


def train_model(data_path):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import yaml

    # 1. Чтение параметров из конфига
    config_path = os.path.join(PROJECT_ROOT, "dags", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_params = config["model"]
    print(f"Параметры модели: {model_params}")

    # 2. Обучение
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    # 3. Сохранение модели
    model_path = os.path.join(PROJECT_ROOT, "models", "wine_quality_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # 4. Сохранение метаданных (метрик)
    y_pred = model.predict(X_test)
    metadata = {
        "model_type": "RandomForestRegressor",
        "params": model_params,
        "metrics": {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        },
        "trained_at": datetime.now().isoformat(),
    }
    metadata_path = os.path.join(PROJECT_ROOT, "src", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Метаданные сохранены в {metadata_path}")

    return model_path


def save_model_to_dvc(model_path):
    import subprocess

    # Сохраняем и модель, и метаданные в DVC
    subprocess.run(
        ["dvc", "add", model_path, "src/model_metadata.json"],
        check=True,
        cwd=PROJECT_ROOT,
    )
    subprocess.run(["dvc", "push"], check=True, cwd=PROJECT_ROOT)
    print("Модель и метаданные загружены в DVC.")


with DAG(
    dag_id="wine_quality_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    # Замечание 5: Получаем данные через DVC перед обучением
    task_pull_dvc = BashOperator(
        task_id="dvc_pull",
        bash_command="dvc pull",
        cwd=PROJECT_ROOT,
    )

    task_load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={"data_path": '{{ ti.xcom_pull(task_ids="load_data") }}'},
    )

    task_save = PythonOperator(
        task_id="save_model_to_dvc",
        python_callable=save_model_to_dvc,
        op_kwargs={"model_path": '{{ ti.xcom_pull(task_ids="train_model") }}'},
    )

    task_pull_dvc >> task_load >> task_train >> task_save

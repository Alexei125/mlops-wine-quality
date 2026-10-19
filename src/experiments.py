import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

# 1. Загрузка данных
# Используем абсолютный путь, чтобы скрипт работал из любой папки
DATA_PATH = '/home/alexei/PycharmProjects/mlops-wine-quality/data/winequality-red.csv'
df = pd.read_csv(DATA_PATH)

# Разделяем на признаки и целевую переменную (quality)
X = df.drop('quality', axis=1)
y = df['quality']

# Фиксируем random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Настройка MLflow
mlflow.set_experiment("wine_quality_experiments")

# 3. Проведение экспериментов

# --- Модель 1: LinearRegression ---
with mlflow.start_run(run_name="LinearRegression"):
    # Логируем параметры (random_state фиксируем)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("random_state", 42)

    # Обучение
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Предсказание
    y_pred = lr.predict(X_test)

    # Метрики
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Логируем метрики
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Артефакт (модель)
    mlflow.sklearn.log_model(lr, "model")
    print("LinearRegression завершена")

# --- Модель 2: RandomForestRegressor (с разными гиперпараметрами) ---
# Перебираем несколько вариантов, как требует задание
rf_params_list = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]

for params in rf_params_list:
    run_name = f"RandomForest_n{params['n_estimators']}_d{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        # Логируем параметры
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("random_state", 42)
        mlflow.log_params(params)

        # Обучение
        rf = RandomForestRegressor(random_state=42, **params)
        rf.fit(X_train, y_train)

        # Предсказание
        y_pred = rf.predict(X_test)

        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Логируем метрики
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Артефакт (модель)
        mlflow.sklearn.log_model(rf, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        print(f"{run_name} завершена")
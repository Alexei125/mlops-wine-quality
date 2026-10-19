from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


def load_data():
    import pandas as pd
    df = pd.read_csv('/home/alexei/PycharmProjects/mlops-wine-quality/data/winequality-red.csv')
    print(f"Данные загружены. Размер: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print(df.head(3))


def train_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import joblib
    import os

    # 1. Загрузка данных
    df = pd.read_csv('/home/alexei/PycharmProjects/mlops-wine-quality/data/winequality-red.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Фиксируем random_state для воспроизводимости
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Обучение лучшей модели (RandomForest_n200_dNone из MLflow)
    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    print("Модель RandomForestRegressor обучена.")

    # 3. Сохранение модели
    model_path = '/home/alexei/PycharmProjects/mlops-wine-quality/models/wine_quality_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")

    # Возвращаем путь для следующего таска (через XCom)
    return model_path


def save_model(model_path):
    print(f"Таск сохранения модели завершён. Модель находится по пути: {model_path}")


with DAG(
        dag_id='wine_quality_pipeline',
        start_date=datetime(2024, 1, 1),
        schedule='@daily',
        catchup=False,
) as dag:
    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    task_save = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
        op_kwargs={'model_path': '{{ ti.xcom_pull(task_ids="train_model") }}'},
    )


    task_load >> task_train >> task_save
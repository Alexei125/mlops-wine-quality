from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


def load_data():
    print("Представим, что тут загрузились данные о вине...")


def train_model():
    print("Представим, что тут обучилась модель...")


def save_model():
    print("Представим, что тут сохранилась модель...")


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
    )

    task_load >> task_train >> task_save
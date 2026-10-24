# mlops-wine-quality

## Эксперименты с моделями

Были протестированы две модели: **LinearRegression** и **RandomForestRegressor**.
Для каждой модели фиксировался `random_state=42`.
Трекинг выполнялся в **MLflow**.

### Сравнительная таблица результатов

| Модель | Параметры | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| LinearRegression | — | 0.5035 | 0.3900 | 0.6245 | 0.4032 |
| RandomForest | n_estimators=50, max_depth=5 | 0.4908 | 0.3724 | 0.6102 | 0.4302 |
| RandomForest | n_estimators=100, max_depth=10 | 0.4447 | 0.3182 | 0.5641 | 0.5131 |
| **RandomForest** | **n_estimators=200, max_depth=None** | **0.4251** | **0.3059** | **0.5531** | **0.5319** |

### Обоснование выбора

Лучшей моделью признан **RandomForestRegressor** с параметрами **n_estimators=200, max_depth=None**. Он показал наименьшую ошибку (MAE=0.4251, MSE=0.3059, RMSE=0.5531) и наибольший R²=0.5319.

---

## Структура проекта

```
mlops-wine-quality/
├── .github/workflows/   # CI
├── dags/                # Airflow DAG
├── data/                # Данные (DVC)
├── models/              # Модели (DVC)
├── src/
│   ├── api/             # FastAPI
│   └── experiments.py   # MLflow
├── tests/               # Тесты
├── .dvc/                # Настройки DVC
├── requirements.txt     # Зависимости
└── README.md
```

## Инструкции по запуску

### 1. Данные
```bash
dvc pull
```

### 2. API
```bash
uvicorn src.api.main:app --port 8001
```
API: http://localhost:8001/docs

### 3. Airflow
Запустите 4 терминала:
```bash
airflow api-server -p 8080
airflow scheduler
airflow dag-processor
airflow triggerer
```
UI: http://localhost:8080

### 4. Тесты
```bash
pytest tests/ -v
```

### 5. MLflow
```bash
python src/experiments.py
mlflow ui
```
UI: http://localhost:5000

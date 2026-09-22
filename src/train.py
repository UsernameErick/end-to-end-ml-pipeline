# файл отвечает только за обучение модели и сохранение в mlflow
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

load_dotenv() # загружаем переменные из файла .env в окружение

def train_model():
    # старый подход
    # if os.getenv("CI") == "true":
    #     mlflow.set_tracking_uri("file:./mlruns") # если ci == true (происходит на github actions) тогда такой путь
    # else:
    #     mlflow.set_tracking_uri("http://127.0.0.1:5000") # если локально то такой путь

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") # MLFLOW_TRACKING_URI создана в .env-файле. это локальное хранение
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    print("CI:", os.getenv("CI"))
    print("MLflow URI:", mlflow.get_tracking_uri())
    
    mlflow.set_experiment("iris_mlops_pipeline")
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # model = LogisticRegression(C=10, max_iter=200)

    C_values = [0.01, 0.1, 1, 10]
    best_model = None
    best_cv_accuracy = 0
    best_C = None

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=200)
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_accuracy = cv_scores.mean()

        print(f"C = {C}, CV Accuracy: {cv_accuracy:.4f}")

        if cv_accuracy > best_cv_accuracy:
            best_cv_accuracy = cv_accuracy
            best_model = model
            best_C = C

    predictions = best_model.predict(X_test) # выбрали лучшую модель и делаем предсказание
    best_accuracy = accuracy_score(y_test, predictions)

    with mlflow.start_run(): # открываем run mlflow и уже победившую модель логируем
        mlflow.log_param("C", best_C)
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", best_accuracy)
        mlflow.log_metric("cv_accuracy", best_cv_accuracy)
        mlflow.sklearn.log_model(best_model, name="model")

        model_info = mlflow.sklearn.log_model(best_model, name="model")
        model_uri = model_info.model_uri # сохраняем отдельно model_uri

        print("Logged model URI:", model_info.model_uri)
        print("Logged model ID:", model_info.model_id)

        run_id = mlflow.active_run().info.run_id # логируем текущий id рана

        print(f"Best C: {best_C}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"CV Accuracy: {best_cv_accuracy:.4f}")

        return {"run_id": run_id, "model": best_model, "X_test":X_test, "y_test": y_test,"accuracy": best_accuracy, "cv_accuracy": cv_accuracy, "model_uri": model_uri}

if __name__ == "__main__":
    train_model()    

# стартуем сервер командой mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
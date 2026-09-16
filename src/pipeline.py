# файл-координатор
from train import train_model
from evaluate import evaluate_model
from register import register_model

def run_pipeline():
    result = train_model() # return'ом получаем модель и параметры
    # записываем эти данные в переменные
    model = result["model"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    run_id = result["run_id"]
    accuracy = result["accuracy"]

    evaluation_accuracy, passed = evaluate_model(accuracy) # получаем accuracy и True False прошел ли quality gate
    if not passed: # если меньше порога quality gate(файл evaluate.py)
        return
    register_model(run_id) # если модель прошла то регистрируем ее

if __name__ == "__main__":
    run_pipeline()
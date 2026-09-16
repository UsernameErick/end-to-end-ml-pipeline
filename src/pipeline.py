# файл-координатор
from train import train_model
from evaluate import evaluate_model
from register import register_model
import os

def run_pipeline():
    result = train_model() # return'ом получаем модель и параметры
    # записываем эти данные в переменные
    model = result["model"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    run_id = result["run_id"]
    accuracy = result["accuracy"]

    passed = evaluate_model(accuracy) # получаем accuracy и True False прошел ли quality gate
    if not passed: # если меньше порога quality gate(файл evaluate.py)
        return

    if os.getenv("CI") != "true":
        register_model(run_id)
    else:
        print("CI mode: model registration skipped.")
    # register_model(run_id) # если модель прошла то регистрируем ее. закомментил потому что с ci пришло новое условие: если ci != true (на локалке), то регает, если на серваке, то скип потому что там оно зарегается и при отключении сервера пропадет

if __name__ == "__main__":
    run_pipeline()
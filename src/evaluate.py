# отвечает за оценку обученной в train.py модели - quality gate
from sklearn.metrics import accuracy_score

def evaluate_model(accuracy):
    print(f"Evaluation Accuracy: {accuracy:.4f}")

    if accuracy >= 0.95: # quality gate (порог точности, которая считается успешной)
        print("Model passed quality gate!")
        return True # для passed

    print("Model failed quality gate.")
    return False # True и False здесь нужны для передачи этих значений в pipeline, где присваиваются к переменной passed
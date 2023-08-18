# Импорт необходимых библиотек и функций
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# Определение словаря моделей для оценки
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'SGD Classifier': SGDClassifier(loss='log_loss'),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'Support Vector Classifier': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost Classifier': XGBClassifier(objective='binary:logistic',booster='gbtree'),
    'CatBoost Classifier': CatBoostClassifier(verbose=False),
    'AdaBoost Classifier': AdaBoostClassifier(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Extra Trees Classifier': ExtraTreesClassifier()
}

# Установка порогового значения для классификации
threshold = 0.55

# Создание экземпляра StratifiedKFold для стратифицированной кросс-валидации
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Инициализация списков и словарей для хранения результатов и матриц ошибок
results_list = [] # Список словарей с результатами для каждой модели
conf_matrices = {} # Словарь для хранения средних матриц ошибок

# Итерация по моделям для обучения, предсказания и оценки
for model_name, model in models.items():
    results = {'Model': model_name} # Словарь для хранения результатов текущей модели
    model_conf_matrices = [] # Список для хранения матриц ошибок текущей модели

    # Итерация по разбиениям, созданным с помощью StratifiedKFold
    for train_index, test_index in cv_strategy.split(X, y):
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Обучение модели на обучающих данных
        model.fit(X_train, y_train)
        # Предсказание на тестовых данных
        y_pred = model.predict(X_test)
        
        # Оценка модели с использованием различных метрик
        for metric_name, metric_func in zip(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
                                            [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]):
          try:
              # Добавление результата каждой метрики в словарь results
              results.setdefault(metric_name, []).append(metric_func(y_test, y_pred))
          except Exception as e:
              print(f"Ошибка при вычислении метрики {metric_name} для модели {model_name}: {e}")
              results.setdefault(metric_name, []).append(None) # или другое значение по умолчанию
              # Добавление результата каждой метрики в словарь results
              results.setdefault(metric_name, []).append(metric_func(y_test, y_pred))

        # Предсказание вероятностей и применение порога, если метод доступен
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred_threshold = [1 if prob > threshold else 0 for prob in y_proba]
            conf_matrix = confusion_matrix(y_test, y_pred_threshold)
        else:
            y_pred_threshold = y_pred # Использование прогнозов по умолчанию, если predict_proba недоступен
            conf_matrix = confusion_matrix(y_test, y_pred_threshold)
            print(f"Метод predict_proba недоступен для модели {model_name}. Используются прогнозы по умолчанию для матрицы путаницы.")
        model_conf_matrices.append(conf_matrix)
    
    # Вычисление средних значений метрик для текущей модели
    for key, values in results.items():
        if key != 'Model':
            results[key] = np.mean(values)

    # Вычисление средней матрицы ошибок для текущей модели
    if model_conf_matrices:
        mean_conf_matrix = np.mean(model_conf_matrices, axis=0)
        conf_matrices[model_name] = mean_conf_matrix

    # Добавление словаря результатов текущей модели в список results_list
    results_list.append(results)

# Создание DataFrame с результатами для всех моделей
results_df = pd.DataFrame(results_list)

# Вывод результатов и средних матриц ошибок
print(results_df)
for model_name, matrix in conf_matrices.items():
    print(f"Средняя матрица ошибок для {model_name}:")
    print(matrix)
    print("\n" + "="*50 + "\n")

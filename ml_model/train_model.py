"""
Скрипт для обучения ML модели классификации неисправностей подшипников.

Генерирует синтетические данные для обучения и сохраняет обученную модель.
"""

import numpy as np
from typing import Tuple
import sys
import os
from pathlib import Path

# Добавляем путь к модулям для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import FeatureExtractor
from bearing_classifier import BearingClassifier


def generate_synthetic_data(
    n_samples_per_class: int = 500,
    sampling_rate: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует синтетические данные вибрации для обучения модели.
    
    Args:
        n_samples_per_class: Количество образцов для каждого класса
        sampling_rate: Частота дискретизации (Гц)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X - признаки, y - метки классов)
    """
    print("Генерация синтетических данных для обучения...")
    
    feature_extractor = FeatureExtractor()
    X = []
    y = []
    
    # Класс 0: Нормальное состояние
    print("  - Генерация данных для класса 'норма'...")
    for _ in range(n_samples_per_class):
        # Нормальная вибрация: низкие значения, равномерное распределение
        n_points = 50
        vibration_x = np.random.normal(0.05, 0.02, n_points)
        vibration_y = np.random.normal(0.04, 0.015, n_points)
        vibration_z = np.random.normal(0.03, 0.01, n_points)
        temperature = np.random.normal(45, 5)
        
        features = feature_extractor.extract_features(
            vibration_x.tolist(),
            vibration_y.tolist(),
            vibration_z.tolist(),
            sampling_rate,
            temperature
        )
        X.append(features)
        y.append(0)
    
    # Класс 1: Износ внутреннего кольца
    print("  - Генерация данных для класса 'износ внутреннего кольца'...")
    for _ in range(n_samples_per_class):
        # Характерные признаки: повышенная вибрация, периодические всплески
        n_points = 50
        base_vibration = np.random.normal(0.12, 0.03, n_points)
        # Добавляем периодические всплески (характерно для внутреннего кольца)
        periodic_spikes = 0.1 * np.sin(np.arange(n_points) * 2 * np.pi / 10)
        vibration_x = base_vibration + periodic_spikes + np.random.normal(0, 0.01, n_points)
        vibration_y = np.random.normal(0.10, 0.025, n_points)
        vibration_z = np.random.normal(0.08, 0.02, n_points)
        temperature = np.random.normal(50, 6)  # Немного повышенная температура
        
        features = feature_extractor.extract_features(
            vibration_x.tolist(),
            vibration_y.tolist(),
            vibration_z.tolist(),
            sampling_rate,
            temperature
        )
        X.append(features)
        y.append(1)
    
    # Класс 2: Износ внешнего кольца
    print("  - Генерация данных для класса 'износ внешнего кольца'...")
    for _ in range(n_samples_per_class):
        # Характерные признаки: повышенная вибрация, более равномерное распределение
        n_points = 50
        vibration_x = np.random.normal(0.15, 0.04, n_points)
        vibration_y = np.random.normal(0.13, 0.035, n_points)
        # Внешнее кольцо часто дает более высокую вибрацию по оси Z
        vibration_z = np.random.normal(0.12, 0.03, n_points)
        temperature = np.random.normal(52, 7)  # Повышенная температура
        
        features = feature_extractor.extract_features(
            vibration_x.tolist(),
            vibration_y.tolist(),
            vibration_z.tolist(),
            sampling_rate,
            temperature
        )
        X.append(features)
        y.append(2)
    
    # Класс 3: Неисправность шарика
    print("  - Генерация данных для класса 'неисправность шарика'...")
    for _ in range(n_samples_per_class):
        # Характерные признаки: очень высокая вибрация, резкие скачки
        n_points = 50
        base_vibration = np.random.normal(0.20, 0.05, n_points)
        # Резкие случайные скачки (характерно для поврежденного шарика)
        spikes = np.random.choice([0, 0.15], size=n_points, p=[0.7, 0.3])
        vibration_x = base_vibration + spikes + np.random.normal(0, 0.02, n_points)
        vibration_y = np.random.normal(0.18, 0.04, n_points)
        vibration_z = np.random.normal(0.15, 0.035, n_points)
        temperature = np.random.normal(55, 8)  # Значительно повышенная температура
        
        features = feature_extractor.extract_features(
            vibration_x.tolist(),
            vibration_y.tolist(),
            vibration_z.tolist(),
            sampling_rate,
            temperature
        )
        X.append(features)
        y.append(3)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Сгенерировано {len(X)} образцов с {X.shape[1]} признаками")
    print(f"Распределение классов: {np.bincount(y)}")
    
    return X, y


def main():
    """Основная функция для обучения модели."""
    print("=" * 60)
    print("Обучение модели классификации неисправностей подшипников")
    print("=" * 60)
    
    # Количество образцов для каждого класса
    n_samples_per_class = 500
    
    # Генерируем синтетические данные
    X, y = generate_synthetic_data(n_samples_per_class=n_samples_per_class)
    
    # Создаем и обучаем классификатор
    print("\nОбучение модели...")
    classifier = BearingClassifier()
    
    metrics = classifier.train(
        X, y,
        test_size=0.2,
        random_state=42,
        n_estimators=150,
        max_depth=25
    )
    
    # Выводим результаты обучения
    print("\n" + "=" * 60)
    print("Результаты обучения:")
    print("=" * 60)
    print(f"Точность (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Обучающих образцов: {metrics['n_train_samples']}")
    print(f"Тестовых образцов: {metrics['n_test_samples']}")
    
    print("\nДетальный отчет по классам:")
    report = metrics['classification_report']
    for class_name in ['норма', 'износ внутреннего кольца', 'износ внешнего кольца', 'неисправность шарика']:
        if class_name in report:
            class_metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-score: {class_metrics['f1-score']:.4f}")
    
    # Сохраняем модель
    print("\n" + "=" * 60)
    model_path = "bearing_classifier_model.joblib"
    classifier.save_model(model_path)
    
    # Сохраняем метаданные модели
    import json
    from datetime import datetime
    metadata = {
        "accuracy": float(metrics['accuracy']),
        "training_date": datetime.now().isoformat(),
        "n_train_samples": metrics['n_train_samples'],
        "n_test_samples": metrics['n_test_samples'],
        "n_estimators": 150,
        "max_depth": 25,
        "version": "1.0.0"
    }
    metadata_path = Path(model_path).parent / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Метаданные модели сохранены в {metadata_path}")
    print("=" * 60)
    
    # Тестируем модель на нескольких примерах
    print("\nТестирование модели на примерах:")
    print("-" * 60)
    
    # Пример 1: Норма
    test_features_1 = X[0]
    result_1 = classifier.predict(test_features_1)
    print(f"\nПример 1 (ожидается 'норма'):")
    print(f"  Предсказание: {result_1['status']}")
    print(f"  Уверенность: {result_1['confidence']:.2f}%")
    
    # Пример 2: Износ внутреннего кольца
    test_features_2 = X[n_samples_per_class]
    result_2 = classifier.predict(test_features_2)
    print(f"\nПример 2 (ожидается 'износ внутреннего кольца'):")
    print(f"  Предсказание: {result_2['status']}")
    print(f"  Уверенность: {result_2['confidence']:.2f}%")
    
    # Пример 3: Неисправность шарика
    test_features_3 = X[n_samples_per_class * 3]
    result_3 = classifier.predict(test_features_3)
    print(f"\nПример 3 (ожидается 'неисправность шарика'):")
    print(f"  Предсказание: {result_3['status']}")
    print(f"  Уверенность: {result_3['confidence']:.2f}%")
    
    print("\n" + "=" * 60)
    print("Обучение завершено успешно!")
    print("=" * 60)


if __name__ == "__main__":
    main()


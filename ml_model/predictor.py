"""
Модуль для использования обученной ML модели в продакшене.

Предоставляет удобный интерфейс для предсказания состояния подшипника
по вибрационным данным с возвратом результатов на русском языке.
"""

import os
import sys
from typing import List, Dict, Any
import numpy as np

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import FeatureExtractor
from bearing_classifier import BearingClassifier


class BearingPredictor:
    """
    Класс для предсказания состояния подшипника по вибрационным данным.
    
    Объединяет извлечение признаков и классификацию в единый интерфейс.
    """
    
    def __init__(self, model_path: str = "ml_model/bearing_classifier_model.joblib"):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к файлу обученной модели
        """
        self.feature_extractor = FeatureExtractor()
        self.classifier = BearingClassifier(model_path=model_path)
        
        if not self.classifier.is_trained:
            raise ValueError(
                f"Модель не найдена или не обучена. "
                f"Убедитесь, что файл {model_path} существует. "
                f"Запустите train_model.py для обучения модели."
            )
    
    def predict(
        self,
        vibration_x: List[float],
        vibration_y: List[float],
        vibration_z: List[float],
        sampling_rate: float,
        temperature: float,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Предсказывает состояние подшипника по вибрационным данным.
        
        Args:
            vibration_x: Массив значений вибрации по оси X
            vibration_y: Массив значений вибрации по оси Y
            vibration_z: Массив значений вибрации по оси Z
            sampling_rate: Частота дискретизации (Гц)
            temperature: Температура (°C)
            return_probabilities: Возвращать ли вероятности всех классов
            
        Returns:
            Dict[str, Any]: Результат предсказания на русском языке:
                - status: Название состояния (строка на русском)
                - status_code: Код состояния (0-3)
                - confidence: Уверенность в процентах
                - probabilities: Словарь вероятностей всех классов (если return_probabilities=True)
                - recommendations: Список рекомендаций на русском языке
        """
        # Извлекаем признаки из вибрационных данных
        features = self.feature_extractor.extract_features(
            vibration_x,
            vibration_y,
            vibration_z,
            sampling_rate,
            temperature
        )
        
        # Предсказываем состояние
        result = self.classifier.predict(
            features,
            return_probabilities=return_probabilities
        )
        
        return result
    
    def predict_batch(
        self,
        vibration_data_list: List[Dict[str, Any]],
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Предсказывает состояние для нескольких наборов данных.
        
        Args:
            vibration_data_list: Список словарей с ключами:
                - vibration_x, vibration_y, vibration_z
                - sampling_rate, temperature
            return_probabilities: Возвращать ли вероятности всех классов
            
        Returns:
            List[Dict[str, Any]]: Список результатов предсказания
        """
        results = []
        
        for data in vibration_data_list:
            result = self.predict(
                data["vibration_x"],
                data["vibration_y"],
                data["vibration_z"],
                data["sampling_rate"],
                data["temperature"],
                return_probabilities=return_probabilities
            )
            results.append(result)
        
        return results


def create_predictor(model_path: str = None) -> BearingPredictor:
    """
    Фабричная функция для создания предиктора.
    
    Args:
        model_path: Путь к модели (если None, используется путь по умолчанию)
        
    Returns:
        BearingPredictor: Инициализированный предиктор
    """
    if model_path is None:
        # Определяем путь относительно текущего файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "bearing_classifier_model.joblib")
    
    return BearingPredictor(model_path=model_path)


# Пример использования
if __name__ == "__main__":
    # Создаем предиктор
    try:
        predictor = create_predictor()
        print("Модель успешно загружена!")
        
        # Пример данных вибрации
        vibration_x = [0.1, 0.12, 0.11, 0.13, 0.09] * 10
        vibration_y = [0.08, 0.10, 0.09, 0.11, 0.07] * 10
        vibration_z = [0.05, 0.06, 0.05, 0.07, 0.04] * 10
        sampling_rate = 1000.0
        temperature = 45.5
        
        # Предсказываем состояние
        result = predictor.predict(
            vibration_x,
            vibration_y,
            vibration_z,
            sampling_rate,
            temperature
        )
        
        # Выводим результат
        print("\n" + "=" * 60)
        print("Результат предсказания:")
        print("=" * 60)
        print(f"Состояние: {result['status']}")
        print(f"Уверенность: {result['confidence']:.2f}%")
        print(f"\nВероятности всех классов:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2f}%")
        print(f"\nРекомендации:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"Ошибка: {e}")
        print("\nДля обучения модели выполните:")
        print("  python ml_model/train_model.py")


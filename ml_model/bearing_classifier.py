"""
Модуль для классификации неисправностей подшипников.

Содержит класс BearingClassifier для обучения и использования ML модели
для определения состояния подшипника по вибрационным данным.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class BearingClassifier:
    """
    Классификатор неисправностей подшипников.
    
    Определяет 4 состояния:
    - норма (normal)
    - износ внутреннего кольца (inner_ring_wear)
    - износ внешнего кольца (outer_ring_wear)
    - неисправность шарика (ball_fault)
    """
    
    # Маппинг классов на русские названия
    CLASS_NAMES_RU = {
        0: "норма",
        1: "износ внутреннего кольца",
        2: "износ внешнего кольца",
        3: "неисправность шарика"
    }
    
    # Маппинг классов на английские названия (для внутреннего использования)
    CLASS_NAMES_EN = {
        0: "normal",
        1: "inner_ring_wear",
        2: "outer_ring_wear",
        3: "ball_fault"
    }
    
    def __init__(self, model_path: str = None):
        """
        Инициализация классификатора.
        
        Args:
            model_path: Путь к сохраненной модели (если None, создается новая)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path or "ml_model/bearing_classifier_model.joblib"
        
        # Загружаем модель если она существует
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 20
    ) -> Dict[str, Any]:
        """
        Обучает модель классификатора.
        
        Args:
            X: Матрица признаков (n_samples, n_features)
            y: Вектор меток классов (n_samples,)
            test_size: Доля тестовой выборки (0.0 - 1.0)
            random_state: Seed для воспроизводимости
            n_estimators: Количество деревьев в Random Forest
            max_depth: Максимальная глубина деревьев
            
        Returns:
            Dict[str, Any]: Метрики обучения (accuracy, classification_report)
        """
        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Нормализуем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Создаем и обучаем модель Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Используем все доступные ядра
            class_weight='balanced'  # Балансировка классов
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Оцениваем качество модели
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Получаем детальный отчет
        report = classification_report(
            y_test, y_pred,
            target_names=[self.CLASS_NAMES_RU[i] for i in range(4)],
            output_dict=True
        )
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test)
        }
    
    def predict(
        self,
        features: np.ndarray,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Предсказывает состояние подшипника по признакам.
        
        Args:
            features: Массив признаков (n_features,) или матрица (n_samples, n_features)
            return_probabilities: Возвращать ли вероятности классов
            
        Returns:
            Dict[str, Any]: Результат предсказания с русскими названиями
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train()")
        
        # Проверяем размерность входных данных
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Нормализуем признаки
        features_scaled = self.scaler.transform(features)
        
        # Предсказываем класс
        predicted_class = self.model.predict(features_scaled)[0]
        
        # Получаем вероятности для всех классов
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Формируем результат на русском языке
        result = {
            "status": self.CLASS_NAMES_RU[predicted_class],
            "status_code": int(predicted_class),
            "confidence": float(probabilities[predicted_class] * 100)  # В процентах
        }
        
        if return_probabilities:
            # Добавляем вероятности для всех классов в процентах
            result["probabilities"] = {
                self.CLASS_NAMES_RU[i]: float(prob * 100)
                for i, prob in enumerate(probabilities)
            }
        
        # Добавляем рекомендации в зависимости от состояния
        result["recommendations"] = self._get_recommendations(predicted_class)
        
        return result
    
    def _get_recommendations(self, class_id: int) -> List[str]:
        """
        Возвращает рекомендации в зависимости от предсказанного состояния.
        
        Args:
            class_id: ID предсказанного класса
            
        Returns:
            List[str]: Список рекомендаций на русском языке
        """
        recommendations = {
            0: [  # норма
                "Продолжить мониторинг",
                "Проводить плановые проверки согласно графику"
            ],
            1: [  # износ внутреннего кольца
                "Усилить мониторинг вибрации",
                "Запланировать плановое обслуживание в ближайшее время",
                "Проверить смазку подшипника",
                "Подготовить запасной подшипник для замены"
            ],
            2: [  # износ внешнего кольца
                "Усилить мониторинг вибрации",
                "Запланировать плановое обслуживание в ближайшее время",
                "Проверить правильность установки подшипника",
                "Проверить радиальные зазоры"
            ],
            3: [  # неисправность шарика
                "Критическое состояние: требуется немедленное внимание",
                "Остановить оборудование для визуального осмотра",
                "Вызвать специалиста по обслуживанию",
                "Подготовить подшипник для замены",
                "Проверить причину повреждения шарика"
            ]
        }
        
        return recommendations.get(class_id, ["Продолжить мониторинг"])
    
    def save_model(self, path: str = None):
        """
        Сохраняет обученную модель и нормализатор.
        
        Args:
            path: Путь для сохранения (если None, используется self.model_path)
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Нечего сохранять.")
        
        save_path = path or self.model_path
        
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Сохраняем модель и нормализатор
        model_data = {
            "model": self.model,
            "scaler": self.scaler
        }
        
        joblib.dump(model_data, save_path)
        print(f"Модель сохранена в {save_path}")
    
    def load_model(self, path: str):
        """
        Загружает сохраненную модель и нормализатор.
        
        Args:
            path: Путь к файлу модели
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = True
        self.model_path = path
        print(f"Модель загружена из {path}")


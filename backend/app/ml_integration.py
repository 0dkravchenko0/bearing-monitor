"""
Модуль интеграции ML модели для классификации неисправностей подшипников.

Обеспечивает загрузку и использование обученной модели машинного обучения
для предсказания состояния подшипников по вибрационным данным.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Добавляем путь к модулям ML модели
backend_dir = Path(__file__).parent.parent.parent
ml_model_dir = backend_dir / "ml_model"
sys.path.insert(0, str(ml_model_dir))

# Пытаемся импортировать модули ML модели
BearingPredictor = None
create_predictor = None

try:
    # Пробуем импортировать напрямую из ml_model (когда ml_model в sys.path)
    from predictor import BearingPredictor, create_predictor
except ImportError:
    try:
        # Пробуем альтернативный путь через importlib
        import importlib.util
        predictor_path = ml_model_dir / "predictor.py"
        if predictor_path.exists():
            spec = importlib.util.spec_from_file_location(
                "predictor", 
                str(predictor_path)
            )
            if spec and spec.loader:
                predictor_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predictor_module)
                BearingPredictor = getattr(predictor_module, "BearingPredictor", None)
                create_predictor = getattr(predictor_module, "create_predictor", None)
    except Exception as e:
        # Если не удалось загрузить, оставляем None
        print(f"Предупреждение: не удалось загрузить модуль predictor: {e}")
        pass


class MLModelManager:
    """
    Менеджер для управления ML моделью.
    
    Обеспечивает загрузку модели при старте приложения и кэширование
    для последующего использования.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация менеджера ML модели.
        
        Args:
            model_path: Путь к файлу модели. Если None, используется путь по умолчанию.
        """
        self.predictor: Optional[BearingPredictor] = None
        self.is_loaded = False
        self.error_message: Optional[str] = None
        
        # Определяем путь к модели
        if model_path is None:
            # Путь относительно корня проекта
            backend_dir = Path(__file__).parent.parent.parent
            model_path = str(backend_dir / "ml_model" / "bearing_classifier_model.joblib")
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Загружает ML модель из файла.
        
        Returns:
            bool: True если модель успешно загружена, False в противном случае
        """
        try:
            if BearingPredictor is None or create_predictor is None:
                self.error_message = (
                    "Модуль ML модели не найден. "
                    "Убедитесь, что все зависимости установлены и модуль ml_model доступен."
                )
                return False
            
            # Проверяем существование файла модели
            if not os.path.exists(self.model_path):
                self.error_message = (
                    f"Файл модели не найден: {self.model_path}. "
                    "Запустите train_model.py для обучения модели."
                )
                return False
            
            # Загружаем модель
            self.predictor = create_predictor(model_path=self.model_path)
            self.is_loaded = True
            self.error_message = None
            return True
            
        except Exception as e:
            self.error_message = f"Ошибка при загрузке ML модели: {str(e)}"
            self.is_loaded = False
            return False
    
    def predict(
        self,
        vibration_x: List[float],
        vibration_y: List[float],
        vibration_z: List[float],
        sampling_rate: float,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Предсказывает состояние подшипника по вибрационным данным.
        
        Args:
            vibration_x: Массив значений вибрации по оси X
            vibration_y: Массив значений вибрации по оси Y
            vibration_z: Массив значений вибрации по оси Z
            sampling_rate: Частота дискретизации (Гц)
            temperature: Температура (°C)
            
        Returns:
            Dict[str, Any]: Результат предсказания на русском языке:
                - состояние: Название состояния
                - вероятность: Вероятность в диапазоне 0-1
                - рекомендация: Первая рекомендация
                - рекомендации: Список всех рекомендаций
                - метрики: Детальные метрики (вероятности всех классов)
        
        Raises:
            ValueError: Если модель не загружена
        """
        if not self.is_loaded or self.predictor is None:
            raise ValueError(
                self.error_message or "ML модель не загружена. "
                "Проверьте наличие файла модели и повторите попытку."
            )
        
        try:
            # Получаем предсказание от модели
            result = self.predictor.predict(
                vibration_x,
                vibration_y,
                vibration_z,
                sampling_rate,
                temperature,
                return_probabilities=True
            )
            
            # Формируем ответ в требуемом формате на русском языке
            response = {
                "состояние": result["status"],
                "вероятность": result["confidence"] / 100.0,  # Преобразуем проценты в 0-1
                "рекомендация": result["recommendations"][0] if result["recommendations"] else "Продолжить мониторинг",
                "рекомендации": result["recommendations"],
                "метрики": {
                    "уверенность_процентах": result["confidence"],
                    "вероятности_классов": result["probabilities"],
                    "код_состояния": result["status_code"]
                }
            }
            
            return response
            
        except Exception as e:
            raise ValueError(f"Ошибка при выполнении предсказания: {str(e)}")
    
    def predict_from_2d_array(
        self,
        vibration_data: List[List[float]],
        sampling_rate: float,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Предсказывает состояние подшипника из 2D массива вибрационных данных.
        
        Args:
            vibration_data: 2D массив [ось][значения]. 
                           Ожидается 3 оси: [X, Y, Z] или [X, Y] (Z будет пустым)
            sampling_rate: Частота дискретизации (Гц)
            temperature: Температура (°C)
            
        Returns:
            Dict[str, Any]: Результат предсказания в том же формате, что и predict()
        
        Raises:
            ValueError: Если формат данных некорректен или модель не загружена
        """
        # Проверяем формат данных
        if not vibration_data or len(vibration_data) == 0:
            raise ValueError("Массив вибрационных данных не может быть пустым")
        
        if len(vibration_data) < 2:
            raise ValueError("Требуется минимум 2 оси вибрации (X и Y)")
        
        # Извлекаем данные по осям
        vibration_x = vibration_data[0] if len(vibration_data) > 0 else []
        vibration_y = vibration_data[1] if len(vibration_data) > 1 else []
        vibration_z = vibration_data[2] if len(vibration_data) > 2 else []
        
        # Если Z не указан, создаем массив нулей той же длины
        if not vibration_z and len(vibration_x) > 0:
            vibration_z = [0.0] * len(vibration_x)
        
        # Проверяем, что массивы не пустые
        if not vibration_x or not vibration_y:
            raise ValueError("Массивы вибрации по осям X и Y не могут быть пустыми")
        
        return self.predict(
            vibration_x,
            vibration_y,
            vibration_z,
            sampling_rate,
            temperature
        )
    
    def is_available(self) -> bool:
        """
        Проверяет, доступна ли ML модель.
        
        Returns:
            bool: True если модель загружена и готова к использованию
        """
        return self.is_loaded and self.predictor is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает статус ML модели.
        
        Returns:
            Dict[str, Any]: Информация о статусе модели
        """
        return {
            "загружена": self.is_loaded,
            "доступна": self.is_available(),
            "путь_к_модели": self.model_path,
            "ошибка": self.error_message if not self.is_loaded else None
        }


# Глобальный экземпляр менеджера модели (загружается при старте приложения)
_ml_manager: Optional[MLModelManager] = None


def get_ml_manager() -> MLModelManager:
    """
    Получает глобальный экземпляр менеджера ML модели.
    
    Returns:
        MLModelManager: Экземпляр менеджера модели
    """
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModelManager()
    return _ml_manager


def initialize_ml_model(model_path: Optional[str] = None) -> MLModelManager:
    """
    Инициализирует ML модель при старте приложения.
    
    Args:
        model_path: Путь к файлу модели (опционально)
        
    Returns:
        MLModelManager: Экземпляр менеджера модели
    """
    global _ml_manager
    _ml_manager = MLModelManager(model_path=model_path)
    return _ml_manager


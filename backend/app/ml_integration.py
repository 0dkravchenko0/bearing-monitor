"""
Модуль интеграции ML модели для классификации неисправностей подшипников.

Предоставляет простой интерфейс для работы с обученной ML моделью
в FastAPI приложении.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Настройка логирования
logger = logging.getLogger(__name__)

# Добавляем путь к модулям ML модели
backend_dir = Path(__file__).parent.parent.parent
ml_model_dir = backend_dir / "ml_model"
sys.path.insert(0, str(ml_model_dir))

# Пытаемся импортировать модули ML модели
BearingPredictor = None
create_predictor = None
FeatureExtractor = None

try:
    from predictor import BearingPredictor, create_predictor
    from feature_extractor import FeatureExtractor
    logger.info("Модули ML модели успешно импортированы")
except ImportError:
    try:
        # Пробуем альтернативный путь через importlib
        import importlib.util
        
        # Загружаем predictor
        predictor_path = ml_model_dir / "predictor.py"
        if predictor_path.exists():
            spec = importlib.util.spec_from_file_location("predictor", str(predictor_path))
            if spec and spec.loader:
                predictor_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predictor_module)
                BearingPredictor = getattr(predictor_module, "BearingPredictor", None)
                create_predictor = getattr(predictor_module, "create_predictor", None)
        
        # Загружаем feature_extractor
        feature_extractor_path = ml_model_dir / "feature_extractor.py"
        if feature_extractor_path.exists():
            spec = importlib.util.spec_from_file_location("feature_extractor", str(feature_extractor_path))
            if spec and spec.loader:
                feature_extractor_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(feature_extractor_module)
                FeatureExtractor = getattr(feature_extractor_module, "FeatureExtractor", None)
        
        if BearingPredictor and create_predictor:
            logger.info("Модули ML модели загружены через importlib")
    except Exception as e:
        logger.warning(f"Не удалось загрузить модули ML модели: {e}")


class MLPredictor:
    """
    Класс для работы с ML моделью классификации неисправностей подшипников.
    
    Обеспечивает загрузку модели, извлечение признаков и предсказание состояния.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация предиктора и загрузка модели.
        
        Args:
            model_path: Путь к файлу модели. Если None, используется путь по умолчанию.
        
        Raises:
            FileNotFoundError: Если файл модели не найден
            ValueError: Если модель не может быть загружена
        """
        # Определяем путь к модели
        if model_path is None:
            model_path = str(ml_model_dir / "bearing_classifier_model.joblib")
        
        self.model_path = model_path
        self.predictor: Optional[BearingPredictor] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.is_loaded = False
        self.error_message: Optional[str] = None
        
        # Загружаем модель
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Загружает ML модель из файла.
        
        Returns:
            bool: True если модель успешно загружена, False в противном случае
        """
        try:
            # Проверяем существование файла модели
            if not os.path.exists(self.model_path):
                error_msg = f"Файл модели не найден: {self.model_path}"
                logger.error(error_msg)
                self.error_message = error_msg
                return False
            
            # Проверяем доступность классов
            if BearingPredictor is None or create_predictor is None:
                error_msg = "Модуль ML модели не найден. Убедитесь, что все зависимости установлены."
                logger.error(error_msg)
                self.error_message = error_msg
                return False
            
            # Инициализируем извлечение признаков
            if FeatureExtractor is None:
                logger.warning("FeatureExtractor не найден, будет использован встроенный в predictor")
            else:
                self.feature_extractor = FeatureExtractor()
            
            # Загружаем модель
            logger.info(f"Загрузка ML модели из {self.model_path}")
            self.predictor = create_predictor(model_path=self.model_path)
            self.is_loaded = True
            self.error_message = None
            logger.info("ML модель успешно загружена")
            return True
            
        except Exception as e:
            error_msg = f"Ошибка при загрузке ML модели: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_message = error_msg
            self.is_loaded = False
            return False
    
    def extract_features(
        self,
        vibration_data: List[List[float]],
        sampling_rate: float,
        temperature: float
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Извлекает признаки из вибрационных данных.
        
        Args:
            vibration_data: 2D массив вибрационных данных [ось][значения]
                           Ожидается минимум 2 оси: [X, Y] или [X, Y, Z]
            sampling_rate: Частота дискретизации (Гц)
            temperature: Температура (°C)
        
        Returns:
            Tuple[List[float], List[float], List[float]]: (vibration_x, vibration_y, vibration_z)
        
        Raises:
            ValueError: Если формат данных некорректен
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
        
        logger.debug(f"Извлечены признаки: X={len(vibration_x)}, Y={len(vibration_y)}, Z={len(vibration_z)}")
        
        return vibration_x, vibration_y, vibration_z
    
    def predict(
        self,
        vibration_data: List[List[float]],
        temperature: float,
        sampling_rate: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Основной метод предсказания состояния подшипника.
        
        Args:
            vibration_data: 2D массив вибрационных данных [ось][значения]
                           Ожидается минимум 2 оси: [X, Y] или [X, Y, Z]
            temperature: Температура (°C)
            sampling_rate: Частота дискретизации (Гц), по умолчанию 1000.0
        
        Returns:
            Dict[str, Any]: Результат предсказания на русском языке:
                - состояние: Название состояния
                - вероятность: Вероятность в диапазоне 0-1
                - рекомендация: Первая рекомендация
                - рекомендации: Список всех рекомендаций
                - метрики: Детальные метрики (вероятности всех классов)
        
        Raises:
            ValueError: Если модель не загружена или данные некорректны
        """
        if not self.is_loaded or self.predictor is None:
            raise ValueError(
                self.error_message or "ML модель не загружена. "
                "Проверьте наличие файла модели и повторите попытку."
            )
        
        try:
            # Извлекаем признаки
            vibration_x, vibration_y, vibration_z = self.extract_features(
                vibration_data,
                sampling_rate,
                temperature
            )
            
            # Получаем предсказание от модели
            logger.debug("Выполнение предсказания ML модели")
            result = self.predictor.predict(
                vibration_x,
                vibration_y,
                vibration_z,
                sampling_rate,
                temperature,
                return_probabilities=True
            )
            
            # Форматируем результат
            formatted_result = self.format_prediction(result, result.get("probabilities", {}))
            
            logger.info(f"Предсказание выполнено: {formatted_result['состояние']} "
                       f"(вероятность: {formatted_result['вероятность']:.2%})")
            
            return formatted_result
            
        except ValueError:
            # Пробрасываем ValueError как есть
            raise
        except Exception as e:
            error_msg = f"Ошибка при выполнении предсказания: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)
    
    def format_prediction(
        self,
        prediction: Dict[str, Any],
        probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Форматирует результат предсказания на русском языке.
        
        Args:
            prediction: Результат предсказания от модели
            probabilities: Словарь вероятностей всех классов
        
        Returns:
            Dict[str, Any]: Отформатированный результат:
                - состояние: Название состояния
                - вероятность: Вероятность в диапазоне 0-1
                - рекомендация: Первая рекомендация
                - рекомендации: Список всех рекомендаций
                - метрики: Детальные метрики
        """
        # Формируем ответ в требуемом формате на русском языке
        formatted = {
            "состояние": prediction.get("status", "неизвестно"),
            "вероятность": prediction.get("confidence", 0.0) / 100.0,  # Преобразуем проценты в 0-1
            "рекомендация": (
                prediction.get("recommendations", [])[0] 
                if prediction.get("recommendations") 
                else "Продолжить мониторинг"
            ),
            "рекомендации": prediction.get("recommendations", []),
            "метрики": {
                "уверенность_процентах": prediction.get("confidence", 0.0),
                "вероятности_классов": probabilities,
                "код_состояния": prediction.get("status_code", -1)
            }
        }
        
        return formatted


# ==================== Функции-утилиты ====================

# Глобальный экземпляр предиктора (singleton)
_predictor_instance: Optional[MLPredictor] = None


def load_model(model_path: Optional[str] = None) -> MLPredictor:
    """
    Загружает и кэширует ML модель.
    
    Args:
        model_path: Путь к файлу модели. Если None, используется путь по умолчанию.
    
    Returns:
        MLPredictor: Экземпляр предиктора с загруженной моделью
    
    Raises:
        FileNotFoundError: Если файл модели не найден
        ValueError: Если модель не может быть загружена
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        logger.info("Инициализация ML предиктора")
        _predictor_instance = MLPredictor(model_path=model_path)
        
        if not _predictor_instance.is_loaded:
            error_msg = _predictor_instance.error_message or "Неизвестная ошибка при загрузке модели"
            logger.error(f"Не удалось загрузить модель: {error_msg}")
            raise ValueError(error_msg)
    else:
        logger.debug("ML предиктор уже загружен, используется существующий экземпляр")
    
    return _predictor_instance


def get_predictor() -> Optional[MLPredictor]:
    """
    Получает singleton экземпляр предиктора.
    
    Returns:
        Optional[MLPredictor]: Экземпляр предиктора или None если модель не загружена
    """
    return _predictor_instance


def health_check() -> Dict[str, Any]:
    """
    Проверяет, что модель загружена и готова к использованию.
    
    Returns:
        Dict[str, Any]: Статус модели:
            - загружена: bool - загружена ли модель
            - доступна: bool - доступна ли модель для использования
            - путь_к_модели: str - путь к файлу модели
            - ошибка: Optional[str] - сообщение об ошибке если есть
    """
    predictor = get_predictor()
    
    if predictor is None:
        return {
            "загружена": False,
            "доступна": False,
            "путь_к_модели": str(ml_model_dir / "bearing_classifier_model.joblib"),
            "ошибка": "Модель не была загружена. Вызовите load_model() для загрузки."
        }
    
    return {
        "загружена": predictor.is_loaded,
        "доступна": predictor.is_loaded and predictor.predictor is not None,
        "путь_к_модели": predictor.model_path,
        "ошибка": predictor.error_message if not predictor.is_loaded else None
    }


# ==================== Тестовые примеры использования ====================

def example_usage():
    """
    Примеры использования MLPredictor.
    
    Демонстрирует основные возможности модуля.
    """
    print("=" * 60)
    print("Примеры использования MLPredictor")
    print("=" * 60)
    
    try:
        # Пример 1: Загрузка модели
        print("\n1. Загрузка модели...")
        predictor = load_model()
        print(f"✓ Модель загружена: {predictor.is_loaded}")
        
        # Пример 2: Проверка здоровья
        print("\n2. Проверка здоровья модели...")
        health = health_check()
        print(f"  Загружена: {health['загружена']}")
        print(f"  Доступна: {health['доступна']}")
        print(f"  Путь: {health['путь_к_модели']}")
        
        # Пример 3: Предсказание состояния
        print("\n3. Предсказание состояния подшипника...")
        vibration_data = [
            [0.1, 0.12, 0.11, 0.13, 0.09] * 10,  # Ось X
            [0.08, 0.10, 0.09, 0.11, 0.07] * 10,  # Ось Y
            [0.05, 0.06, 0.05, 0.07, 0.04] * 10   # Ось Z
        ]
        temperature = 45.5
        sampling_rate = 1000.0
        
        result = predictor.predict(
            vibration_data=vibration_data,
            temperature=temperature,
            sampling_rate=sampling_rate
        )
        
        print(f"  Состояние: {result['состояние']}")
        print(f"  Вероятность: {result['вероятность']:.2%}")
        print(f"  Рекомендация: {result['рекомендация']}")
        print(f"  Вероятности классов:")
        for class_name, prob in result['метрики']['вероятности_классов'].items():
            print(f"    - {class_name}: {prob:.2f}%")
        
        # Пример 4: Извлечение признаков
        print("\n4. Извлечение признаков...")
        vibration_x, vibration_y, vibration_z = predictor.extract_features(
            vibration_data=vibration_data,
            sampling_rate=sampling_rate,
            temperature=temperature
        )
        print(f"  Извлечено признаков:")
        print(f"    - Ось X: {len(vibration_x)} значений")
        print(f"    - Ось Y: {len(vibration_y)} значений")
        print(f"    - Ось Z: {len(vibration_z)} значений")
        
        # Пример 5: Использование singleton
        print("\n5. Использование singleton get_predictor()...")
        predictor2 = get_predictor()
        if predictor2 is predictor:
            print("  ✓ Получен тот же экземпляр (singleton работает)")
        else:
            print("  ✗ Получен другой экземпляр")
        
        print("\n" + "=" * 60)
        print("Все примеры выполнены успешно!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Ошибка при выполнении примеров: {e}")
        logger.error("Ошибка в примерах использования", exc_info=True)


if __name__ == "__main__":
    # Настройка логирования для тестов
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запускаем примеры
    example_usage()

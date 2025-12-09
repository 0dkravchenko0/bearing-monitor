"""
Модуль для логирования предсказаний ML модели.

Сохраняет все предсказания в файл для последующего анализа.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Класс для логирования предсказаний ML модели.
    
    Сохраняет предсказания в JSON файл с метаданными.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Инициализация логгера.
        
        Args:
            log_file: Путь к файлу логов. Если None, используется путь по умолчанию.
        """
        if log_file is None:
            # Создаем директорию logs если её нет
            backend_dir = Path(__file__).parent.parent
            logs_dir = backend_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            log_file = str(logs_dir / "predictions.jsonl")
        
        self.log_file = log_file
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Создает файл логов если его нет."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            log_path.touch()
            logger.info(f"Создан файл логов: {self.log_file}")
    
    def log_prediction(
        self,
        device_id: str,
        vibration_data: Dict[str, Any],
        prediction: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Логирует предсказание ML модели.
        
        Args:
            device_id: Идентификатор устройства
            vibration_data: Исходные данные вибрации
            prediction: Результат предсказания
            timestamp: Время предсказания (если None, используется текущее время)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "device_id": device_id,
            "vibration_data": {
                "vibration_x_count": len(vibration_data.get("vibration_x", [])),
                "vibration_y_count": len(vibration_data.get("vibration_y", [])),
                "vibration_z_count": len(vibration_data.get("vibration_z", [])),
                "sampling_rate": vibration_data.get("sampling_rate"),
                "temperature": vibration_data.get("temperature")
            },
            "prediction": prediction
        }
        
        try:
            # Записываем в формате JSONL (одна строка JSON на запись)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            logger.debug(f"Предсказание залогировано для устройства {device_id}")
        except Exception as e:
            logger.error(f"Ошибка при логировании предсказания: {e}", exc_info=True)
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Получает последние предсказания из лога.
        
        Args:
            limit: Максимальное количество записей
        
        Returns:
            List[Dict[str, Any]]: Список последних предсказаний
        """
        predictions = []
        
        try:
            if not os.path.exists(self.log_file):
                return predictions
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Берем последние limit строк
                for line in lines[-limit:]:
                    try:
                        predictions.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            # Сортируем по времени (новые сначала)
            predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Ошибка при чтении логов: {e}", exc_info=True)
        
        return predictions


# Глобальный экземпляр логгера
_prediction_logger: Optional[PredictionLogger] = None


def get_logger() -> PredictionLogger:
    """
    Получает singleton экземпляр логгера.
    
    Returns:
        PredictionLogger: Экземпляр логгера
    """
    global _prediction_logger
    if _prediction_logger is None:
        _prediction_logger = PredictionLogger()
    return _prediction_logger


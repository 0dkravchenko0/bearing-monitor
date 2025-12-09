"""
FastAPI приложение для мониторинга вибрации подшипников электродвигателей.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uuid
from contextlib import asynccontextmanager

# Импортируем модуль интеграции ML модели
from .ml_integration import load_model, get_predictor, health_check

# ==================== Инициализация ML модели при старте ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения.
    Загружает ML модель при старте и выгружает при остановке.
    """
    # Загрузка ML модели при старте
    print("Загрузка ML модели...")
    try:
        predictor = load_model()
        if predictor.is_loaded:
            print("✓ ML модель успешно загружена")
        else:
            print(f"⚠ ML модель не загружена: {predictor.error_message}")
    except Exception as e:
        print(f"⚠ Ошибка при загрузке ML модели: {e}")
    yield
    # Очистка при остановке (если нужна)
    print("Остановка приложения...")


# Инициализация FastAPI приложения
app = FastAPI(
    title="Система мониторинга вибрации подшипников",
    description="Веб-система для предиктивного анализа вибрации подшипников электродвигателей",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Модели данных Pydantic ====================

class VibrationDataRequest(BaseModel):
    """Модель для приема данных вибрации."""
    device_id: str = Field(..., description="Идентификатор устройства")
    timestamp: datetime = Field(..., description="Время замера")
    vibration_x: List[float] = Field(..., description="Массив значений вибрации по оси X")
    vibration_y: List[float] = Field(..., description="Массив значений вибрации по оси Y")
    vibration_z: List[float] = Field(..., description="Массив значений вибрации по оси Z")
    sampling_rate: float = Field(..., gt=0, description="Частота дискретизации (Гц)")
    temperature: float = Field(..., description="Температура (°C)")

    @validator('vibration_x', 'vibration_y', 'vibration_z')
    def validate_vibration_arrays(cls, v):
        """Проверка, что массивы вибрации не пустые."""
        if len(v) == 0:
            raise ValueError('Массив значений вибрации не может быть пустым')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "motor_001",
                "timestamp": "2024-01-15T10:30:00",
                "vibration_x": [0.1, 0.15, 0.12, 0.18],
                "vibration_y": [0.08, 0.11, 0.09, 0.13],
                "vibration_z": [0.05, 0.07, 0.06, 0.08],
                "sampling_rate": 1000.0,
                "temperature": 45.5
            }
        }


class VibrationDataResponse(BaseModel):
    """Модель ответа при сохранении данных вибрации."""
    id: str = Field(..., description="Уникальный идентификатор записи")
    device_id: str = Field(..., description="Идентификатор устройства")
    timestamp: datetime = Field(..., description="Время замера")
    message: str = Field(..., description="Сообщение о результате операции")


class DeviceStatus(BaseModel):
    """Модель статуса устройства."""
    device_id: str = Field(..., description="Идентификатор устройства")
    last_measurement: Optional[datetime] = Field(None, description="Время последнего замера")
    total_measurements: int = Field(..., description="Общее количество замеров")
    status: str = Field(..., description="Статус устройства (активно/неактивно)")


class StatusResponse(BaseModel):
    """Модель ответа для эндпоинта статуса."""
    system_status: str = Field(..., description="Статус системы")
    devices: List[DeviceStatus] = Field(..., description="Список устройств и их статусов")
    total_devices: int = Field(..., description="Общее количество устройств")


class HistoryEntry(BaseModel):
    """Модель записи истории измерений."""
    id: str = Field(..., description="Уникальный идентификатор записи")
    device_id: str = Field(..., description="Идентификатор устройства")
    timestamp: datetime = Field(..., description="Время замера")
    vibration_x_count: int = Field(..., description="Количество точек по оси X")
    vibration_y_count: int = Field(..., description="Количество точек по оси Y")
    vibration_z_count: int = Field(..., description="Количество точек по оси Z")
    sampling_rate: float = Field(..., description="Частота дискретизации")
    temperature: float = Field(..., description="Температура")


class HistoryResponse(BaseModel):
    """Модель ответа для истории измерений."""
    total: int = Field(..., description="Общее количество записей")
    entries: List[HistoryEntry] = Field(..., description="Список записей истории")


class ControlRequest(BaseModel):
    """Модель запроса для управления устройством."""
    device_id: str = Field(..., description="Идентификатор устройства")
    action: str = Field(..., description="Действие (start/stop/reset)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Дополнительные параметры")

    @validator('action')
    def validate_action(cls, v):
        """Проверка допустимых действий."""
        allowed_actions = ['start', 'stop', 'reset']
        if v not in allowed_actions:
            raise ValueError(f'Действие должно быть одним из: {", ".join(allowed_actions)}')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "motor_001",
                "action": "start",
                "parameters": {"frequency": 1000}
            }
        }


class ControlResponse(BaseModel):
    """Модель ответа для управления устройством."""
    device_id: str = Field(..., description="Идентификатор устройства")
    action: str = Field(..., description="Выполненное действие")
    status: str = Field(..., description="Статус выполнения")
    message: str = Field(..., description="Сообщение о результате")


class PredictRequest(BaseModel):
    """Модель запроса для предсказания состояния подшипника."""
    vibration_data: List[List[float]] = Field(
        ..., 
        description="2D массив вибрационных данных [ось][значения]. Ожидается минимум 2 оси (X, Y), опционально Z"
    )
    temperature: float = Field(..., description="Температура (°C)")
    sampling_rate: float = Field(..., gt=0, description="Частота дискретизации (Гц)")
    
    @validator('vibration_data')
    def validate_vibration_data(cls, v):
        """Проверка формата вибрационных данных."""
        if not v or len(v) == 0:
            raise ValueError('Массив вибрационных данных не может быть пустым')
        if len(v) < 2:
            raise ValueError('Требуется минимум 2 оси вибрации (X и Y)')
        # Проверяем, что все оси не пустые
        for i, axis in enumerate(v):
            if not axis or len(axis) == 0:
                raise ValueError(f'Ось {i} не может быть пустой')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "vibration_data": [
                    [0.1, 0.12, 0.11, 0.13, 0.09],  # Ось X
                    [0.08, 0.10, 0.09, 0.11, 0.07],  # Ось Y
                    [0.05, 0.06, 0.05, 0.07, 0.04]   # Ось Z (опционально)
                ],
                "temperature": 45.5,
                "sampling_rate": 1000.0
            }
        }


class PredictResponse(BaseModel):
    """Модель ответа для предсказания состояния подшипника."""
    состояние: str = Field(..., description="Название состояния на русском языке")
    вероятность: float = Field(..., ge=0.0, le=1.0, description="Вероятность предсказания (0-1)")
    рекомендация: str = Field(..., description="Первая рекомендация")
    рекомендации: List[str] = Field(..., description="Список всех рекомендаций")
    метрики: Dict[str, Any] = Field(..., description="Детальные метрики предсказания")


class VibrationDataResponseWithPrediction(BaseModel):
    """Модель ответа при сохранении данных вибрации с предсказанием ML."""
    id: str = Field(..., description="Уникальный идентификатор записи")
    device_id: str = Field(..., description="Идентификатор устройства")
    timestamp: datetime = Field(..., description="Время замера")
    message: str = Field(..., description="Сообщение о результате операции")
    ml_prediction: Optional[PredictResponse] = Field(
        None, 
        description="Предсказание ML модели (если модель доступна)"
    )


# ==================== Хранилище данных в памяти ====================

# Хранилище для данных вибрации
vibration_storage: List[Dict[str, Any]] = []

# Хранилище для статусов устройств
device_statuses: Dict[str, Dict[str, Any]] = {}


# ==================== Эндпоинты API ====================

@app.post(
    "/api/v1/vibration-data",
    response_model=VibrationDataResponseWithPrediction,
    status_code=status.HTTP_201_CREATED,
    summary="Прием данных вибрации",
    description="Эндпоинт для приема и сохранения данных вибрации подшипников. Автоматически вызывает ML модель для предсказания состояния."
)
async def receive_vibration_data(data: VibrationDataRequest):
    """
    Принимает данные вибрации от устройства, сохраняет их в памяти и автоматически
    вызывает ML модель для предсказания состояния подшипника.
    
    Args:
        data: Данные вибрации с валидацией через Pydantic
        
    Returns:
        VibrationDataResponseWithPrediction: Подтверждение сохранения с ID записи и предсказанием ML
    """
    try:
        # Генерируем уникальный ID для записи
        record_id = str(uuid.uuid4())
        
        # Сохраняем полные данные вибрации
        record = {
            "id": record_id,
            "device_id": data.device_id,
            "timestamp": data.timestamp,
            "vibration_x": data.vibration_x,
            "vibration_y": data.vibration_y,
            "vibration_z": data.vibration_z,
            "sampling_rate": data.sampling_rate,
            "temperature": data.temperature
        }
        vibration_storage.append(record)
        
        # Обновляем статус устройства
        if data.device_id not in device_statuses:
            device_statuses[data.device_id] = {
                "device_id": data.device_id,
                "last_measurement": data.timestamp,
                "total_measurements": 1,
                "status": "активно"
            }
        else:
            device_statuses[data.device_id]["last_measurement"] = data.timestamp
            device_statuses[data.device_id]["total_measurements"] += 1
            device_statuses[data.device_id]["status"] = "активно"
        
        # Автоматически вызываем ML модель для предсказания
        ml_prediction = None
        try:
            predictor = get_predictor()
            if predictor and predictor.is_loaded:
                # Преобразуем данные в формат 2D массива
                vibration_data = [
                    data.vibration_x,
                    data.vibration_y,
                    data.vibration_z
                ]
                prediction_result = predictor.predict(
                    vibration_data=vibration_data,
                    temperature=data.temperature,
                    sampling_rate=data.sampling_rate
                )
                ml_prediction = PredictResponse(**prediction_result)
        except Exception as ml_error:
            # Логируем ошибку, но не прерываем сохранение данных
            print(f"Ошибка при вызове ML модели: {str(ml_error)}")
            # Продолжаем выполнение без предсказания
        
        return VibrationDataResponseWithPrediction(
            id=record_id,
            device_id=data.device_id,
            timestamp=data.timestamp,
            message=f"Данные вибрации успешно сохранены для устройства {data.device_id}",
            ml_prediction=ml_prediction
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при сохранении данных: {str(e)}"
        )


@app.get(
    "/api/v1/status",
    response_model=StatusResponse,
    summary="Получение статуса системы",
    description="Возвращает текущий статус системы и всех устройств"
)
async def get_status():
    """
    Возвращает текущий статус системы мониторинга.
    
    Returns:
        StatusResponse: Статус системы и список устройств
    """
    try:
        # Определяем статус системы
        system_status = "работает" if len(vibration_storage) > 0 else "ожидает данных"
        
        # Формируем список статусов устройств
        devices_list = []
        for device_id, device_data in device_statuses.items():
            devices_list.append(DeviceStatus(**device_data))
        
        # Если устройств нет, но есть данные - создаем статусы из данных
        if len(devices_list) == 0 and len(vibration_storage) > 0:
            # Извлекаем уникальные device_id из хранилища
            unique_devices = set(record["device_id"] for record in vibration_storage)
            for device_id in unique_devices:
                device_records = [r for r in vibration_storage if r["device_id"] == device_id]
                last_record = max(device_records, key=lambda x: x["timestamp"])
                devices_list.append(DeviceStatus(
                    device_id=device_id,
                    last_measurement=last_record["timestamp"],
                    total_measurements=len(device_records),
                    status="активно"
                ))
        
        return StatusResponse(
            system_status=system_status,
            devices=devices_list,
            total_devices=len(device_statuses) if device_statuses else len(set(r["device_id"] for r in vibration_storage))
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статуса: {str(e)}"
        )


@app.get(
    "/api/v1/history",
    response_model=HistoryResponse,
    summary="История измерений",
    description="Возвращает историю всех измерений вибрации"
)
async def get_history(
    device_id: Optional[str] = None,
    limit: Optional[int] = 100
):
    """
    Возвращает историю измерений вибрации.
    
    Args:
        device_id: Опциональный фильтр по идентификатору устройства
        limit: Максимальное количество записей для возврата (по умолчанию 100)
        
    Returns:
        HistoryResponse: Список записей истории измерений
    """
    try:
        # Фильтруем по device_id если указан
        filtered_records = vibration_storage
        if device_id:
            filtered_records = [r for r in vibration_storage if r["device_id"] == device_id]
            if len(filtered_records) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Устройство с ID '{device_id}' не найдено"
                )
        
        # Сортируем по времени (новые сначала)
        sorted_records = sorted(
            filtered_records,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        # Ограничиваем количество записей
        limited_records = sorted_records[:limit] if limit else sorted_records
        
        # Формируем ответ
        entries = []
        for record in limited_records:
            entries.append(HistoryEntry(
                id=record["id"],
                device_id=record["device_id"],
                timestamp=record["timestamp"],
                vibration_x_count=len(record["vibration_x"]),
                vibration_y_count=len(record["vibration_y"]),
                vibration_z_count=len(record["vibration_z"]),
                sampling_rate=record["sampling_rate"],
                temperature=record["temperature"]
            ))
        
        return HistoryResponse(
            total=len(filtered_records),
            entries=entries
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении истории: {str(e)}"
        )


@app.post(
    "/api/v1/control",
    response_model=ControlResponse,
    summary="Управление устройством",
    description="Эндпоинт для управления устройством (запуск, остановка, сброс)"
)
async def control_device(control: ControlRequest):
    """
    Управляет устройством: запуск, остановка или сброс.
    
    Args:
        control: Запрос на управление устройством
        
    Returns:
        ControlResponse: Результат выполнения команды
    """
    try:
        device_id = control.device_id
        action = control.action
        
        # Проверяем существование устройства (если есть данные)
        if device_id not in device_statuses:
            # Проверяем в хранилище данных
            device_exists = any(r["device_id"] == device_id for r in vibration_storage)
            if not device_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Устройство с ID '{device_id}' не найдено"
                )
            # Создаем запись статуса для нового устройства
            device_statuses[device_id] = {
                "device_id": device_id,
                "last_measurement": None,
                "total_measurements": 0,
                "status": "неизвестно"
            }
        
        # Выполняем действие
        if action == "start":
            device_statuses[device_id]["status"] = "активно"
            message = f"Устройство {device_id} успешно запущено"
        elif action == "stop":
            device_statuses[device_id]["status"] = "остановлено"
            message = f"Устройство {device_id} успешно остановлено"
        elif action == "reset":
            device_statuses[device_id]["status"] = "сброшено"
            device_statuses[device_id]["total_measurements"] = 0
            message = f"Устройство {device_id} успешно сброшено"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестное действие: {action}"
            )
        
        return ControlResponse(
            device_id=device_id,
            action=action,
            status="успешно",
            message=message
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при управлении устройством: {str(e)}"
        )


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Предсказание состояния подшипника",
    description="Эндпоинт для получения предсказания состояния подшипника от ML модели по вибрационным данным"
)
async def predict_bearing_status(request: PredictRequest):
    """
    Предсказывает состояние подшипника по вибрационным данным используя ML модель.
    
    Args:
        request: Запрос с вибрационными данными в формате 2D массива
        
    Returns:
        PredictResponse: Результат предсказания на русском языке с вероятностями и рекомендациями
        
    Raises:
        HTTPException: Если ML модель не доступна или произошла ошибка
    """
    try:
        predictor = get_predictor()
        
        # Проверяем доступность модели
        if not predictor or not predictor.is_loaded:
            health = health_check()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ML модель не доступна: {health.get('ошибка', 'Модель не загружена')}"
            )
        
        # Выполняем предсказание
        prediction_result = predictor.predict(
            vibration_data=request.vibration_data,
            temperature=request.temperature,
            sampling_rate=request.sampling_rate
        )
        
        return PredictResponse(**prediction_result)
        
    except ValueError as e:
        # Ошибки валидации данных
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка валидации данных: {str(e)}"
        )
    except HTTPException:
        # Пробрасываем HTTP исключения как есть
        raise
    except Exception as e:
        # Прочие ошибки
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при выполнении предсказания: {str(e)}"
        )


@app.get("/api/v1/ml-status", summary="Статус ML модели")
async def get_ml_status():
    """
    Возвращает статус ML модели.
    
    Returns:
        Dict: Информация о статусе ML модели
    """
    return health_check()


@app.get("/", summary="Корневой эндпоинт")
async def root():
    """Корневой эндпоинт с информацией о API."""
    health = health_check()
    ml_status = "доступна" if health.get("доступна", False) else "недоступна"
    
    return {
        "message": "Система мониторинга вибрации подшипников",
        "version": "1.0.0",
        "ml_model": ml_status,
        "endpoints": {
            "vibration_data": "/api/v1/vibration-data",
            "predict": "/api/v1/predict",
            "status": "/api/v1/status",
            "ml_status": "/api/v1/ml-status",
            "history": "/api/v1/history",
            "control": "/api/v1/control",
            "docs": "/docs"
        }
    }


@app.get("/health", summary="Проверка здоровья системы")
async def health_check():
    """Эндпоинт для проверки работоспособности системы."""
    return {
        "status": "ok",
        "message": "Система работает нормально",
        "total_records": len(vibration_storage),
        "total_devices": len(device_statuses)
    }



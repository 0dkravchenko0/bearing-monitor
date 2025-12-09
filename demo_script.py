"""
Демонстрационный скрипт работы системы мониторинга подшипников.

Показывает работу ML модели на примерах:
- Нормальные данные → "Норма"
- Данные с неисправностью → соответствующий диагноз
"""

import requests
import numpy as np
from datetime import datetime
import time


API_BASE_URL = "http://localhost:8000"


def generate_vibration_data(scenario: str) -> dict:
    """
    Генерирует вибрационные данные для указанного сценария.
    
    Args:
        scenario: 'normal', 'inner_ring', 'outer_ring', 'ball_fault'
    
    Returns:
        dict: Данные вибрации
    """
    samples = 50
    base_time = datetime.now()
    
    if scenario == 'normal':
        # Нормальное состояние
        vibration_x = np.random.normal(0.05, 0.02, samples).tolist()
        vibration_y = np.random.normal(0.04, 0.015, samples).tolist()
        vibration_z = np.random.normal(0.03, 0.01, samples).tolist()
        temperature = 45.0
        expected = "норма"
    elif scenario == 'inner_ring':
        # Износ внутреннего кольца
        base_vibration = np.random.normal(0.12, 0.03, samples)
        periodic_spikes = 0.1 * np.sin(np.arange(samples) * 2 * np.pi / 10)
        vibration_x = (base_vibration + periodic_spikes + np.random.normal(0, 0.01, samples)).tolist()
        vibration_y = np.random.normal(0.10, 0.025, samples).tolist()
        vibration_z = np.random.normal(0.08, 0.02, samples).tolist()
        temperature = 50.0
        expected = "износ внутреннего кольца"
    elif scenario == 'outer_ring':
        # Износ внешнего кольца
        vibration_x = np.random.normal(0.15, 0.04, samples).tolist()
        vibration_y = np.random.normal(0.13, 0.035, samples).tolist()
        vibration_z = np.random.normal(0.12, 0.03, samples).tolist()
        temperature = 52.0
        expected = "износ внешнего кольца"
    else:  # ball_fault
        # Неисправность шарика
        base_vibration = np.random.normal(0.20, 0.05, samples)
        spikes = np.random.choice([0, 0.15], size=samples, p=[0.7, 0.3])
        vibration_x = (base_vibration + spikes + np.random.normal(0, 0.02, samples)).tolist()
        vibration_y = np.random.normal(0.18, 0.04, samples).tolist()
        vibration_z = np.random.normal(0.15, 0.035, samples).tolist()
        temperature = 55.0
        expected = "неисправность шарика"
    
    return {
        "device_id": "demo_motor_001",
        "timestamp": base_time.isoformat(),
        "vibration_x": vibration_x,
        "vibration_y": vibration_y,
        "vibration_z": vibration_z,
        "sampling_rate": 1000.0,
        "temperature": temperature,
        "expected": expected
    }


def print_prediction_result(data: dict, response: dict):
    """Выводит результат предсказания в красивом формате."""
    print("\n" + "=" * 80)
    print(f"СЦЕНАРИЙ: {data['expected'].upper()}")
    print("=" * 80)
    
    print(f"\n📊 Параметры вибрации:")
    print(f"   - Температура: {data['temperature']:.1f} °C")
    print(f"   - Частота дискретизации: {data['sampling_rate']:.0f} Гц")
    print(f"   - Количество точек: {len(data['vibration_x'])}")
    print(f"   - Максимальная вибрация X: {max(data['vibration_x']):.3f} мм/с")
    print(f"   - Максимальная вибрация Y: {max(data['vibration_y']):.3f} мм/с")
    print(f"   - Максимальная вибрация Z: {max(data['vibration_z']):.3f} мм/с")
    
    if "ml_prediction" in response:
        ml_pred = response["ml_prediction"]
        predicted = ml_pred.get("состояние", "неизвестно")
        probability = ml_pred.get("вероятность", 0.0)
        confidence = ml_pred.get("метрики", {}).get("уверенность_процентах", 0.0)
        recommendations = ml_pred.get("рекомендации", [])
        
        print(f"\n🤖 Результат ML модели:")
        print(f"   - Предсказанное состояние: {predicted}")
        print(f"   - Вероятность: {probability:.2%}")
        print(f"   - Уверенность: {confidence:.2f}%")
        
        # Проверка корректности
        is_correct = predicted == data['expected']
        status_icon = "✅" if is_correct else "❌"
        print(f"   - Статус: {status_icon} {'ПРАВИЛЬНО' if is_correct else 'НЕПРАВИЛЬНО'}")
        
        if recommendations:
            print(f"\n💡 Рекомендации:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        # Вероятности всех классов
        probabilities = ml_pred.get("метрики", {}).get("вероятности_классов", {})
        if probabilities:
            print(f"\n📈 Вероятности всех классов:")
            for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                bar_length = int(prob / 2)  # Масштабируем для визуализации
                bar = "█" * bar_length
                print(f"   - {class_name:30s}: {prob:6.2f}% {bar}")
    else:
        print("\n⚠️ ML предсказание недоступно")
    
    print("\n" + "-" * 80)


def check_api_health():
    """Проверяет доступность API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Получает информацию о модели."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def main():
    """Основная функция демонстрации."""
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ СИСТЕМЫ МОНИТОРИНГА ПОДШИПНИКОВ")
    print("=" * 80)
    
    # Проверка доступности API
    print("\n🔍 Проверка доступности API...")
    if not check_api_health():
        print("❌ API недоступен. Убедитесь, что сервер запущен на http://localhost:8000")
        return
    
    print("✅ API доступен")
    
    # Информация о модели
    print("\n📋 Информация о ML модели:")
    model_info = get_model_info()
    if model_info:
        print(f"   - Загружена: {'Да' if model_info.get('загружена') else 'Нет'}")
        print(f"   - Количество классов: {model_info.get('количество_классов', 'N/A')}")
        if model_info.get('точность_обучения'):
            print(f"   - Точность обучения: {model_info.get('точность_обучения'):.2%}")
        print(f"   - Версия: {model_info.get('версия_модели', 'N/A')}")
    else:
        print("   ⚠️ Не удалось получить информацию о модели")
    
    # Демонстрация различных сценариев
    scenarios = [
        ('normal', 'Нормальное состояние'),
        ('inner_ring', 'Износ внутреннего кольца'),
        ('outer_ring', 'Износ внешнего кольца'),
        ('ball_fault', 'Неисправность шарика')
    ]
    
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ ML МОДЕЛИ")
    print("=" * 80)
    
    for scenario, description in scenarios:
        print(f"\n⏳ Генерация данных для: {description}...")
        data = generate_vibration_data(scenario)
        
        try:
            # Отправляем данные на API
            response = requests.post(
                f"{API_BASE_URL}/api/v1/vibration-data",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Выводим результат
            print_prediction_result(data, result)
            
            # Небольшая пауза между запросами
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Ошибка при отправке данных: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    print("\n💡 Для полного тестирования запустите: python test_integration.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


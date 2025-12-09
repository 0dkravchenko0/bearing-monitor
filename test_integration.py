"""
Скрипт для тестирования полной интеграции ML модели с бэкендом.

Генерирует тестовые данные для всех 4 состояний, отправляет их на API
и проверяет корректность предсказаний.
"""

import requests
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path


# Конфигурация
API_BASE_URL = "http://localhost:8000"
TEST_REPORT_FILE = "test_report.md"


def generate_test_data_for_class(class_id: int, n_samples: int = 10) -> List[Dict]:
    """
    Генерирует тестовые вибрационные данные для указанного класса.
    
    Args:
        class_id: ID класса (0-норма, 1-внутреннее кольцо, 2-внешнее кольцо, 3-шарик)
        n_samples: Количество образцов для генерации
    
    Returns:
        List[Dict]: Список тестовых данных
    """
    test_data = []
    
    for i in range(n_samples):
        samples = 50
        base_time = datetime.now()
        
        if class_id == 0:  # Норма
            vibration_x = np.random.normal(0.05, 0.02, samples).tolist()
            vibration_y = np.random.normal(0.04, 0.015, samples).tolist()
            vibration_z = np.random.normal(0.03, 0.01, samples).tolist()
            temperature = np.random.normal(45, 5)
        elif class_id == 1:  # Износ внутреннего кольца
            base_vibration = np.random.normal(0.12, 0.03, samples)
            periodic_spikes = 0.1 * np.sin(np.arange(samples) * 2 * np.pi / 10)
            vibration_x = (base_vibration + periodic_spikes + np.random.normal(0, 0.01, samples)).tolist()
            vibration_y = np.random.normal(0.10, 0.025, samples).tolist()
            vibration_z = np.random.normal(0.08, 0.02, samples).tolist()
            temperature = np.random.normal(50, 6)
        elif class_id == 2:  # Износ внешнего кольца
            vibration_x = np.random.normal(0.15, 0.04, samples).tolist()
            vibration_y = np.random.normal(0.13, 0.035, samples).tolist()
            vibration_z = np.random.normal(0.12, 0.03, samples).tolist()
            temperature = np.random.normal(52, 7)
        else:  # Неисправность шарика
            base_vibration = np.random.normal(0.20, 0.05, samples)
            spikes = np.random.choice([0, 0.15], size=samples, p=[0.7, 0.3])
            vibration_x = (base_vibration + spikes + np.random.normal(0, 0.02, samples)).tolist()
            vibration_y = np.random.normal(0.18, 0.04, samples).tolist()
            vibration_z = np.random.normal(0.15, 0.035, samples).tolist()
            temperature = np.random.normal(55, 8)
        
        test_data.append({
            "device_id": f"test_motor_{class_id}_{i}",
            "timestamp": base_time.isoformat(),
            "vibration_x": vibration_x,
            "vibration_y": vibration_y,
            "vibration_z": vibration_z,
            "sampling_rate": 1000.0,
            "temperature": float(temperature),
            "expected_class": class_id,
            "expected_class_name": ["норма", "износ внутреннего кольца", 
                                   "износ внешнего кольца", "неисправность шарика"][class_id]
        })
    
    return test_data


def send_vibration_data(data: Dict) -> Tuple[bool, Dict, str]:
    """
    Отправляет данные вибрации на API и получает предсказание.
    
    Args:
        data: Данные вибрации
    
    Returns:
        Tuple[bool, Dict, str]: (успех, результат, ошибка)
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/vibration-data",
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return True, response.json(), ""
    except requests.exceptions.RequestException as e:
        return False, {}, str(e)


def get_model_info() -> Dict:
    """Получает информацию о модели."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model-info", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def check_prediction_accuracy(prediction: Dict, expected_class: int, expected_class_name: str) -> Tuple[bool, float]:
    """
    Проверяет точность предсказания.
    
    Args:
        prediction: Результат предсказания от API
        expected_class: Ожидаемый класс (0-3)
        expected_class_name: Ожидаемое название класса
    
    Returns:
        Tuple[bool, float]: (корректно ли предсказано, вероятность)
    """
    if not prediction or "ml_prediction" not in prediction:
        return False, 0.0
    
    ml_pred = prediction["ml_prediction"]
    predicted_state = ml_pred.get("состояние", "")
    probability = ml_pred.get("вероятность", 0.0)
    
    # Проверяем соответствие
    is_correct = predicted_state == expected_class_name
    
    return is_correct, probability


def run_integration_test() -> Dict:
    """
    Запускает полный тест интеграции.
    
    Returns:
        Dict: Результаты тестирования
    """
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ ML МОДЕЛИ С БЭКЕНДОМ")
    print("=" * 80)
    
    # Проверяем доступность API
    print("\n1. Проверка доступности API...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ API доступен")
        else:
            print(f"   ✗ API вернул код {response.status_code}")
            return {"error": "API недоступен"}
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Ошибка подключения к API: {e}")
        return {"error": f"Не удалось подключиться к API: {e}"}
    
    # Получаем информацию о модели
    print("\n2. Получение информации о модели...")
    model_info = get_model_info()
    if "error" not in model_info:
        print(f"   ✓ Модель загружена: {model_info.get('загружена', False)}")
        print(f"   ✓ Количество классов: {model_info.get('количество_классов', 'N/A')}")
        print(f"   ✓ Точность обучения: {model_info.get('точность_обучения', 'N/A')}")
    else:
        print(f"   ⚠ Не удалось получить информацию о модели: {model_info['error']}")
    
    # Генерируем тестовые данные для всех классов
    print("\n3. Генерация тестовых данных...")
    all_test_data = []
    class_names = ["норма", "износ внутреннего кольца", "износ внешнего кольца", "неисправность шарика"]
    
    for class_id in range(4):
        test_data = generate_test_data_for_class(class_id, n_samples=10)
        all_test_data.extend(test_data)
        print(f"   ✓ Сгенерировано {len(test_data)} образцов для класса '{class_names[class_id]}'")
    
    # Отправляем данные и проверяем предсказания
    print("\n4. Отправка данных и проверка предсказаний...")
    results = {
        "total": len(all_test_data),
        "correct": 0,
        "incorrect": 0,
        "no_prediction": 0,
        "by_class": {name: {"total": 0, "correct": 0, "incorrect": 0} for name in class_names},
        "details": []
    }
    
    for i, test_data in enumerate(all_test_data):
        expected_class = test_data["expected_class"]
        expected_name = test_data["expected_class_name"]
        
        # Удаляем служебные поля перед отправкой
        data_to_send = {k: v for k, v in test_data.items() 
                       if k not in ["expected_class", "expected_class_name"]}
        
        success, response, error = send_vibration_data(data_to_send)
        
        if not success:
            results["no_prediction"] += 1
            results["by_class"][expected_name]["total"] += 1
            results["details"].append({
                "test_id": i + 1,
                "expected": expected_name,
                "status": "error",
                "error": error
            })
            print(f"   ✗ Тест {i+1}: Ошибка отправки - {error}")
            continue
        
        is_correct, probability = check_prediction_accuracy(
            response, expected_class, expected_name
        )
        
        results["by_class"][expected_name]["total"] += 1
        if is_correct:
            results["correct"] += 1
            results["by_class"][expected_name]["correct"] += 1
            status_icon = "✓"
        else:
            results["incorrect"] += 1
            results["by_class"][expected_name]["incorrect"] += 1
            status_icon = "✗"
            if "ml_prediction" in response:
                predicted = response["ml_prediction"].get("состояние", "неизвестно")
            else:
                predicted = "нет предсказания"
        
        results["details"].append({
            "test_id": i + 1,
            "expected": expected_name,
            "predicted": response.get("ml_prediction", {}).get("состояние", "нет предсказания") if "ml_prediction" in response else "нет предсказания",
            "correct": is_correct,
            "probability": probability,
            "status": "success" if is_correct else "incorrect"
        })
        
        if (i + 1) % 10 == 0:
            print(f"   Обработано {i+1}/{len(all_test_data)} тестов...")
    
    # Вычисляем точность
    accuracy = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 80)
    print(f"\nОбщая точность: {accuracy:.2f}%")
    print(f"Правильных предсказаний: {results['correct']}/{results['total']}")
    print(f"Неправильных предсказаний: {results['incorrect']}/{results['total']}")
    print(f"Ошибок/нет предсказания: {results['no_prediction']}/{results['total']}")
    
    print("\nТочность по классам:")
    for class_name, stats in results["by_class"].items():
        if stats["total"] > 0:
            class_accuracy = (stats["correct"] / stats["total"] * 100)
            print(f"  {class_name}: {class_accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    results["accuracy"] = accuracy
    results["model_info"] = model_info
    results["test_date"] = datetime.now().isoformat()
    
    return results


def generate_report(results: Dict, output_file: str):
    """
    Генерирует отчет в формате Markdown.
    
    Args:
        results: Результаты тестирования
        output_file: Путь к файлу отчета
    """
    report_lines = [
        "# Отчет о тестировании интеграции ML модели",
        "",
        f"**Дата тестирования:** {results.get('test_date', 'N/A')}",
        "",
        "## Общие результаты",
        "",
        f"- **Общая точность:** {results.get('accuracy', 0):.2f}%",
        f"- **Всего тестов:** {results.get('total', 0)}",
        f"- **Правильных предсказаний:** {results.get('correct', 0)}",
        f"- **Неправильных предсказаний:** {results.get('incorrect', 0)}",
        f"- **Ошибок/нет предсказания:** {results.get('no_prediction', 0)}",
        "",
        "## Информация о модели",
        ""
    ]
    
    model_info = results.get("model_info", {})
    if "error" not in model_info:
        report_lines.extend([
            f"- **Загружена:** {'Да' if model_info.get('загружена') else 'Нет'}",
            f"- **Количество классов:** {model_info.get('количество_классов', 'N/A')}",
            f"- **Точность обучения:** {model_info.get('точность_обучения', 'N/A')}",
            f"- **Дата обучения:** {model_info.get('дата_обучения', 'N/A')}",
            f"- **Версия модели:** {model_info.get('версия_модели', 'N/A')}",
            f"- **Размер файла:** {model_info.get('размер_файла_мб', 'N/A'):.2f} МБ" if model_info.get('размер_файла_мб') else "- **Размер файла:** N/A",
            ""
        ])
    else:
        report_lines.append(f"- **Ошибка:** {model_info['error']}\n")
    
    report_lines.extend([
        "## Точность по классам",
        "",
        "| Класс | Точность | Правильно | Всего |",
        "|-------|----------|-----------|-------|"
    ])
    
    for class_name, stats in results.get("by_class", {}).items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"] * 100)
            report_lines.append(
                f"| {class_name} | {accuracy:.2f}% | {stats['correct']} | {stats['total']} |"
            )
    
    report_lines.extend([
        "",
        "## Детали тестов",
        "",
        "| № | Ожидаемый класс | Предсказанный класс | Правильно | Вероятность |",
        "|---|-----------------|---------------------|-----------|-------------|"
    ])
    
    for detail in results.get("details", [])[:50]:  # Показываем первые 50
        correct_icon = "✓" if detail.get("correct") else "✗"
        predicted = detail.get("predicted", "N/A")
        probability = detail.get("probability", 0)
        report_lines.append(
            f"| {detail.get('test_id', 'N/A')} | {detail.get('expected', 'N/A')} | "
            f"{predicted} | {correct_icon} | {probability:.2%} |"
        )
    
    if len(results.get("details", [])) > 50:
        report_lines.append(f"\n*Показаны первые 50 из {len(results['details'])} тестов*")
    
    # Сохраняем отчет
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Отчет сохранен в {output_file}")


def main():
    """Основная функция."""
    try:
        results = run_integration_test()
        
        if "error" in results:
            print(f"\n✗ Тестирование не завершено: {results['error']}")
            return
        
        # Генерируем отчет
        print("\n5. Генерация отчета...")
        generate_report(results, TEST_REPORT_FILE)
        
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nТестирование прервано пользователем")
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


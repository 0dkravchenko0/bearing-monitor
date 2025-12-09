"""
Модуль для извлечения признаков из вибрационных данных подшипников.

Этот модуль содержит функции для преобразования сырых данных вибрации
в числовые признаки, которые используются для обучения и предсказания ML модели.
"""

import numpy as np
from typing import List, Dict, Any
from scipy import stats
from scipy.fft import fft


class FeatureExtractor:
    """
    Класс для извлечения признаков из вибрационных данных.
    
    Извлекает статистические, временные и частотные признаки
    из массивов вибрации по осям X, Y, Z.
    """
    
    def __init__(self):
        """Инициализация экстрактора признаков."""
        pass
    
    def extract_features(
        self,
        vibration_x: List[float],
        vibration_y: List[float],
        vibration_z: List[float],
        sampling_rate: float,
        temperature: float
    ) -> np.ndarray:
        """
        Извлекает признаки из вибрационных данных.
        
        Args:
            vibration_x: Массив значений вибрации по оси X
            vibration_y: Массив значений вибрации по оси Y
            vibration_z: Массив значений вибрации по оси Z
            sampling_rate: Частота дискретизации (Гц)
            temperature: Температура (°C)
            
        Returns:
            np.ndarray: Массив извлеченных признаков
        """
        # Преобразуем списки в numpy массивы для удобства вычислений
        x = np.array(vibration_x)
        y = np.array(vibration_y)
        z = np.array(vibration_z)
        
        # Объединенный массив всех осей для общих статистик
        all_axes = np.concatenate([x, y, z])
        
        features = []
        
        # ========== СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ПО КАЖДОЙ ОСИ ==========
        
        # Признаки для оси X
        features.extend(self._extract_statistical_features(x))
        # Признаки для оси Y
        features.extend(self._extract_statistical_features(y))
        # Признаки для оси Z
        features.extend(self._extract_statistical_features(z))
        
        # ========== ОБЩИЕ СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ==========
        features.extend(self._extract_statistical_features(all_axes))
        
        # ========== ЧАСТОТНЫЕ ПРИЗНАКИ ==========
        
        # Признаки из частотной области для каждой оси
        features.extend(self._extract_frequency_features(x, sampling_rate))
        features.extend(self._extract_frequency_features(y, sampling_rate))
        features.extend(self._extract_frequency_features(z, sampling_rate))
        
        # ========== ПРИЗНАКИ ВЗАИМОДЕЙСТВИЯ ОСЕЙ ==========
        
        # Корреляции между осями
        features.append(np.corrcoef(x, y)[0, 1])  # Корреляция X-Y
        features.append(np.corrcoef(x, z)[0, 1])  # Корреляция X-Z
        features.append(np.corrcoef(y, z)[0, 1])  # Корреляция Y-Z
        
        # ========== ТЕМПЕРАТУРНЫЕ ПРИЗНАКИ ==========
        features.append(temperature)
        features.append(temperature / 100.0)  # Нормализованная температура
        
        # Преобразуем в numpy массив и заменяем NaN и Inf на 0
        features_array = np.array(features)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    def _extract_statistical_features(self, data: np.ndarray) -> List[float]:
        """
        Извлекает статистические признаки из массива данных.
        
        Args:
            data: Массив значений вибрации
            
        Returns:
            List[float]: Список статистических признаков
        """
        if len(data) == 0:
            return [0.0] * 10  # Возвращаем нули если данных нет
        
        features = []
        
        # Базовые статистики
        features.append(np.mean(data))           # Среднее значение
        features.append(np.std(data))            # Стандартное отклонение
        features.append(np.var(data))           # Дисперсия
        features.append(np.max(data))            # Максимальное значение
        features.append(np.min(data))            # Минимальное значение
        features.append(np.max(data) - np.min(data))  # Размах (peak-to-peak)
        
        # RMS (Root Mean Square) - эффективное значение
        features.append(np.sqrt(np.mean(data**2)))
        
        # Асимметрия (skewness) - мера асимметрии распределения
        features.append(stats.skew(data))
        
        # Эксцесс (kurtosis) - мера остроты пиков
        features.append(stats.kurtosis(data))
        
        # Медиана
        features.append(np.median(data))
        
        return features
    
    def _extract_frequency_features(
        self,
        data: np.ndarray,
        sampling_rate: float
    ) -> List[float]:
        """
        Извлекает признаки из частотной области (FFT анализ).
        
        Args:
            data: Массив значений вибрации
            sampling_rate: Частота дискретизации (Гц)
            
        Returns:
            List[float]: Список частотных признаков
        """
        if len(data) == 0:
            return [0.0] * 8
        
        # Вычисляем FFT (быстрое преобразование Фурье)
        fft_values = fft(data)
        fft_magnitude = np.abs(fft_values)
        
        # Частоты для каждой компоненты FFT
        frequencies = np.fft.fftfreq(len(data), 1.0 / sampling_rate)
        
        # Берем только положительные частоты (симметрия FFT)
        positive_freq_idx = frequencies >= 0
        fft_magnitude = fft_magnitude[positive_freq_idx]
        frequencies = frequencies[positive_freq_idx]
        
        features = []
        
        # Доминирующая частота (частота с максимальной амплитудой)
        if len(fft_magnitude) > 0:
            dominant_freq_idx = np.argmax(fft_magnitude)
            features.append(frequencies[dominant_freq_idx])
            features.append(fft_magnitude[dominant_freq_idx])
        else:
            features.extend([0.0, 0.0])
        
        # Средняя амплитуда в частотной области
        features.append(np.mean(fft_magnitude))
        
        # Максимальная амплитуда в частотной области
        features.append(np.max(fft_magnitude) if len(fft_magnitude) > 0 else 0.0)
        
        # Энергия в частотной области (сумма квадратов амплитуд)
        features.append(np.sum(fft_magnitude**2))
        
        # Стандартное отклонение амплитуд
        features.append(np.std(fft_magnitude))
        
        # Количество пиков выше среднего
        if len(fft_magnitude) > 0:
            threshold = np.mean(fft_magnitude)
            peaks_count = np.sum(fft_magnitude > threshold)
            features.append(peaks_count)
        else:
            features.append(0.0)
        
        # Процент энергии в верхних 10% частот
        if len(fft_magnitude) > 0:
            sorted_magnitude = np.sort(fft_magnitude)[::-1]
            top_10_percent = int(len(sorted_magnitude) * 0.1)
            if top_10_percent > 0:
                top_energy = np.sum(sorted_magnitude[:top_10_percent]**2)
                total_energy = np.sum(sorted_magnitude**2)
                features.append(top_energy / total_energy if total_energy > 0 else 0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features


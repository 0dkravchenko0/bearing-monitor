/**
 * Утилиты для форматирования данных на русском языке
 */

/**
 * Форматирует число с запятой в качестве разделителя десятичных
 */
export const formatNumber = (value: number, decimals: number = 2): string => {
  return value.toFixed(decimals).replace('.', ',');
};

/**
 * Форматирует дату и время на русском языке
 */
export const formatDateTime = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

/**
 * Получает цвет статуса для индикации
 */
export const getStatusColor = (status: string): 'success' | 'warning' | 'error' | 'info' => {
  const statusLower = status.toLowerCase();
  if (statusLower.includes('норма') || statusLower.includes('активно') || statusLower === 'normal') {
    return 'success';
  }
  if (statusLower.includes('предупреждение') || statusLower.includes('warning')) {
    return 'warning';
  }
  if (statusLower.includes('критично') || statusLower.includes('critical') || statusLower.includes('остановлено')) {
    return 'error';
  }
  return 'info';
};



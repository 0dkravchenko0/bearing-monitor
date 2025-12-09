import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Chip,
  Paper,
  Alert,
  CircularProgress,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings,
  Refresh,
  Info
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import axios from 'axios';
import { VibrationData, MLDiagnosis, MonitoringStatus, MLPrediction } from '../types/vibration';
import { formatNumber, formatDateTime, getStatusColor } from '../utils/formatters';

// Регистрация компонентов Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Компонент Dashboard для мониторинга вибрации подшипников
 */
const Dashboard: React.FC = () => {
  const [monitoringStatus, setMonitoringStatus] = useState<MonitoringStatus>('stopped');
  const [vibrationData, setVibrationData] = useState<VibrationData | null>(null);
  const [mlDiagnosis, setMlDiagnosis] = useState<MLDiagnosis | null>(null);
  const [mlPrediction, setMlPrediction] = useState<MLPrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingML, setIsLoadingML] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);
  const [mlDetailsOpen, setMlDetailsOpen] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Симуляция данных вибрации (в реальном приложении данные приходят с бэкенда)
  const generateMockData = (): VibrationData => {
    const now = new Date();
    const samples = 50;
    const baseX = 0.1 + Math.random() * 0.1;
    const baseY = 0.08 + Math.random() * 0.1;
    const baseZ = 0.05 + Math.random() * 0.08;

    return {
      device_id: 'motor_001',
      timestamp: now.toISOString(),
      vibration_x: Array.from({ length: samples }, () => baseX + (Math.random() - 0.5) * 0.05),
      vibration_y: Array.from({ length: samples }, () => baseY + (Math.random() - 0.5) * 0.05),
      vibration_z: Array.from({ length: samples }, () => baseZ + (Math.random() - 0.5) * 0.03),
      sampling_rate: 1000,
      temperature: 45 + Math.random() * 5
    };
  };

  // Симуляция диагноза ML модели
  const generateMLDiagnosis = (data: VibrationData): MLDiagnosis => {
    const maxVibration = Math.max(
      ...data.vibration_x,
      ...data.vibration_y,
      ...data.vibration_z
    );

    if (maxVibration > 0.25) {
      return {
        status: 'critical',
        message: 'Критическое состояние: обнаружена сильная вибрация. Требуется немедленное обслуживание.',
        confidence: 0.95,
        recommendations: [
          'Остановить оборудование',
          'Провести визуальный осмотр подшипников',
          'Вызвать специалиста по обслуживанию'
        ]
      };
    } else if (maxVibration > 0.18) {
      return {
        status: 'warning',
        message: 'Предупреждение: повышенный уровень вибрации. Рекомендуется плановое обслуживание.',
        confidence: 0.85,
        recommendations: [
          'Усилить мониторинг',
          'Запланировать обслуживание в ближайшее время'
        ]
      };
    } else {
      return {
        status: 'normal',
        message: 'Нормальное состояние: вибрация в пределах допустимых значений.',
        confidence: 0.92,
        recommendations: ['Продолжить мониторинг']
      };
    }
  };

  // Загрузка ML предсказания
  const fetchMLPrediction = async (data: VibrationData) => {
    try {
      setIsLoadingML(true);
      setMlError(null);

      // Формируем данные для запроса
      const vibrationDataArray = [
        data.vibration_x,
        data.vibration_y,
        data.vibration_z
      ];

      const response = await axios.post<MLPrediction>(
        `${API_BASE_URL}/api/v1/predict`,
        {
          vibration_data: vibrationDataArray,
          temperature: data.temperature,
          sampling_rate: data.sampling_rate
        }
      );

      setMlPrediction(response.data);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Ошибка при получении предсказания ML модели';
      setMlError(errorMessage);
      console.error('Ошибка ML предсказания:', err);
      // Не сбрасываем mlPrediction, оставляем предыдущее значение
    } finally {
      setIsLoadingML(false);
    }
  };

  // Загрузка данных вибрации
  const fetchVibrationData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Симуляция для демонстрации (можно заменить на реальный API)
      const mockData = generateMockData();
      setVibrationData(mockData);
      setMlDiagnosis(generateMLDiagnosis(mockData));

      // Отправка данных на сервер и получение ML предсказания
      try {
        const response = await axios.post(
          `${API_BASE_URL}/api/v1/vibration-data`,
          {
            device_id: mockData.device_id,
            timestamp: mockData.timestamp,
            vibration_x: mockData.vibration_x,
            vibration_y: mockData.vibration_y,
            vibration_z: mockData.vibration_z,
            sampling_rate: mockData.sampling_rate,
            temperature: mockData.temperature
          }
        );

        // Если в ответе есть ml_prediction, используем его
        if (response.data.ml_prediction) {
          setMlPrediction(response.data.ml_prediction);
        } else {
          // Иначе запрашиваем отдельно
          await fetchMLPrediction(mockData);
        }
      } catch (apiErr) {
        // Если API недоступен, используем симуляцию
        console.warn('API недоступен, используется симуляция:', apiErr);
        // Можно также вызвать fetchMLPrediction для отдельного запроса
      }
    } catch (err) {
      setError('Ошибка при загрузке данных вибрации');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Ручное обновление ML диагностики
  const handleRefreshMLDiagnosis = async () => {
    if (vibrationData) {
      await fetchMLPrediction(vibrationData);
    }
  };

  // Управление мониторингом
  const handleStartMonitoring = async () => {
    try {
      setMonitoringStatus('running');
      setError(null);

      // Запуск периодического обновления данных
      intervalRef.current = setInterval(() => {
        fetchVibrationData();
      }, 2000); // Обновление каждые 2 секунды

      // Первая загрузка данных
      await fetchVibrationData();
    } catch (err) {
      setError('Ошибка при запуске мониторинга');
      console.error(err);
    }
  };

  const handleStopMonitoring = () => {
    setMonitoringStatus('stopped');
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const handleCalibration = async () => {
    try {
      setMonitoringStatus('calibrating');
      setError(null);
      setIsLoading(true);

      // Симуляция калибровки
      await new Promise(resolve => setTimeout(resolve, 2000));

      // В реальном приложении здесь будет запрос к API
      // await axios.post(`${API_BASE_URL}/api/v1/control`, {
      //   device_id: 'motor_001',
      //   action: 'reset'
      // });

      setMonitoringStatus('stopped');
      alert('Калибровка завершена успешно');
    } catch (err) {
      setError('Ошибка при калибровке');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Автоматическое получение ML предсказания при изменении данных вибрации
  useEffect(() => {
    if (vibrationData && monitoringStatus === 'running' && !isLoadingML) {
      fetchMLPrediction(vibrationData);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vibrationData?.timestamp, monitoringStatus]);

  // Очистка интервала при размонтировании
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Подготовка данных для графика
  const getChartData = () => {
    if (!vibrationData) return null;

    const timeLabels = vibrationData.vibration_x.map((_, index) => 
      `${(index / vibrationData.sampling_rate * 1000).toFixed(0)} мс`
    );

    return {
      labels: timeLabels,
      datasets: [
        {
          label: 'Ось X',
          data: vibrationData.vibration_x,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          tension: 0.4,
          fill: true
        },
        {
          label: 'Ось Y',
          data: vibrationData.vibration_y,
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          tension: 0.4,
          fill: true
        },
        {
          label: 'Ось Z',
          data: vibrationData.vibration_z,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          tension: 0.4,
          fill: true
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: 'График вибрации в реальном времени',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            return `${context.dataset.label}: ${formatNumber(context.parsed.y)} мм/с`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Время, мс'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Вибрация, мм/с'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: (value: any) => formatNumber(value)
        }
      }
    }
  };

  const chartData = getChartData();

  // Получение цвета для состояния на основе вероятности
  const getStatusColorFromProbability = (probability: number): 'success' | 'warning' | 'error' => {
    if (probability > 0.9) return 'success';
    if (probability >= 0.7) return 'warning';
    return 'error';
  };

  // Получение цвета для индикатора
  const getStatusIndicatorColor = (probability: number): string => {
    if (probability > 0.9) return '#4caf50'; // Зеленый
    if (probability >= 0.7) return '#ff9800'; // Желтый/Оранжевый
    return '#f44336'; // Красный
  };

  // Подготовка данных для гистограммы вероятностей
  const getProbabilityChartData = () => {
    if (!mlPrediction) return null;

    const probabilities = mlPrediction.метрики.вероятности_классов;
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities);

    // Цвета для каждого класса
    const colors = labels.map(label => {
      if (label === 'норма') return 'rgba(76, 175, 80, 0.8)';
      if (label.includes('внутреннего')) return 'rgba(255, 152, 0, 0.8)';
      if (label.includes('внешнего')) return 'rgba(255, 193, 7, 0.8)';
      return 'rgba(244, 67, 54, 0.8)'; // Неисправность шарика
    });

    return {
      labels,
      datasets: [
        {
          label: 'Вероятность (%)',
          data: values,
          backgroundColor: colors,
          borderColor: colors.map(c => c.replace('0.8', '1')),
          borderWidth: 2
        }
      ]
    };
  };

  const probabilityChartData = getProbabilityChartData();

  const probabilityChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Вероятности классов',
        font: {
          size: 14,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            return `${context.label}: ${formatNumber(context.parsed.y)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Вероятность (%)'
        },
        ticks: {
          callback: (value: any) => `${value}%`
        }
      },
      x: {
        title: {
          display: true,
          text: 'Состояние'
        }
      }
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Заголовок */}
      <Typography
        variant="h4"
        component="h1"
        gutterBottom
        sx={{ mb: 4, fontWeight: 'bold', textAlign: 'center' }}
      >
        Мониторинг состояния подшипников
      </Typography>

      {/* Сообщения об ошибках */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Панель управления */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Панель управления
              </Typography>
              <Stack direction="row" spacing={2} flexWrap="wrap">
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrow />}
                  onClick={handleStartMonitoring}
                  disabled={monitoringStatus === 'running' || isLoading}
                >
                  Старт мониторинга
                </Button>
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<Stop />}
                  onClick={handleStopMonitoring}
                  disabled={monitoringStatus === 'stopped' || isLoading}
                >
                  Стоп
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<Settings />}
                  onClick={handleCalibration}
                  disabled={monitoringStatus === 'running' || isLoading}
                >
                  Калибровка
                </Button>
                {isLoading && <CircularProgress size={24} />}
              </Stack>
              {monitoringStatus === 'running' && (
                <Chip
                  label="Мониторинг активен"
                  color="success"
                  sx={{ mt: 2 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Карточка статуса */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Статус системы
              </Typography>
              {vibrationData ? (
                <Box>
                  <Stack spacing={2}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Устройство
                      </Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {vibrationData.device_id}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Последнее обновление
                      </Typography>
                      <Typography variant="body1">
                        {formatDateTime(vibrationData.timestamp)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Температура
                      </Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {formatNumber(vibrationData.temperature)} °C
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Частота дискретизации
                      </Typography>
                      <Typography variant="body1">
                        {formatNumber(vibrationData.sampling_rate)} Гц
                      </Typography>
                    </Box>
                    {mlDiagnosis && (
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Диагноз ML модели
                        </Typography>
                        <Chip
                          label={mlDiagnosis.status === 'normal' ? 'Норма' : 
                                 mlDiagnosis.status === 'warning' ? 'Предупреждение' : 'Критично'}
                          color={getStatusColor(mlDiagnosis.status)}
                          sx={{ mb: 1 }}
                        />
                        <Alert 
                          severity={mlDiagnosis.status === 'normal' ? 'success' : 
                                   mlDiagnosis.status === 'warning' ? 'warning' : 'error'}
                          sx={{ mt: 1 }}
                        >
                          <Typography variant="body2">
                            {mlDiagnosis.message}
                          </Typography>
                          <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                            Уверенность: {formatNumber(mlDiagnosis.confidence * 100)}%
                          </Typography>
                          {mlDiagnosis.recommendations && mlDiagnosis.recommendations.length > 0 && (
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="caption" fontWeight="bold">
                                Рекомендации:
                              </Typography>
                              <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                {mlDiagnosis.recommendations.map((rec, idx) => (
                                  <li key={idx}>
                                    <Typography variant="caption">{rec}</Typography>
                                  </li>
                                ))}
                              </ul>
                            </Box>
                          )}
                        </Alert>
                      </Box>
                    )}
                  </Stack>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Данные не загружены
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* График вибрации */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ height: 400 }}>
                {chartData ? (
                  <Line data={chartData} options={chartOptions} />
                ) : (
                  <Box
                    sx={{
                      height: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <Typography variant="body1" color="text.secondary">
                      Запустите мониторинг для отображения данных
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Статистика вибрации */}
        {vibrationData && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Статистика вибрации
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Максимум по оси X
                      </Typography>
                      <Typography variant="h6" color="error">
                        {formatNumber(Math.max(...vibrationData.vibration_x))} мм/с
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Максимум по оси Y
                      </Typography>
                      <Typography variant="h6" color="error">
                        {formatNumber(Math.max(...vibrationData.vibration_y))} мм/с
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Максимум по оси Z
                      </Typography>
                      <Typography variant="h6" color="error">
                        {formatNumber(Math.max(...vibrationData.vibration_z))} мм/с
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Диагностика ML модели */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Диагностика ML модели
                </Typography>
                <Stack direction="row" spacing={2}>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<Refresh />}
                    onClick={handleRefreshMLDiagnosis}
                    disabled={!vibrationData || isLoadingML}
                  >
                    Обновить диагностику
                  </Button>
                  {mlPrediction && (
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<Info />}
                      onClick={() => setMlDetailsOpen(true)}
                    >
                      Подробнее
                    </Button>
                  )}
                </Stack>
              </Box>

              {mlError && (
                <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setMlError(null)}>
                  {mlError}
                </Alert>
              )}

              {isLoadingML && (
                <Box sx={{ mb: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Выполняется анализ ML модели...
                  </Typography>
                </Box>
              )}

              {mlPrediction ? (
                <Grid container spacing={3}>
                  {/* Текущее состояние и вероятность */}
                  <Grid item xs={12} md={4}>
                    <Paper 
                      sx={{ 
                        p: 3, 
                        textAlign: 'center',
                        border: `3px solid ${getStatusIndicatorColor(mlPrediction.вероятность)}`,
                        borderRadius: 2
                      }}
                    >
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Предсказанное состояние
                      </Typography>
                      <Chip
                        label={mlPrediction.состояние}
                        color={getStatusColorFromProbability(mlPrediction.вероятность)}
                        sx={{ 
                          fontSize: '1.1rem',
                          fontWeight: 'bold',
                          mb: 2,
                          height: 36
                        }}
                      />
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="h4" fontWeight="bold" color={getStatusIndicatorColor(mlPrediction.вероятность)}>
                          {formatNumber(mlPrediction.вероятность * 100)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Уверенность предсказания
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>

                  {/* Рекомендации */}
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                      <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                        Рекомендации
                      </Typography>
                      <List dense>
                        {mlPrediction.рекомендации.slice(0, 3).map((rec, idx) => (
                          <ListItem key={idx} sx={{ px: 0 }}>
                            <ListItemText
                              primary={rec}
                              primaryTypographyProps={{ variant: 'body2' }}
                            />
                          </ListItem>
                        ))}
                        {mlPrediction.рекомендации.length > 3 && (
                          <Typography variant="caption" color="text.secondary">
                            и еще {mlPrediction.рекомендации.length - 3}...
                          </Typography>
                        )}
                      </List>
                    </Paper>
                  </Grid>

                  {/* Гистограмма вероятностей */}
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, height: '100%' }}>
                      <Box sx={{ height: 200 }}>
                        {probabilityChartData ? (
                          <Bar data={probabilityChartData} options={probabilityChartOptions} />
                        ) : (
                          <Box
                            sx={{
                              height: '100%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                          >
                            <Typography variant="body2" color="text.secondary">
                              Нет данных для графика
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    </Paper>
                  </Grid>
                </Grid>
              ) : (
                <Alert severity="info">
                  Запустите мониторинг для получения диагностики ML модели
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Модальное окно с подробной информацией о предсказании */}
      <Dialog
        open={mlDetailsOpen}
        onClose={() => setMlDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Подробная информация о предсказании ML модели
        </DialogTitle>
        <DialogContent>
          {mlPrediction && (
            <Box>
              <Stack spacing={3}>
                {/* Основная информация */}
                <Box>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    Основная информация
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Состояние
                      </Typography>
                      <Chip
                        label={mlPrediction.состояние}
                        color={getStatusColorFromProbability(mlPrediction.вероятность)}
                        sx={{ mt: 1 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Вероятность
                      </Typography>
                      <Typography variant="h6" color={getStatusIndicatorColor(mlPrediction.вероятность)} sx={{ mt: 1 }}>
                        {formatNumber(mlPrediction.вероятность * 100)}%
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Код состояния
                      </Typography>
                      <Typography variant="body1" sx={{ mt: 1 }}>
                        {mlPrediction.метрики.код_состояния}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Уверенность (в процентах)
                      </Typography>
                      <Typography variant="body1" sx={{ mt: 1 }}>
                        {formatNumber(mlPrediction.метрики.уверенность_процентах)}%
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                {/* Вероятности всех классов */}
                <Box>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    Вероятности всех классов
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <List>
                    {Object.entries(mlPrediction.метрики.вероятности_классов).map(([className, probability]) => (
                      <ListItem key={className}>
                        <ListItemText
                          primary={className}
                          secondary={`${formatNumber(probability)}%`}
                        />
                        <Box sx={{ width: 200, mr: 2 }}>
                          <LinearProgress
                            variant="determinate"
                            value={probability}
                            sx={{
                              height: 8,
                              borderRadius: 4,
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: className === 'норма' 
                                  ? '#4caf50' 
                                  : className.includes('износ')
                                  ? '#ff9800'
                                  : '#f44336'
                              }
                            }}
                          />
                        </Box>
                      </ListItem>
                    ))}
                  </List>
                </Box>

                {/* Все рекомендации */}
                <Box>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    Все рекомендации
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <List>
                    {mlPrediction.рекомендации.map((rec, idx) => (
                      <ListItem key={idx}>
                        <ListItemText
                          primary={`${idx + 1}. ${rec}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </Stack>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMlDetailsOpen(false)}>Закрыть</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Dashboard;



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
  Stack
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import { VibrationData, MLDiagnosis, MonitoringStatus } from '../types/vibration';
import { formatNumber, formatDateTime, getStatusColor } from '../utils/formatters';

// Регистрация компонентов Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  // Загрузка данных вибрации
  const fetchVibrationData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // В реальном приложении здесь будет запрос к API
      // const response = await axios.get(`${API_BASE_URL}/api/v1/vibration-data/latest`);
      // setVibrationData(response.data);

      // Симуляция для демонстрации
      const mockData = generateMockData();
      setVibrationData(mockData);
      setMlDiagnosis(generateMLDiagnosis(mockData));

      // Отправка данных на сервер (если нужно)
      // await axios.post(`${API_BASE_URL}/api/v1/vibration-data`, mockData);
    } catch (err) {
      setError('Ошибка при загрузке данных вибрации');
      console.error(err);
    } finally {
      setIsLoading(false);
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
      </Grid>
    </Container>
  );
};

export default Dashboard;



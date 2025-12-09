/**
 * Типы данных для системы мониторинга вибрации
 */

export interface VibrationData {
  device_id: string;
  timestamp: string;
  vibration_x: number[];
  vibration_y: number[];
  vibration_z: number[];
  sampling_rate: number;
  temperature: number;
}

export interface DeviceStatus {
  device_id: string;
  last_measurement: string | null;
  total_measurements: number;
  status: string;
}

export interface SystemStatus {
  system_status: string;
  devices: DeviceStatus[];
  total_devices: number;
}

export interface MLDiagnosis {
  status: 'normal' | 'warning' | 'critical';
  message: string;
  confidence: number;
  recommendations?: string[];
}

export type MonitoringStatus = 'stopped' | 'running' | 'calibrating';



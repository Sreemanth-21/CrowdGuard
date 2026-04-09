/**
 * Design System Constants
 * 
 * Centralized constants for risk levels, anomaly types, API endpoints,
 * and color mappings used throughout the CrowdGuard application.
 */

// Risk Levels (Requirements 13.4, 14.4, 32.5)
export const RISK_LEVELS = {
  SAFE: 'SAFE',
  CAUTION: 'CAUTION',
  WARNING: 'WARNING',
  CRITICAL: 'CRITICAL',
} as const;

export type RiskLevel = typeof RISK_LEVELS[keyof typeof RISK_LEVELS];

// Risk Level Ranges
export const RISK_LEVEL_RANGES = {
  SAFE: { min: 0, max: 25 },
  CAUTION: { min: 26, max: 50 },
  WARNING: { min: 51, max: 75 },
  CRITICAL: { min: 76, max: 100 },
} as const;

// Risk Level Colors (Requirement 32.5)
export const RISK_COLORS = {
  SAFE: {
    bg: 'bg-green-500',
    text: 'text-green-500',
    border: 'border-green-500',
    hex: '#22c55e',
  },
  CAUTION: {
    bg: 'bg-amber-500',
    text: 'text-amber-500',
    border: 'border-amber-500',
    hex: '#f59e0b',
  },
  WARNING: {
    bg: 'bg-orange-500',
    text: 'text-orange-500',
    border: 'border-orange-500',
    hex: '#f97316',
  },
  CRITICAL: {
    bg: 'bg-red-500',
    text: 'text-red-500',
    border: 'border-red-500',
    hex: '#ef4444',
  },
} as const;

// Anomaly Types
export const ANOMALY_TYPES = {
  HIGH_DENSITY: 'HIGH_DENSITY',
  RAPID_MOVEMENT: 'RAPID_MOVEMENT',
  SUDDEN_DISPERSAL: 'SUDDEN_DISPERSAL',
  CROWD_SURGE: 'CROWD_SURGE',
  STATIONARY_CROWD: 'STATIONARY_CROWD',
  FIGHTING: 'FIGHTING',
} as const;

export type AnomalyType = typeof ANOMALY_TYPES[keyof typeof ANOMALY_TYPES];

// Anomaly Type Display Names
export const ANOMALY_LABELS = {
  HIGH_DENSITY: 'High Density',
  RAPID_MOVEMENT: 'Rapid Movement',
  SUDDEN_DISPERSAL: 'Sudden Dispersal',
  CROWD_SURGE: 'Crowd Surge',
  STATIONARY_CROWD: 'Stationary Crowd',
  FIGHTING: 'Fighting',
} as const;

// API Endpoints
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

export const API_ENDPOINTS = {
  // Health
  HEALTH: '/health',
  
  // Video Management
  VIDEO_UPLOAD: '/api/video/upload',
  VIDEO_START: '/api/video/start',
  VIDEO_STOP: '/api/video/stop',
  VIDEO_STATUS: '/api/video/status',
  VIDEO_SOURCES: '/api/video/sources',
  
  // Alerts
  ALERTS: '/api/alerts',
  ALERT_DETAIL: (id: string) => `/api/alerts/${id}`,
  ALERT_DISMISS: (id: string) => `/api/alerts/${id}/dismiss`,
  ALERT_BULK_DISMISS: '/api/alerts/bulk-dismiss',
  ALERT_SUMMARY: '/api/alerts/summary',
  ALERT_EXPORT: '/api/alerts/export',
  ALERT_SNAPSHOT: (id: string) => `/api/alerts/${id}/snapshot`,
  
  // Analytics
  ANALYTICS_DENSITY_TIMESERIES: '/api/analytics/density-timeseries',
  ANALYTICS_RISK_TIMESERIES: '/api/analytics/risk-timeseries',
  ANALYTICS_ALERT_FREQUENCY: '/api/analytics/alert-frequency',
  ANALYTICS_SESSION_STATS: '/api/analytics/session-stats',
  ANALYTICS_KPIS: '/api/analytics/kpis',
  
  // Settings
  SETTINGS: '/api/settings',
  SETTINGS_RESET: '/api/settings/reset',
  
  // Federated Learning
  FEDERATED_SIMULATE: '/api/federated/simulate',
  FEDERATED_STATUS: '/api/federated/status',
  
  // WebSocket
  WEBSOCKET: '/ws',
} as const;

// WebSocket Message Types
export const WS_MESSAGE_TYPES = {
  FRAME: 'frame',
  ALERT: 'alert',
  STATUS: 'status',
  ERROR: 'error',
  COMMAND: 'command',
  CONNECTED: 'connected',
} as const;

export type WSMessageType = typeof WS_MESSAGE_TYPES[keyof typeof WS_MESSAGE_TYPES];

// Video Source Types
export const VIDEO_SOURCE_TYPES = {
  WEBCAM: 'webcam',
  UPLOAD: 'upload',
} as const;

export type VideoSourceType = typeof VIDEO_SOURCE_TYPES[keyof typeof VIDEO_SOURCE_TYPES];

// File Upload Constraints
export const FILE_UPLOAD = {
  MAX_SIZE_MB: 500,
  MAX_SIZE_BYTES: 500 * 1024 * 1024,
  SUPPORTED_FORMATS: ['mp4', 'avi', 'mov', 'mkv'],
  SUPPORTED_MIME_TYPES: [
    'video/mp4',
    'video/x-msvideo',
    'video/quicktime',
    'video/x-matroska',
  ],
} as const;

/**
 * Get risk level color mapping
 * @param level - Risk level
 * @returns Color object with Tailwind classes and hex value
 */
export function getRiskColor(level: RiskLevel) {
  return RISK_COLORS[level];
}

/**
 * Get risk level from score
 * @param score - Risk score (0-100)
 * @returns Risk level
 */
export function getRiskLevelFromScore(score: number): RiskLevel {
  if (score >= 0 && score <= 25) return RISK_LEVELS.SAFE;
  if (score >= 26 && score <= 50) return RISK_LEVELS.CAUTION;
  if (score >= 51 && score <= 75) return RISK_LEVELS.WARNING;
  return RISK_LEVELS.CRITICAL;
}

/**
 * Get anomaly type display label
 * @param type - Anomaly type
 * @returns Human-readable label
 */
export function getAnomalyLabel(type: AnomalyType): string {
  return ANOMALY_LABELS[type];
}

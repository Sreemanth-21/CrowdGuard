/**
 * AlertCard Component
 * Displays individual alert with timestamp, type, risk level, and dismiss functionality
 */

import React from 'react';
import { RiskBadge } from '../shared';
import { formatRelativeTime } from '../../utils/formatters';
import { ANOMALY_LABELS } from '../../utils/constants';
import { useAlerts } from '../../hooks';

interface Alert {
  id: string;
  timestamp: string;
  anomalyType: string;
  riskLevel: string;
  confidence: number;
  personCount: number;
  density: number;
  description: string;
  snapshotPath?: string;
}

interface AlertCardProps {
  alert: Alert;
  onDismiss?: (alertId: string) => void;
  className?: string;
}

export const AlertCard: React.FC<AlertCardProps> = ({
  alert,
  onDismiss,
  className = '',
}) => {
  const { dismissAlert, getSnapshotUrl } = useAlerts();

  const handleDismiss = async () => {
    try {
      await dismissAlert(alert.id);
      onDismiss?.(alert.id);
    } catch (error) {
      console.error('Failed to dismiss alert:', error);
    }
  };

  const getAnomalyLabel = (type: string): string => {
    return ANOMALY_LABELS[type as keyof typeof ANOMALY_LABELS] || type;
  };

  const getAnomalyIcon = (type: string): string => {
    switch (type) {
      case 'HIGH_DENSITY': return '👥';
      case 'RAPID_MOVEMENT': return '🏃';
      case 'SUDDEN_DISPERSAL': return '💨';
      case 'CROWD_SURGE': return '🌊';
      case 'STATIONARY_CROWD': return '⏸️';
      case 'FIGHTING': return '⚡';
      default: return '⚠️';
    }
  };

  return (
    <div className={`bg-slate-700 rounded-lg p-4 border-l-4 border-red-500 ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {/* Alert header */}
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-lg">{getAnomalyIcon(alert.anomalyType)}</span>
            <h4 className="font-semibold text-white">
              {getAnomalyLabel(alert.anomalyType)}
            </h4>
            <RiskBadge level={alert.riskLevel as any} size="sm" />
          </div>

          {/* Alert details */}
          <div className="space-y-1 text-sm text-gray-300">
            <p>{alert.description}</p>
            <div className="flex items-center space-x-4">
              <span>Confidence: {(alert.confidence * 100).toFixed(1)}%</span>
              <span>Persons: {alert.personCount}</span>
              <span>Density: {(alert.density * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Timestamp */}
          <div className="mt-2 text-xs text-gray-400">
            {formatRelativeTime(alert.timestamp)}
          </div>
        </div>

        {/* Dismiss button */}
        <button
          onClick={handleDismiss}
          className="ml-4 text-gray-400 hover:text-white transition-colors p-1"
          title="Dismiss alert"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Snapshot thumbnail */}
      {alert.snapshotPath && (
        <div className="mt-3">
          <img
            src={getSnapshotUrl(alert.id)}
            alt="Alert snapshot"
            className="w-full h-20 object-cover rounded border border-slate-600"
            onError={(e) => {
              // Hide image if it fails to load
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
        </div>
      )}
    </div>
  );
};
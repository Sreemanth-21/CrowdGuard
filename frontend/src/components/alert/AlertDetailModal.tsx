/**
 * AlertDetailModal Component
 * Modal for displaying detailed alert information and full-size snapshot
 */

import React from 'react';
import { RiskBadge } from '../shared';
import { formatAbsoluteTime } from '../../utils/formatters';
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
  dismissed: boolean;
}

interface AlertDetailModalProps {
  alert: Alert | null;
  isOpen: boolean;
  onClose: () => void;
}

export const AlertDetailModal: React.FC<AlertDetailModalProps> = ({
  alert,
  isOpen,
  onClose,
}) => {
  const { getSnapshotUrl } = useAlerts();

  if (!isOpen || !alert) return null;

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <h2 className="text-xl font-semibold text-white">Alert Details</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Alert Information */}
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">
                  {ANOMALY_LABELS[alert.anomalyType as keyof typeof ANOMALY_LABELS] || alert.anomalyType}
                </h3>
                <RiskBadge level={alert.riskLevel as any} size="lg" />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Timestamp:</span>
                  <span className="text-white">{formatAbsoluteTime(alert.timestamp)}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="text-white">{(alert.confidence * 100).toFixed(1)}%</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Person Count:</span>
                  <span className="text-white">{alert.personCount}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Density:</span>
                  <span className="text-white">{(alert.density * 100).toFixed(1)}%</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={alert.dismissed ? 'text-gray-400' : 'text-green-400'}>
                    {alert.dismissed ? 'Dismissed' : 'Active'}
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-2">Description</h4>
                <p className="text-white">{alert.description}</p>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-2">Alert ID</h4>
                <code className="text-xs text-gray-300 bg-slate-700 px-2 py-1 rounded">
                  {alert.id}
                </code>
              </div>
            </div>

            {/* Snapshot */}
            <div>
              <h4 className="text-sm font-medium text-gray-400 mb-3">Snapshot</h4>
              {alert.snapshotPath ? (
                <div className="bg-slate-900 rounded-lg p-4">
                  <img
                    src={getSnapshotUrl(alert.id)}
                    alt="Alert snapshot"
                    className="w-full h-auto rounded border border-slate-600"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                      const parent = e.target as HTMLElement;
                      parent.innerHTML = '<div class="text-center py-8 text-gray-500">Snapshot not available</div>';
                    }}
                  />
                </div>
              ) : (
                <div className="bg-slate-900 rounded-lg p-8 text-center">
                  <div className="text-gray-500 text-4xl mb-2">📷</div>
                  <p className="text-gray-400">No snapshot available</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end p-6 border-t border-slate-700">
          <button
            onClick={onClose}
            className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
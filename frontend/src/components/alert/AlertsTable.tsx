/**
 * AlertsTable Component
 * Displays paginated table of alerts with sorting and selection
 */

import React, { useState } from 'react';
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

interface AlertsTableProps {
  alerts: Alert[];
  selectedAlerts: string[];
  onSelectAlert: (alertId: string) => void;
  onSelectAll: (selected: boolean) => void;
  onSort: (column: string, direction: 'asc' | 'desc') => void;
  onViewDetails: (alert: Alert) => void;
  className?: string;
}

type SortColumn = 'timestamp' | 'anomalyType' | 'riskLevel' | 'confidence';
type SortDirection = 'asc' | 'desc';

export const AlertsTable: React.FC<AlertsTableProps> = ({
  alerts,
  selectedAlerts,
  onSelectAlert,
  onSelectAll,
  onSort,
  onViewDetails,
  className = '',
}) => {
  const [sortColumn, setSortColumn] = useState<SortColumn>('timestamp');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const { getSnapshotUrl, dismissAlert } = useAlerts();

  const handleSort = (column: SortColumn) => {
    const newDirection = sortColumn === column && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortColumn(column);
    setSortDirection(newDirection);
    onSort(column, newDirection);
  };

  const handleDismiss = async (alertId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await dismissAlert(alertId);
    } catch (error) {
      console.error('Failed to dismiss alert:', error);
    }
  };

  const getSortIcon = (column: SortColumn) => {
    if (sortColumn !== column) {
      return (
        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
        </svg>
      );
    }
    
    return sortDirection === 'asc' ? (
      <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
      </svg>
    ) : (
      <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    );
  };

  const allSelected = alerts.length > 0 && selectedAlerts.length === alerts.length;
  const someSelected = selectedAlerts.length > 0 && selectedAlerts.length < alerts.length;

  return (
    <div className={`bg-slate-800 rounded-lg overflow-hidden ${className}`}>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-slate-700">
            <tr>
              {/* Select All Checkbox */}
              <th className="px-4 py-3 text-left">
                <input
                  type="checkbox"
                  checked={allSelected}
                  ref={(input) => {
                    if (input) input.indeterminate = someSelected;
                  }}
                  onChange={(e) => onSelectAll(e.target.checked)}
                  className="rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
                />
              </th>

              {/* Timestamp */}
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('timestamp')}
                  className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white transition-colors"
                >
                  <span>Timestamp</span>
                  {getSortIcon('timestamp')}
                </button>
              </th>

              {/* Anomaly Type */}
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('anomalyType')}
                  className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white transition-colors"
                >
                  <span>Type</span>
                  {getSortIcon('anomalyType')}
                </button>
              </th>

              {/* Risk Level */}
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('riskLevel')}
                  className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white transition-colors"
                >
                  <span>Risk Level</span>
                  {getSortIcon('riskLevel')}
                </button>
              </th>

              {/* Confidence */}
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('confidence')}
                  className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white transition-colors"
                >
                  <span>Confidence</span>
                  {getSortIcon('confidence')}
                </button>
              </th>

              {/* Description */}
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">
                Description
              </th>

              {/* Thumbnail */}
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">
                Snapshot
              </th>

              {/* Actions */}
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">
                Actions
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-700">
            {alerts.map((alert) => (
              <tr
                key={alert.id}
                className={`
                  hover:bg-slate-700 transition-colors cursor-pointer
                  ${alert.dismissed ? 'opacity-60' : ''}
                `}
                onClick={() => onViewDetails(alert)}
              >
                {/* Checkbox */}
                <td className="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={selectedAlerts.includes(alert.id)}
                    onChange={() => onSelectAlert(alert.id)}
                    onClick={(e) => e.stopPropagation()}
                    className="rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
                  />
                </td>

                {/* Timestamp */}
                <td className="px-4 py-3 text-sm text-gray-300">
                  {formatAbsoluteTime(alert.timestamp)}
                </td>

                {/* Anomaly Type */}
                <td className="px-4 py-3 text-sm text-white">
                  {ANOMALY_LABELS[alert.anomalyType as keyof typeof ANOMALY_LABELS] || alert.anomalyType}
                </td>

                {/* Risk Level */}
                <td className="px-4 py-3">
                  <RiskBadge level={alert.riskLevel as any} size="sm" />
                </td>

                {/* Confidence */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2 min-w-[80px]">
                    <div className="flex-1 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${(alert.confidence * 100).toFixed(0)}%`,
                          background: alert.confidence >= 0.85
                            ? '#22c55e'
                            : alert.confidence >= 0.70
                            ? '#f59e0b'
                            : '#ef4444',
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono text-gray-300 tabular-nums w-10 text-right">
                      {(alert.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>

                {/* Description */}
                <td className="px-4 py-3 text-sm text-gray-300 max-w-xs truncate">
                  {alert.description}
                </td>

                {/* Thumbnail */}
                <td className="px-4 py-3">
                  {alert.snapshotPath ? (
                    <img
                      src={getSnapshotUrl(alert.id)}
                      alt="Alert snapshot"
                      className="w-12 h-8 object-cover rounded border border-slate-600"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  ) : (
                    <div className="w-12 h-8 bg-slate-700 rounded flex items-center justify-center">
                      <span className="text-xs text-gray-500">N/A</span>
                    </div>
                  )}
                </td>

                {/* Actions */}
                <td className="px-4 py-3">
                  <div className="flex items-center space-x-2">
                    {!alert.dismissed && (
                      <button
                        onClick={(e) => handleDismiss(alert.id, e)}
                        className="text-gray-400 hover:text-red-400 transition-colors"
                        title="Dismiss alert"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Empty state */}
        {alerts.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-4xl mb-2">📋</div>
            <p className="text-gray-400">No alerts found</p>
            <p className="text-gray-500 text-sm mt-1">
              Try adjusting your filters or check back later
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
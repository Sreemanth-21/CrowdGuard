/**
 * AlertFrequencyChart Component
 * Displays alert frequency by anomaly type as a bar chart
 */

import React, { useState, useEffect } from 'react';
import { analyticsApi } from '../../utils/api';
import { ANOMALY_LABELS, ANOMALY_TYPES } from '../../utils/constants';

interface AlertFrequencyData {
  anomalyType: string;
  count: number;
}

interface AlertFrequencyChartProps {
  timeRange?: number; // in minutes, default 60
  autoRefresh?: boolean;
  refreshInterval?: number; // in seconds
  className?: string;
}

export const AlertFrequencyChart: React.FC<AlertFrequencyChartProps> = ({
  timeRange = 60,
  autoRefresh = true,
  refreshInterval = 30,
  className = '',
}) => {
  const [data, setData] = useState<AlertFrequencyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setError(null);
      // Paginate through all alerts (backend limit=100 per page)
      let allAlerts: any[] = [];
      let skip = 0;
      const pageSize = 100;

      // Build time filter: only alerts within the selected timeRange
      const since = new Date(Date.now() - timeRange * 60 * 1000).toISOString();

      while (true) {
        const url = `/api/alerts?limit=${pageSize}&skip=${skip}&start_date=${encodeURIComponent(since)}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const json = await response.json();
        const page: any[] = json.alerts ?? [];
        allAlerts = allAlerts.concat(page);
        if (page.length < pageSize) break;
        skip += pageSize;
        if (skip >= 500) break;
      }

      // Count by anomaly type
      const counts: Record<string, number> = {};
      for (const alert of allAlerts) {
        const t = alert.anomaly_type ?? 'UNKNOWN';
        counts[t] = (counts[t] ?? 0) + 1;
      }

      const transformed: AlertFrequencyData[] = Object.entries(counts)
        .map(([anomalyType, count]) => ({ anomalyType, count }))
        .sort((a, b) => b.count - a.count);

      setData(transformed);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch alert frequency data');
      setData([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchData, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  const getAnomalyColor = (anomalyType: string) => {
    const colors: Record<string, string> = {
      [ANOMALY_TYPES.HIGH_DENSITY]: '#EF4444',
      [ANOMALY_TYPES.RAPID_MOVEMENT]: '#F59E0B',
      [ANOMALY_TYPES.SUDDEN_DISPERSAL]: '#EAB308',
      [ANOMALY_TYPES.CROWD_SURGE]: '#DC2626',
      [ANOMALY_TYPES.STATIONARY_CROWD]: '#3B82F6',
      [ANOMALY_TYPES.FIGHTING]: '#7C2D12',
    };
    return colors[anomalyType] || '#6B7280';
  };

  if (loading) {
    return (
      <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
          <div className="h-64 bg-slate-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
        <h3 className="text-lg font-semibold text-white mb-4">Alert Frequency by Type</h3>
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-400 font-medium">Failed to load alert frequency data</p>
          <p className="text-red-300 text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const maxCount = Math.max(...data.map(d => d.count), 1);
  const chartWidth = 600;
  const chartHeight = 300;
  const padding = 60;
  const barWidth = data.length > 0 ? (chartWidth - 2 * padding) / data.length * 0.8 : 0;
  const barSpacing = data.length > 0 ? (chartWidth - 2 * padding) / data.length * 0.2 : 0;

  return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Alert Frequency by Type</h3>
        <span className="text-sm text-gray-400">
          Past {timeRange} minutes
        </span>
      </div>

      {data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-500">
          <div className="text-center">
            <div className="text-4xl mb-2">📊</div>
            <p>No alert data available</p>
          </div>
        </div>
      ) : (
        <div className="relative">
          <svg width={chartWidth} height={chartHeight} className="w-full h-auto">
            {/* Y-axis grid lines and labels */}
            {[0, Math.ceil(maxCount * 0.25), Math.ceil(maxCount * 0.5), Math.ceil(maxCount * 0.75), maxCount].map((value) => {
              const y = chartHeight - padding - (value / maxCount) * (chartHeight - 2 * padding);
              return (
                <g key={value}>
                  <line
                    x1={padding}
                    y1={y}
                    x2={chartWidth - padding}
                    y2={y}
                    stroke="#4B5563"
                    strokeWidth="1"
                    opacity="0.5"
                  />
                  <text
                    x={padding - 10}
                    y={y + 4}
                    fill="#9CA3AF"
                    fontSize="12"
                    textAnchor="end"
                  >
                    {value}
                  </text>
                </g>
              );
            })}

            {/* Bars */}
            {data.map((item, index) => {
              const x = padding + index * (barWidth + barSpacing) + barSpacing / 2;
              const barHeight = (item.count / maxCount) * (chartHeight - 2 * padding);
              const y = chartHeight - padding - barHeight;

              return (
                <g key={item.anomalyType}>
                  {/* Bar */}
                  <rect
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    fill={getAnomalyColor(item.anomalyType)}
                    className="hover:opacity-80 transition-opacity cursor-pointer"
                  >
                    <title>
                      {ANOMALY_LABELS[item.anomalyType as keyof typeof ANOMALY_LABELS] || item.anomalyType}: {item.count} alerts
                    </title>
                  </rect>

                  {/* Count label on top of bar */}
                  {item.count > 0 && (
                    <text
                      x={x + barWidth / 2}
                      y={y - 5}
                      fill="#FFFFFF"
                      fontSize="12"
                      fontWeight="bold"
                      textAnchor="middle"
                    >
                      {item.count}
                    </text>
                  )}

                  {/* X-axis label */}
                  <text
                    x={x + barWidth / 2}
                    y={chartHeight - padding + 20}
                    fill="#9CA3AF"
                    fontSize="10"
                    textAnchor="middle"
                    className="max-w-20"
                  >
                    {(ANOMALY_LABELS[item.anomalyType as keyof typeof ANOMALY_LABELS] || item.anomalyType)
                      .split(' ')
                      .map((word, i) => (
                        <tspan key={i} x={x + barWidth / 2} dy={i === 0 ? 0 : 12}>
                          {word}
                        </tspan>
                      ))}
                  </text>
                </g>
              );
            })}

            {/* Axes */}
            <line
              x1={padding}
              y1={chartHeight - padding}
              x2={chartWidth - padding}
              y2={chartHeight - padding}
              stroke="#6B7280"
              strokeWidth="2"
            />
            <line
              x1={padding}
              y1={padding}
              x2={padding}
              y2={chartHeight - padding}
              stroke="#6B7280"
              strokeWidth="2"
            />
          </svg>

          {/* Legend */}
          <div className="mt-4">
            <div className="flex flex-wrap gap-4 justify-center">
              {Object.values(ANOMALY_TYPES).map((type) => (
                <div key={type} className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-sm" 
                    style={{ backgroundColor: getAnomalyColor(type) }}
                  ></div>
                  <span className="text-xs text-gray-300">
                    {ANOMALY_LABELS[type as keyof typeof ANOMALY_LABELS] || type}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
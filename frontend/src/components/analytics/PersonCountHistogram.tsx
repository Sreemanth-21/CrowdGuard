/**
 * PersonCountHistogram Component
 * Displays person count distribution as a histogram
 */

import React, { useState, useEffect } from 'react';
import { analyticsApi } from '../../utils/api';

interface HistogramBin {
  range: string;
  count: number;
  minValue: number;
  maxValue: number;
}

interface PersonCountHistogramProps {
  timeRange?: number; // in minutes, default 60
  autoRefresh?: boolean;
  refreshInterval?: number; // in seconds
  className?: string;
}

export const PersonCountHistogram: React.FC<PersonCountHistogramProps> = ({
  timeRange = 60,
  autoRefresh = true,
  refreshInterval = 30,
  className = '',
}) => {
  const [data, setData] = useState<HistogramBin[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setError(null);
      const response = await analyticsApi.getDensityTimeseries(timeRange);

      // Backend returns { data: [{timestamp, value}], ... } — unwrap
      const raw: any[] = Array.isArray(response)
        ? response
        : Array.isArray((response as any)?.data)
        ? (response as any).data
        : [];

      if (raw.length === 0) {
        setData([]);
        return;
      }

      // Each point has `person_count` (real) or we fall back to estimating from `value`/`density`
      const personCounts: number[] = raw.map((d: any) =>
        d.person_count ?? Math.round((d.density ?? d.value ?? 0) * 60)
      );

      const maxPersons = Math.max(...personCounts, 1);
      const binCount = 8;
      const binSize = Math.ceil(maxPersons / binCount);

      const bins: HistogramBin[] = Array.from({ length: binCount }, (_, i) => {
        const minValue = i * binSize;
        const maxValue = (i + 1) * binSize;
        const range = i === binCount - 1
          ? `${minValue}+`
          : `${minValue}–${maxValue - 1}`;
        const count = personCounts.filter(p => p >= minValue && p < maxValue).length;
        return { range, count, minValue, maxValue };
      });

      setData(bins);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch person count data');
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
        <h3 className="text-lg font-semibold text-white mb-4">Person Count Distribution</h3>
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-400 font-medium">Failed to load person count data</p>
          <p className="text-red-300 text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const maxCount = Math.max(...data.map(d => d.count), 1);
  const chartWidth = 600;
  const chartHeight = 300;
  const padding = 60;
  const barWidth = data.length > 0 ? (chartWidth - 2 * padding) / data.length * 0.9 : 0;
  const barSpacing = data.length > 0 ? (chartWidth - 2 * padding) / data.length * 0.1 : 0;

  const getBarColor = (binIndex: number) => {
    // Color gradient from green (low counts) to red (high counts)
    const intensity = binIndex / (data.length - 1);
    if (intensity < 0.3) return '#10B981'; // Green
    if (intensity < 0.6) return '#F59E0B'; // Amber
    return '#EF4444'; // Red
  };

  return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Person Count Distribution</h3>
        <span className="text-sm text-gray-400">
          Past {timeRange} minutes
        </span>
      </div>

      {data.every(d => d.count === 0) ? (
        <div className="h-64 flex items-center justify-center text-gray-500">
          <div className="text-center">
            <div className="text-4xl mb-2">👥</div>
            <p>No person count data available</p>
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

            {/* Histogram bars */}
            {data.map((bin, index) => {
              const x = padding + index * (barWidth + barSpacing);
              const barHeight = (bin.count / maxCount) * (chartHeight - 2 * padding);
              const y = chartHeight - padding - barHeight;

              return (
                <g key={bin.range}>
                  {/* Bar */}
                  <rect
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    fill={getBarColor(index)}
                    className="hover:opacity-80 transition-opacity cursor-pointer"
                  >
                    <title>
                      {bin.range} persons: {bin.count} occurrences
                    </title>
                  </rect>

                  {/* Count label on top of bar */}
                  {bin.count > 0 && (
                    <text
                      x={x + barWidth / 2}
                      y={y - 5}
                      fill="#FFFFFF"
                      fontSize="12"
                      fontWeight="bold"
                      textAnchor="middle"
                    >
                      {bin.count}
                    </text>
                  )}

                  {/* X-axis label */}
                  <text
                    x={x + barWidth / 2}
                    y={chartHeight - padding + 20}
                    fill="#9CA3AF"
                    fontSize="10"
                    textAnchor="middle"
                  >
                    {bin.range}
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

            {/* Axis labels */}
            <text
              x={chartWidth / 2}
              y={chartHeight - 5}
              fill="#9CA3AF"
              fontSize="12"
              textAnchor="middle"
              fontWeight="bold"
            >
              Person Count Range
            </text>
            <text
              x={15}
              y={chartHeight / 2}
              fill="#9CA3AF"
              fontSize="12"
              textAnchor="middle"
              fontWeight="bold"
              transform={`rotate(-90, 15, ${chartHeight / 2})`}
            >
              Frequency
            </text>
          </svg>

          {/* Statistics */}
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-sm text-gray-400">Total Samples</div>
              <div className="text-lg font-semibold text-white">
                {data.reduce((sum, bin) => sum + bin.count, 0)}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Most Common Range</div>
              <div className="text-lg font-semibold text-white">
                {data.length > 0 
                  ? data.reduce((max, bin) => bin.count > max.count ? bin : max, data[0]).range
                  : 'N/A'
                }
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Peak Frequency</div>
              <div className="text-lg font-semibold text-white">
                {Math.max(...data.map(d => d.count))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
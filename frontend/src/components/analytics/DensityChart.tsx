import React, { useState, useEffect } from 'react';
import { analyticsApi } from '../../utils/api';
import { formatAbsoluteTime, formatDensity } from '../../utils/formatters';

interface DensityDataPoint { timestamp: string; density: number; }
interface DensityChartProps {
  timeRange?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
  className?: string;
}

function downsample<T>(arr: T[], max: number): T[] {
  if (arr.length <= max) return arr;
  const step = arr.length / max;
  return Array.from({ length: max }, (_, i) => arr[Math.min(Math.round(i * step), arr.length - 1)]);
}

export const DensityChart: React.FC<DensityChartProps> = ({
  timeRange = 60, autoRefresh = true, refreshInterval = 30, className = '',
}) => {
  const [data, setData]       = useState<DensityDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setError(null);
      const res = await analyticsApi.getDensityTimeseries(timeRange);
      const raw: any[] = Array.isArray(res) ? res : Array.isArray((res as any)?.data) ? (res as any).data : [];
      setData(raw.map((d: any) => ({ timestamp: d.timestamp, density: d.density ?? d.value ?? 0 })));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch density data');
    } finally { setLoading(false); }
  };

  useEffect(() => { fetchData(); }, [timeRange]);
  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(fetchData, refreshInterval * 1000);
    return () => clearInterval(id);
  }, [autoRefresh, refreshInterval]);

  if (loading) return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <div className="animate-pulse space-y-3">
        <div className="h-5 bg-slate-700 rounded w-1/3" />
        <div className="h-52 bg-slate-700 rounded" />
      </div>
    </div>
  );

  if (error) return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-white mb-3">Density Over Time</h3>
      <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400 font-medium">Failed to load density data</p>
        <p className="text-red-300 text-sm mt-1">{error}</p>
      </div>
    </div>
  );

  const pts = downsample(data, 80);
  const showDots = pts.length <= 40;

  // Layout — all content stays inside these bounds
  const W = 560, H = 200;
  const top = 10, bottom = 28, left = 42, right = 10;
  const iW = W - left - right;
  const iH = H - top - bottom;

  const gx = (i: number) => left + (i / Math.max(pts.length - 1, 1)) * iW;
  const gy = (v: number) => top + (1 - Math.max(0, Math.min(1, v))) * iH;

  const linePath = pts.length > 0
    ? 'M ' + pts.map((d, i) => `${gx(i).toFixed(1)},${gy(d.density).toFixed(1)}`).join(' L ')
    : '';
  const areaPath = linePath
    ? `${linePath} L ${gx(pts.length - 1).toFixed(1)},${gy(0).toFixed(1)} L ${gx(0).toFixed(1)},${gy(0).toFixed(1)} Z`
    : '';

  const fmtX = (ts: string) => {
    const d = new Date(ts);
    return timeRange > 1440
      ? d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
      : d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const xIdx = pts.length > 1
    ? [0, Math.floor(pts.length * 0.25), Math.floor(pts.length * 0.5),
       Math.floor(pts.length * 0.75), pts.length - 1]
    : [0];

  return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Density Over Time</h3>
        <span className="text-sm text-gray-400">Past {timeRange} min</span>
      </div>

      {pts.length === 0 ? (
        <div className="h-52 flex items-center justify-center text-gray-500">
          <div className="text-center"><div className="text-4xl mb-2">📊</div><p>No density data</p></div>
        </div>
      ) : (
        <>
          {/* viewBox matches W×H exactly — no overflow */}
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: 'block' }}>
            {/* clip everything to the inner plot area */}
            <defs>
              <clipPath id="dc-clip">
                <rect x={left} y={top} width={iW} height={iH} />
              </clipPath>
            </defs>

            {/* grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(v => (
              <g key={v}>
                <line x1={left} y1={gy(v)} x2={left + iW} y2={gy(v)}
                  stroke="#374151" strokeWidth="1" strokeDasharray="4 3" />
                <text x={left - 4} y={gy(v) + 4} fill="#6B7280" fontSize="10" textAnchor="end">
                  {Math.round(v * 100)}%
                </text>
              </g>
            ))}

            {/* area + line clipped */}
            <g clipPath="url(#dc-clip)">
              {areaPath && <path d={areaPath} fill="#3B82F6" fillOpacity="0.15" />}
              {linePath  && <path d={linePath} fill="none" stroke="#3B82F6" strokeWidth="2.5"
                strokeLinecap="round" strokeLinejoin="round" />}
              {showDots && pts.map((pt, i) => (
                <circle key={i} cx={gx(i)} cy={gy(pt.density)} r="3.5"
                  fill="#3B82F6" stroke="#1e293b" strokeWidth="1.5">
                  <title>{formatAbsoluteTime(pt.timestamp)}: {formatDensity(pt.density)}</title>
                </circle>
              ))}
            </g>

            {/* x-axis labels — below the plot area, always visible */}
            {xIdx.map(i => i < pts.length && (
              <text key={i} x={gx(i)} y={H - 4} fill="#6B7280" fontSize="9" textAnchor="middle">
                {fmtX(pts[i].timestamp)}
              </text>
            ))}
          </svg>

          <div className="flex items-center justify-center mt-2 gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full" />
            <span className="text-sm text-gray-300">Crowd Density</span>
          </div>
        </>
      )}
    </div>
  );
};

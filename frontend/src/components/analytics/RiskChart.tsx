import React, { useState, useEffect } from 'react';
import { analyticsApi } from '../../utils/api';
import { formatAbsoluteTime, formatRiskScore } from '../../utils/formatters';

interface RiskDataPoint { timestamp: string; riskScore: number; }
interface RiskChartProps {
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

const ZONES = [
  { min: 76, max: 100, color: '#EF4444', label: 'CRITICAL' },
  { min: 51, max: 76,  color: '#F59E0B', label: 'WARNING'  },
  { min: 26, max: 51,  color: '#EAB308', label: 'CAUTION'  },
  { min: 0,  max: 26,  color: '#10B981', label: 'SAFE'     },
];

const riskColor = (s: number) => {
  if (s >= 76) return '#EF4444';
  if (s >= 51) return '#F59E0B';
  if (s >= 26) return '#EAB308';
  return '#10B981';
};

export const RiskChart: React.FC<RiskChartProps> = ({
  timeRange = 60, autoRefresh = true, refreshInterval = 30, className = '',
}) => {
  const [data, setData]       = useState<RiskDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setError(null);
      const res = await analyticsApi.getRiskTimeseries(timeRange);
      const raw: any[] = Array.isArray(res) ? res : Array.isArray((res as any)?.data) ? (res as any).data : [];
      setData(raw.map((d: any) => ({
        timestamp: d.timestamp,
        riskScore: d.riskScore ?? d.risk_score ?? d.value ?? 0,
      })));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch risk data');
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
      <h3 className="text-lg font-semibold text-white mb-3">Risk Score Over Time</h3>
      <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400 font-medium">Failed to load risk data</p>
        <p className="text-red-300 text-sm mt-1">{error}</p>
      </div>
    </div>
  );

  const pts = downsample(data, 80);
  const showDots = pts.length <= 40;

  // Layout — all content stays inside W×H
  const W = 560, H = 200;
  const top = 10, bottom = 28, left = 36, right = 10;
  const iW = W - left - right;
  const iH = H - top - bottom;

  const gx = (i: number) => left + (i / Math.max(pts.length - 1, 1)) * iW;
  const gy = (v: number) => top + (1 - Math.max(0, Math.min(100, v)) / 100) * iH;

  const linePath = pts.length > 0
    ? 'M ' + pts.map((d, i) => `${gx(i).toFixed(1)},${gy(d.riskScore).toFixed(1)}`).join(' L ')
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
        <h3 className="text-lg font-semibold text-white">Risk Score Over Time</h3>
        <span className="text-sm text-gray-400">Past {timeRange} min</span>
      </div>

      {pts.length === 0 ? (
        <div className="h-52 flex items-center justify-center text-gray-500">
          <div className="text-center"><div className="text-4xl mb-2">⚡</div><p>No risk data</p></div>
        </div>
      ) : (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: 'block' }}>
            <defs>
              <clipPath id="rc-clip">
                <rect x={left} y={top} width={iW} height={iH} />
              </clipPath>
            </defs>

            {/* risk zone bands — clipped */}
            <g clipPath="url(#rc-clip)">
              {ZONES.map(z => (
                <rect key={z.label}
                  x={left} y={gy(z.max)} width={iW} height={gy(z.min) - gy(z.max)}
                  fill={z.color} fillOpacity="0.08" />
              ))}
            </g>

            {/* grid lines + y-axis labels */}
            {[0, 25, 50, 75, 100].map(v => (
              <g key={v}>
                <line x1={left} y1={gy(v)} x2={left + iW} y2={gy(v)}
                  stroke="#374151" strokeWidth="1" strokeDasharray="4 3" />
                <text x={left - 4} y={gy(v) + 4} fill="#6B7280" fontSize="10" textAnchor="end">{v}</text>
              </g>
            ))}

            {/* area + line + dots — clipped */}
            <g clipPath="url(#rc-clip)">
              {areaPath && <path d={areaPath} fill="#60A5FA" fillOpacity="0.10" />}
              {linePath  && <path d={linePath} fill="none" stroke="#60A5FA" strokeWidth="2.5"
                strokeLinecap="round" strokeLinejoin="round" />}
              {showDots && pts.map((pt, i) => (
                <circle key={i} cx={gx(i)} cy={gy(pt.riskScore)} r="4"
                  fill={riskColor(pt.riskScore)} stroke="#1e293b" strokeWidth="1.5">
                  <title>{formatAbsoluteTime(pt.timestamp)}: {formatRiskScore(pt.riskScore)}</title>
                </circle>
              ))}
            </g>

            {/* x-axis labels */}
            {xIdx.map(i => i < pts.length && (
              <text key={i} x={gx(i)} y={H - 4} fill="#6B7280" fontSize="9" textAnchor="middle">
                {fmtX(pts[i].timestamp)}
              </text>
            ))}
          </svg>

          {/* legend */}
          <div className="flex flex-wrap items-center justify-center mt-2 gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-0.5 bg-blue-400 rounded" />
              <span className="text-xs text-gray-300">Risk Score</span>
            </div>
            {ZONES.map(z => (
              <div key={z.label} className="flex items-center gap-1">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: z.color }} />
                <span className="text-xs text-gray-400">{z.label}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

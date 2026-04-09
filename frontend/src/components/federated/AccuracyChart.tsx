import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { motion } from 'framer-motion';

interface AccuracyDataPoint {
  round: number;
  globalAccuracy: number;
  [key: string]: number;
}

interface AccuracyChartProps {
  data: AccuracyDataPoint[];
  nodeNames: string[];
  isAnimated?: boolean;
  height?: number;
}

const NODE_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#6366f1'];
const GLOBAL_COLOR = '#8b5cf6';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-lg p-3 shadow-xl text-sm">
      <p className="text-white font-semibold mb-2">Round {label}</p>
      {payload.map((e: any, i: number) => (
        <div key={i} className="flex items-center gap-2 mb-1">
          <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: e.color }} />
          <span className="text-[#94a3b8]">{e.name}:</span>
          <span className="font-mono text-white ml-auto pl-3">{(e.value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
};

const CustomLegend = ({ payload }: any) => (
  <div className="flex flex-wrap gap-x-5 gap-y-1.5 justify-end pr-2 mb-2">
    {payload?.map((e: any, i: number) => (
      <div key={i} className="flex items-center gap-1.5">
        <div className="w-4 h-0.5 rounded" style={{ background: e.color }} />
        <span className="text-xs text-[#94a3b8]">{e.value}</span>
      </div>
    ))}
  </div>
);

const AccuracyChart: React.FC<AccuracyChartProps> = ({
  data, nodeNames, isAnimated = true, height = 350,
}) => {
  const first = data[0]?.globalAccuracy ?? 0;
  const last  = data[data.length - 1]?.globalAccuracy ?? 0;

  return (
    <motion.div
      initial={isAnimated ? { opacity: 0, y: 12 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="space-y-4"
    >
      <div>
        <h3 className="text-base font-semibold text-white">Accuracy Convergence</h3>
        <p className="text-xs text-[#94a3b8] mt-0.5">
          Global and per-node accuracy across {data.length} training rounds
        </p>
      </div>

      <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-4">
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 28 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.6} />
            <XAxis
              dataKey="round"
              stroke="#475569"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              label={{ value: 'Round', position: 'insideBottom', offset: -14, fill: '#64748b', fontSize: 12 }}
            />
            <YAxis
              domain={[0, 1]}
              width={60}
              stroke="#475569"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', offset: 14, fill: '#64748b', fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend content={<CustomLegend />} verticalAlign="top" />

            {/* Global — solid, thicker, purple */}
            <Line
              type="monotone"
              dataKey="globalAccuracy"
              name="Global"
              stroke={GLOBAL_COLOR}
              strokeWidth={3}
              dot={{ fill: GLOBAL_COLOR, r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, stroke: GLOBAL_COLOR, strokeWidth: 2, fill: '#0f172a' }}
              animationDuration={isAnimated ? 1600 : 0}
            />

            {/* Per-node — dashed, thinner */}
            {nodeNames.map((name, i) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                name={name.replace(/_/g, ' ')}
                stroke={NODE_COLORS[i % NODE_COLORS.length]}
                strokeWidth={1.8}
                strokeDasharray="5 3"
                dot={false}
                activeDot={{ r: 4 }}
                animationDuration={isAnimated ? 1600 : 0}
                animationBegin={isAnimated ? i * 120 : 0}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary stats */}
      {data.length > 1 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Initial accuracy', value: `${(first * 100).toFixed(1)}%`, color: 'text-white' },
            { label: 'Final accuracy',   value: `${(last  * 100).toFixed(1)}%`, color: 'text-white' },
            { label: 'Improvement',      value: `+${((last - first) * 100).toFixed(1)}%`, color: 'text-green-400' },
            { label: 'Rounds',           value: String(data.length), color: 'text-white' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-[#1e293b] rounded-lg border border-[#334155] p-3">
              <p className="text-xs text-[#94a3b8] mb-1">{label}</p>
              <p className={`text-lg font-bold font-mono ${color}`}>{value}</p>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default AccuracyChart;

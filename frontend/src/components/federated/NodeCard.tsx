import React from 'react';
import { motion } from 'framer-motion';

interface NodeCardProps {
  name: string;
  datasetSize: number;
  currentAccuracy: number;
  trainingStatus: 'idle' | 'training' | 'completed';
  roundsCompleted: number;
  isActive?: boolean;
}

const STATUS: Record<string, { badge: string; dot: string; label: string }> = {
  training:  { badge: 'text-yellow-300 bg-yellow-900/40 border-yellow-500/40',   dot: 'bg-yellow-400',  label: 'Training'  },
  completed: { badge: 'text-emerald-300 bg-emerald-900/40 border-emerald-500/40', dot: 'bg-emerald-400', label: 'Completed' },
  idle:      { badge: 'text-slate-400 bg-slate-800/60 border-slate-600/40',       dot: 'bg-slate-500',   label: 'Idle'      },
};

const NodeCard: React.FC<NodeCardProps> = ({
  name, datasetSize, currentAccuracy, trainingStatus, roundsCompleted, isActive = false,
}) => {
  const s   = STATUS[trainingStatus] ?? STATUS.idle;
  const pct = Math.min(currentAccuracy * 100, 100);
  const displayName = name.replace(/_/g, ' ');

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`relative flex flex-col gap-3 p-4 rounded-xl border transition-all duration-300
        ${isActive
          ? 'bg-[#1e293b] border-blue-500/60 shadow-lg shadow-blue-500/10'
          : 'bg-[#1e293b] border-slate-700/60 hover:border-slate-600/60'}`}
    >
      {isActive && (
        <motion.div
          animate={{ opacity: [0.2, 0.5, 0.2] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute inset-0 rounded-xl border-2 border-blue-500/40 pointer-events-none"
        />
      )}

      {/* Row 1: status dot + node name (full width) */}
      <div className="flex items-center gap-2">
        {trainingStatus === 'training' ? (
          <motion.span
            animate={{ scale: [1, 1.5, 1], opacity: [0.6, 1, 0.6] }}
            transition={{ duration: 1.2, repeat: Infinity }}
            className={`w-2 h-2 rounded-full shrink-0 ${s.dot}`}
          />
        ) : (
          <span className={`w-2 h-2 rounded-full shrink-0 ${s.dot}`} />
        )}
        <span className="text-sm font-bold text-white leading-tight">{displayName}</span>
      </div>

      {/* Row 2: status badge on its own row */}
      <span className={`self-start text-xs font-semibold px-2.5 py-0.5 rounded-full border ${s.badge}`}>
        {s.label}
      </span>

      {/* Divider */}
      <div className="border-t border-slate-700/40" />

      {/* Dataset size */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">Dataset size</span>
        <span className="font-mono text-slate-200 tabular-nums">{datasetSize.toLocaleString()}</span>
      </div>

      {/* Local accuracy */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Local accuracy</span>
          <span className="font-mono font-semibold text-white tabular-nums">{pct.toFixed(1)}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-700 overflow-hidden">
          <motion.div
            initial={{ width: 0 }} animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-400"
          />
        </div>
      </div>

      {/* Rounds completed */}
      <div className="flex items-center justify-between text-xs border-t border-slate-700/40 pt-2">
        <span className="text-slate-400">Rounds completed</span>
        <span className="font-mono text-slate-200 tabular-nums">{roundsCompleted}</span>
      </div>
    </motion.div>
  );
};

export default NodeCard;

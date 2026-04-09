import React from "react";
import { motion } from "framer-motion";

interface CoordinatorCardProps {
  globalAccuracy: number;
  currentRound: number;
  totalRounds: number;
  status: "idle" | "running" | "completed" | "failed";
  convergenceRate?: number;
  isAggregating?: boolean;
}

const STATUS_STYLES: Record<string, string> = {
  running:   "text-blue-300 bg-blue-900/40 border-blue-500/40",
  completed: "text-emerald-300 bg-emerald-900/40 border-emerald-500/40",
  failed:    "text-red-300 bg-red-900/40 border-red-500/40",
  idle:      "text-slate-400 bg-slate-800/60 border-slate-600/40",
};

const CoordinatorCard: React.FC<CoordinatorCardProps> = ({
  globalAccuracy, currentRound, totalRounds, status,
  convergenceRate = 0, isAggregating = false,
}) => {
  const pct      = Math.min(globalAccuracy * 100, 100);
  const roundPct = totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;
  const badge    = STATUS_STYLES[status] ?? STATUS_STYLES.idle;

  return (
    <div className="relative flex flex-col gap-5 p-5 rounded-xl border h-full bg-[#1e293b] border-purple-500/50 shadow-lg shadow-purple-500/10">
      {status === "running" && (
        <motion.div
          animate={{ opacity: [0.15, 0.4, 0.15] }}
          transition={{ duration: 2.2, repeat: Infinity }}
          className="absolute inset-0 rounded-xl border-2 border-purple-500/40 pointer-events-none"
        />
      )}

      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <motion.div
            animate={isAggregating ? { rotate: 360 } : { rotate: 0 }}
            transition={isAggregating ? { duration: 3, repeat: Infinity, ease: "linear" } : {}}
            className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center shrink-0"
          >
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
              <circle cx="12" cy="5" r="2" />
              <circle cx="5" cy="19" r="2" />
              <circle cx="19" cy="19" r="2" />
              <line x1="12" y1="7" x2="5" y2="17" />
              <line x1="12" y1="7" x2="19" y2="17" />
              <line x1="5" y1="19" x2="19" y2="19" />
            </svg>
          </motion.div>
          <div>
            <p className="text-sm font-bold text-white">Coordinator</p>
            <p className="text-xs text-slate-400">FedAvg Aggregation</p>
          </div>
        </div>
        <span className={"self-start text-xs font-semibold px-2.5 py-1 rounded-full border " + badge}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>

      <div className="text-center">
        <p className="text-xs uppercase tracking-widest text-slate-400 mb-2">Global Accuracy</p>
        <p className="text-5xl font-extrabold text-purple-400 leading-none tabular-nums">
          {pct.toFixed(1)}<span className="text-2xl text-purple-300/70 ml-0.5">%</span>
        </p>
        <div className="mt-3 h-2 w-full rounded-full bg-slate-700 overflow-hidden">
          <motion.div
            initial={{ width: 0 }} animate={{ width: `${pct}%` }}
            transition={{ duration: 0.9, ease: "easeOut" }}
            className="h-full rounded-full bg-gradient-to-r from-purple-500 via-blue-500 to-emerald-400"
          />
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Round progress</span>
          <span className="text-sm font-mono font-semibold text-white tabular-nums">{currentRound} / {totalRounds}</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-700 overflow-hidden">
          <motion.div
            initial={{ width: 0 }} animate={{ width: `${roundPct}%` }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className="h-full rounded-full bg-gradient-to-r from-blue-500 to-purple-500"
          />
        </div>
      </div>

      <div className="border-t border-slate-700/60 pt-3">
        <p className="text-xs text-slate-400 mb-0.5">Convergence rate</p>
        <p className="text-sm font-mono text-white">
          {convergenceRate > 0 ? `${(convergenceRate * 100).toFixed(2)}% / round` : "—"}
        </p>
      </div>

      {status === "running" && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-900/30 border border-blue-500/30">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
            className="w-3.5 h-3.5 border-2 border-blue-400 border-t-transparent rounded-full shrink-0"
          />
          <span className="text-xs text-blue-300">Aggregating models…</span>
        </div>
      )}
      {status === "completed" && (
        <motion.div
          initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-900/30 border border-emerald-500/30"
        >
          <svg className="w-4 h-4 text-emerald-400 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          <span className="text-xs text-emerald-300 font-semibold">Training Completed</span>
        </motion.div>
      )}
    </div>
  );
};

export default CoordinatorCard;
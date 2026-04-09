/**
 * KPICards Component
 * Displays key performance indicator cards fetched from the real backend API.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { StatCard } from '../shared';
import { analyticsApi } from '../../utils/api';
import { formatDuration, formatNumber, formatRiskScore, formatDensity } from '../../utils/formatters';

interface KPIData {
  averageDensity: number;
  totalAlerts: number;
  peakRiskScore: number;
  sessionDuration: number;
}

interface KPICardsProps {
  autoRefresh?: boolean;
  refreshInterval?: number; // seconds
  timeRange?: number;       // minutes — passed to backend ?minutes= param
  className?: string;
}

export const KPICards: React.FC<KPICardsProps> = ({
  autoRefresh = true,
  refreshInterval = 10,
  timeRange = 60,
  className = '',
}) => {
  const [kpiData, setKpiData] = useState<KPIData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchKPIs = useCallback(async () => {
    try {
      setError(null);
      const data = await analyticsApi.getKPIs(timeRange);
      setKpiData({
        averageDensity: data.average_density ?? 0,
        totalAlerts: data.total_alerts ?? 0,
        peakRiskScore: data.peak_risk_score ?? 0,
        sessionDuration: data.session_duration_minutes ?? 0,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch KPIs');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchKPIs();
  }, [fetchKPIs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchKPIs, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchKPIs]);

  if (loading) {
    return (
      <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 ${className}`}>
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-slate-800 rounded-lg p-6 animate-pulse">
            <div className="h-4 bg-slate-700 rounded w-3/4 mb-2" />
            <div className="h-8 bg-slate-700 rounded w-1/2" />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-900/20 border border-red-500/30 rounded-lg p-4 ${className}`}>
        <p className="text-red-400 font-medium">Backend not connected</p>
        <p className="text-red-300 text-sm mt-1">{error}</p>
      </div>
    );
  }

  if (!kpiData) return null;

  return (
    <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 ${className}`}>
      <StatCard
        label="Average Density"
        value={formatDensity(kpiData.averageDensity)}
        icon={
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        }
        trend={kpiData.averageDensity > 0.5 ? 'up' : 'neutral'}
        className="bg-blue-900/20 border-blue-500/30"
      />
      <StatCard
        label="Total Alerts"
        value={formatNumber(kpiData.totalAlerts)}
        icon={
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        }
        trend={kpiData.totalAlerts > 0 ? 'up' : 'neutral'}
        className="bg-amber-900/20 border-amber-500/30"
      />
      <StatCard
        label="Peak Risk Score"
        value={formatRiskScore(kpiData.peakRiskScore)}
        icon={
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        }
        trend={kpiData.peakRiskScore > 75 ? 'up' : kpiData.peakRiskScore > 50 ? 'neutral' : 'down'}
        className="bg-red-900/20 border-red-500/30"
      />
      <StatCard
        label="Session Duration"
        value={formatDuration(kpiData.sessionDuration)}
        icon={
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        }
        trend="neutral"
        className="bg-green-900/20 border-green-500/30"
      />
    </div>
  );
};

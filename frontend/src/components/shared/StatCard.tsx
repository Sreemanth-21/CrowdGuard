/**
 * StatCard Component
 * Displays a metric with label, value, and optional icon
 */

import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  icon,
  trend,
  className = '',
}) => {
  const trendColor = {
    up: 'text-red-400',
    down: 'text-green-400',
    neutral: 'text-gray-400',
  };

  return (
    <div
      className={`
        bg-slate-800 rounded-lg p-4 border border-slate-700
        hover:border-slate-600 transition-colors
        ${className}
      `}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-1">{label}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
        </div>
        {icon && (
          <div className="text-gray-500 ml-2">
            {icon}
          </div>
        )}
      </div>
      {trend && (
        <div className={`text-xs mt-2 ${trendColor[trend]}`}>
          {trend === 'up' && '↑ Increasing'}
          {trend === 'down' && '↓ Decreasing'}
          {trend === 'neutral' && '→ Stable'}
        </div>
      )}
    </div>
  );
};

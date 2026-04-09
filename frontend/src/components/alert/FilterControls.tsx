/**
 * FilterControls Component
 * Provides filtering controls for alerts history page
 */

import React, { useState } from 'react';
import { RISK_LEVELS, ANOMALY_TYPES, ANOMALY_LABELS } from '../../utils/constants';

interface FilterState {
  riskLevels: string[];
  anomalyTypes: string[];
  startDate: string;
  endDate: string;
  dismissed: boolean | null;
}

interface FilterControlsProps {
  onFiltersChange: (filters: FilterState) => void;
  className?: string;
}

export const FilterControls: React.FC<FilterControlsProps> = ({
  onFiltersChange,
  className = '',
}) => {
  const [filters, setFilters] = useState<FilterState>({
    riskLevels: [],
    anomalyTypes: [],
    startDate: '',
    endDate: '',
    dismissed: null,
  });

  const updateFilters = (newFilters: Partial<FilterState>) => {
    const updatedFilters = { ...filters, ...newFilters };
    setFilters(updatedFilters);
    onFiltersChange(updatedFilters);
  };

  const clearFilters = () => {
    const clearedFilters: FilterState = {
      riskLevels: [],
      anomalyTypes: [],
      startDate: '',
      endDate: '',
      dismissed: null,
    };
    setFilters(clearedFilters);
    onFiltersChange(clearedFilters);
  };

  const toggleRiskLevel = (level: string) => {
    const newRiskLevels = filters.riskLevels.includes(level)
      ? filters.riskLevels.filter(l => l !== level)
      : [...filters.riskLevels, level];
    updateFilters({ riskLevels: newRiskLevels });
  };

  const toggleAnomalyType = (type: string) => {
    const newAnomalyTypes = filters.anomalyTypes.includes(type)
      ? filters.anomalyTypes.filter(t => t !== type)
      : [...filters.anomalyTypes, type];
    updateFilters({ anomalyTypes: newAnomalyTypes });
  };

  const hasActiveFilters = 
    filters.riskLevels.length > 0 ||
    filters.anomalyTypes.length > 0 ||
    filters.startDate ||
    filters.endDate ||
    filters.dismissed !== null;

  return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Filters</h3>
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      <div className="space-y-6">
        {/* Risk Level Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Risk Level
          </label>
          <div className="flex flex-wrap gap-2">
            {Object.values(RISK_LEVELS).map((level) => (
              <button
                key={level}
                onClick={() => toggleRiskLevel(level)}
                className={`
                  px-3 py-1 rounded-full text-sm font-medium transition-colors
                  ${filters.riskLevels.includes(level)
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                  }
                `}
              >
                {level}
              </button>
            ))}
          </div>
        </div>

        {/* Anomaly Type Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Anomaly Type
          </label>
          <div className="grid grid-cols-2 gap-2">
            {Object.values(ANOMALY_TYPES).map((type) => (
              <button
                key={type}
                onClick={() => toggleAnomalyType(type)}
                className={`
                  px-3 py-2 rounded text-sm font-medium transition-colors text-left
                  ${filters.anomalyTypes.includes(type)
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                  }
                `}
              >
                {ANOMALY_LABELS[type as keyof typeof ANOMALY_LABELS]}
              </button>
            ))}
          </div>
        </div>

        {/* Date Range Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Date Range
          </label>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">From</label>
              <input
                type="date"
                value={filters.startDate}
                onChange={(e) => updateFilters({ startDate: e.target.value })}
                className="w-full bg-slate-700 text-white px-3 py-2 rounded border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">To</label>
              <input
                type="date"
                value={filters.endDate}
                onChange={(e) => updateFilters({ endDate: e.target.value })}
                className="w-full bg-slate-700 text-white px-3 py-2 rounded border border-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Dismissed Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Status
          </label>
          <div className="flex space-x-2">
            <button
              onClick={() => updateFilters({ dismissed: null })}
              className={`
                px-3 py-2 rounded text-sm font-medium transition-colors
                ${filters.dismissed === null
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }
              `}
            >
              All
            </button>
            <button
              onClick={() => updateFilters({ dismissed: false })}
              className={`
                px-3 py-2 rounded text-sm font-medium transition-colors
                ${filters.dismissed === false
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }
              `}
            >
              Active
            </button>
            <button
              onClick={() => updateFilters({ dismissed: true })}
              className={`
                px-3 py-2 rounded text-sm font-medium transition-colors
                ${filters.dismissed === true
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }
              `}
            >
              Dismissed
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
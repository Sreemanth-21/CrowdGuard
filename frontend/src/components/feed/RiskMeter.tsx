/**
 * RiskMeter Component
 * Displays risk score and level with pulsing animation for critical levels
 */

import React from 'react';
import { useStore } from '../../store';
import { RiskBadge } from '../shared';
import { RISK_COLORS } from '../../utils/constants';

interface RiskMeterProps {
  className?: string;
}

export const RiskMeter: React.FC<RiskMeterProps> = ({ className = '' }) => {
  const riskScore = useStore((state) => state.video.riskScore);
  const riskLevel = useStore((state) => state.video.riskLevel);
  const density = useStore((state) => state.video.density);
  const personCount = useStore((state) => state.video.personCount);

  const getRiskPercentage = () => {
    return Math.min(Math.max(riskScore, 0), 100);
  };

  const getRiskColor = () => {
    return RISK_COLORS[riskLevel as keyof typeof RISK_COLORS];
  };

  const isCritical = riskLevel === 'CRITICAL';

  return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-white mb-6">Risk Assessment</h3>
      
      {/* Main risk score display */}
      <div className="text-center mb-6">
        <div 
          className={`
            relative inline-flex items-center justify-center w-32 h-32 rounded-full border-4
            transition-all duration-500
            ${getRiskColor().border}
            ${isCritical ? 'animate-pulse shadow-lg shadow-red-500/50' : ''}
          `}
        >
          {/* Pulsing glow for critical */}
          {isCritical && (
            <div className="absolute inset-0 rounded-full bg-red-500 opacity-20 animate-ping"></div>
          )}
          
          <div className="text-center z-10">
            <div
              key={Math.round(riskScore)}
              className={`text-3xl font-bold transition-colors duration-500 ${getRiskColor().text}`}
            >
              {Math.round(riskScore)}
            </div>
            <div className="text-xs text-gray-400 uppercase tracking-wide">
              Risk Score
            </div>
          </div>
        </div>
      </div>

      {/* Risk level badge */}
      <div className="flex justify-center mb-6">
        <RiskBadge level={riskLevel as any} size="lg" />
      </div>

      {/* Risk breakdown */}
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Current Level</span>
          <span className={`font-semibold ${getRiskColor().text}`}>
            {riskLevel}
          </span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Person Count</span>
          <span className="text-white font-semibold">{personCount}</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Density</span>
          <span className="text-white font-semibold">
            {(density * 100).toFixed(1)}%
          </span>
        </div>

        {/* Risk progress bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-gray-400">
            <span>SAFE</span>
            <span>CRITICAL</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-500 ${getRiskColor().bg}`}
              style={{ width: `${getRiskPercentage()}%` }}
            ></div>
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>0</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
          </div>
        </div>
      </div>

      {/* Risk level thresholds */}
      <div className="mt-6 pt-4 border-t border-slate-700">
        <div className="text-xs text-gray-400 mb-2">Risk Levels</div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-gray-300">SAFE (0-25)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
            <span className="text-gray-300">CAUTION (26-50)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
            <span className="text-gray-300">WARNING (51-75)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
            <span className="text-gray-300">CRITICAL (76-100)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
/**
 * RiskBadge Component
 * Displays risk level with color coding
 */

import React from 'react';
import { RISK_COLORS } from '../../utils/constants';

interface RiskBadgeProps {
  level: 'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  level,
  size = 'md',
  className = '',
}) => {
  const colors = RISK_COLORS[level];

  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm',
    lg: 'px-4 py-2 text-base',
  };

  return (
    <div
      className={`
        inline-flex items-center rounded-full font-semibold text-white
        ${sizeClasses[size]}
        ${colors.bg}
        ${className}
      `}
    >
      {level}
    </div>
  );
};

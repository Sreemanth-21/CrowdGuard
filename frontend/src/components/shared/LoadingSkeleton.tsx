/**
 * LoadingSkeleton Component
 * Animated skeleton loader for loading states
 */

import React from 'react';

interface LoadingSkeletonProps {
  width?: string;
  height?: string;
  count?: number;
  circle?: boolean;
  className?: string;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  width = 'w-full',
  height = 'h-4',
  count = 1,
  circle = false,
  className = '',
}) => {
  const skeletons = Array.from({ length: count });

  return (
    <div className={className}>
      {skeletons.map((_, index) => (
        <div
          key={index}
          className={`
            bg-gradient-to-r from-slate-700 via-slate-600 to-slate-700
            animate-pulse
            ${width} ${height}
            ${circle ? 'rounded-full' : 'rounded'}
            ${index < count - 1 ? 'mb-3' : ''}
          `}
        />
      ))}
    </div>
  );
};

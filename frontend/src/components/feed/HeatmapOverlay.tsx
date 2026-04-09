/**
 * HeatmapOverlay Component
 * Renders heatmap overlay on video feed with configurable opacity and toggle
 */

import React from 'react';
import { useStore } from '../../store';

interface HeatmapOverlayProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  opacity?: number;
  className?: string;
}

export const HeatmapOverlay: React.FC<HeatmapOverlayProps> = ({
  canvasRef,
  opacity = 0.6,
  className = '',
}) => {
  const heatmapEnabled = useStore((state) => state.video.heatmapEnabled);
  const setVideoState = useStore((state) => state.setVideoState);
  const density = useStore((state) => state.video.density);
  const densityZone = useStore((state) => state.video.densityZone);

  const toggleHeatmap = () => {
    setVideoState({ heatmapEnabled: !heatmapEnabled });
  };

  const getDensityColor = (zone: string): string => {
    switch (zone) {
      case 'LOW': return 'rgba(0, 255, 0, 0.3)';
      case 'MODERATE': return 'rgba(255, 255, 0, 0.4)';
      case 'HIGH': return 'rgba(255, 165, 0, 0.5)';
      case 'CRITICAL': return 'rgba(255, 0, 0, 0.6)';
      default: return 'rgba(0, 255, 0, 0.3)';
    }
  };

  if (!heatmapEnabled) {
    return (
      <div className={`absolute top-4 left-4 ${className}`}>
        <button
          onClick={toggleHeatmap}
          className="bg-slate-800 bg-opacity-80 text-white px-3 py-2 rounded-lg text-sm hover:bg-slate-700 transition-colors"
        >
          Show Heatmap
        </button>
      </div>
    );
  }

  return (
    <div className={`absolute inset-0 pointer-events-none ${className}`}>
      {/* Heatmap overlay */}
      <div
        className="absolute inset-0 rounded"
        style={{
          background: getDensityColor(densityZone),
          opacity: opacity,
        }}
      />
      
      {/* Heatmap controls */}
      <div className="absolute top-4 left-4 pointer-events-auto">
        <div className="bg-slate-800 bg-opacity-90 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-white text-sm font-medium">Heatmap</span>
            <button
              onClick={toggleHeatmap}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>
          
          <div className="text-xs text-gray-300">
            <div>Density: {(density * 100).toFixed(1)}%</div>
            <div>Zone: <span className={getZoneTextColor(densityZone)}>{densityZone}</span></div>
          </div>
          
          {/* Density legend */}
          <div className="space-y-1">
            <div className="text-xs text-gray-400">Legend:</div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-green-500 rounded opacity-60"></div>
              <span className="text-xs text-gray-300">Low</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-yellow-500 rounded opacity-60"></div>
              <span className="text-xs text-gray-300">Moderate</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-orange-500 rounded opacity-60"></div>
              <span className="text-xs text-gray-300">High</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-red-500 rounded opacity-60"></div>
              <span className="text-xs text-gray-300">Critical</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const getZoneTextColor = (zone: string): string => {
  switch (zone) {
    case 'LOW': return 'text-green-400';
    case 'MODERATE': return 'text-yellow-400';
    case 'HIGH': return 'text-orange-400';
    case 'CRITICAL': return 'text-red-400';
    default: return 'text-gray-400';
  }
};
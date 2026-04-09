/**
 * AlertPanel Component
 * Displays recent alerts with auto-scroll and real-time updates
 */

import React, { useEffect, useRef } from 'react';
import { useStore } from '../../store';
import { AlertCard } from './AlertCard';

interface AlertPanelProps {
  maxAlerts?: number;
  className?: string;
}

export const AlertPanel: React.FC<AlertPanelProps> = ({
  maxAlerts = 10,
  className = '',
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Get alerts from store
  const alerts = useStore((state) => state.alerts.activeAlerts);
  const dismissAlert = useStore((state) => state.dismissAlert);

  // Auto-scroll to new alerts
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [alerts]);

  const handleDismissAlert = (alertId: string) => {
    dismissAlert(alertId);
  };

  // Get the most recent alerts (limited by maxAlerts)
  const recentAlerts = alerts.slice(0, maxAlerts);

  return (
    <div className={`bg-slate-800 rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Recent Alerts</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-400">Live</span>
        </div>
      </div>

      {/* Alerts list */}
      <div 
        ref={scrollRef}
        className="space-y-3 max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800"
      >
        {recentAlerts.length > 0 ? (
          recentAlerts.map((alert) => (
            <AlertCard
              key={alert.alert_id}
              alert={alert}
              onDismiss={handleDismissAlert}
              className="animate-fade-in"
            />
          ))
        ) : (
          <div className="text-center py-8">
            <div className="text-gray-500 text-4xl mb-2">🔔</div>
            <p className="text-gray-400">No recent alerts</p>
            <p className="text-gray-500 text-sm mt-1">
              Alerts will appear here when anomalies are detected
            </p>
          </div>
        )}
      </div>

      {/* Alert count indicator */}
      {recentAlerts.length > 0 && (
        <div className="mt-4 pt-3 border-t border-slate-700">
          <div className="flex justify-between items-center text-sm text-gray-400">
            <span>Showing {recentAlerts.length} of {alerts.length} alerts</span>
            {alerts.length > maxAlerts && (
              <span className="text-blue-400">
                +{alerts.length - maxAlerts} more in history
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
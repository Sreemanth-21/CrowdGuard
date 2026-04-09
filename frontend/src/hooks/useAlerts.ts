/**
 * Alerts Hook for CrowdGuard
 * Manages alert fetching, filtering, and dismissal
 */

import { useCallback } from 'react';
import { alertsApi, AlertFilters, PaginationParams } from '../utils/api';
import { useStore } from '../store';

export function useAlerts() {
  const setWsError = useStore((state) => state.setWsError);

  const fetchAlerts = useCallback(
    async (filters: AlertFilters = {}, pagination: PaginationParams = {}) => {
      try {
        const response = await alertsApi.list(filters, pagination);
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch alerts';
        setWsError(message);
        throw error;
      }
    },
    [setWsError]
  );

  const getAlert = useCallback(
    async (alertId: string) => {
      try {
        return await alertsApi.get(alertId);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch alert';
        setWsError(message);
        throw error;
      }
    },
    [setWsError]
  );

  const dismissAlert = useCallback(
    async (alertId: string) => {
      try {
        const response = await alertsApi.dismiss(alertId);
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to dismiss alert';
        setWsError(message);
        throw error;
      }
    },
    [setWsError]
  );

  const bulkDismissAlerts = useCallback(
    async (alertIds: string[]) => {
      try {
        const response = await alertsApi.bulkDismiss(alertIds);
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to dismiss alerts';
        setWsError(message);
        throw error;
      }
    },
    [setWsError]
  );

  const getSummary = useCallback(async () => {
    try {
      return await alertsApi.getSummary();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch alert summary';
      setWsError(message);
      throw error;
    }
  }, [setWsError]);

  const exportAlerts = useCallback(
    async (filters: AlertFilters = {}) => {
      try {
        const blob = await alertsApi.export(filters);
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `alerts-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to export alerts';
        setWsError(message);
        throw error;
      }
    },
    [setWsError]
  );

  const getSnapshotUrl = useCallback((alertId: string) => {
    return alertsApi.getSnapshotUrl(alertId);
  }, []);

  return {
    fetchAlerts,
    getAlert,
    dismissAlert,
    bulkDismissAlerts,
    getSummary,
    exportAlerts,
    getSnapshotUrl,
  };
}

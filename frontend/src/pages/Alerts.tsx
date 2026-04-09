import { useState, useEffect, useCallback } from 'react';
import { PageWrapper } from '../components/layout';
import { FilterControls } from '../components/alert/FilterControls';
import { AlertsTable } from '../components/alert/AlertsTable';
import { AlertDetailModal } from '../components/alert/AlertDetailModal';
import { useAlerts } from '../hooks/useAlerts';
import { LoadingSkeleton } from '../components/shared/LoadingSkeleton';
import { Alert } from '../utils/api';
import { useStore } from '../store';

interface FilterState {
  riskLevels: string[];
  anomalyTypes: string[];
  startDate: string;
  endDate: string;
  dismissed: boolean | null;
}

// Transform API Alert to component Alert format
const transformAlert = (apiAlert: Alert) => ({
  id: apiAlert.alert_id,
  timestamp: apiAlert.timestamp,
  anomalyType: apiAlert.anomaly_type,
  riskLevel: apiAlert.risk_level,
  confidence: apiAlert.confidence_score,          // backend field is confidence_score
  personCount: apiAlert.affected_persons ?? apiAlert.person_count ?? 0,
  density: apiAlert.density ?? 0,
  description: apiAlert.description,
  ...(apiAlert.frame_snapshot_path && { snapshotPath: apiAlert.frame_snapshot_path }),
  dismissed: apiAlert.is_dismissed ?? apiAlert.dismissed ?? false,
});

function Alerts() {
  const [alerts, setAlerts] = useState<ReturnType<typeof transformAlert>[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  const [filters, setFilters] = useState<FilterState>({
    riskLevels: [],
    anomalyTypes: [],
    startDate: '',
    endDate: '',
    dismissed: null,
  });

  const [pagination, setPagination] = useState({ page: 1, limit: 20 });
  const [selectedAlert, setSelectedAlert] = useState<ReturnType<typeof transformAlert> | null>(null);
  const [selectedAlerts, setSelectedAlerts] = useState<string[]>([]);
  const [sortColumn, setSortColumn] = useState<string>('timestamp');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  const { fetchAlerts, bulkDismissAlerts, exportAlerts } = useAlerts();

  // Real-time: pick up new alerts pushed via WebSocket
  const wsAlerts = useStore((state) => state.alerts.activeAlerts);

  const loadAlerts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const apiFilters = {
        risk_level: filters.riskLevels,
        anomaly_type: filters.anomalyTypes,
        start_date: filters.startDate || undefined,
        end_date: filters.endDate || undefined,
        dismissed: filters.dismissed !== null ? filters.dismissed : undefined,
      };
      const apiPagination = {
        page: pagination.page,
        limit: pagination.limit,
        sort_by: sortColumn,
        sort_order: sortDirection,
      };
      const response = await fetchAlerts(apiFilters, apiPagination);
      setAlerts(response.alerts.map(transformAlert));
      setTotalCount(response.total);
      setTotalPages(response.pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  }, [filters, pagination, sortColumn, sortDirection, fetchAlerts]);

  useEffect(() => {
    loadAlerts();
  }, [loadAlerts]);

  // Refresh when a new WS alert arrives (only if on page 1 with default sort)
  useEffect(() => {
    if (wsAlerts.length > 0 && pagination.page === 1) {
      loadAlerts();
    }
  }, [wsAlerts.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFiltersChange = (newFilters: FilterState) => {
    setFilters(newFilters);
    setPagination((prev) => ({ ...prev, page: 1 }));
  };

  const handlePageChange = (newPage: number) => {
    setPagination((prev) => ({ ...prev, page: newPage }));
  };

  const handleSort = (column: string, direction: 'asc' | 'desc') => {
    setSortColumn(column);
    setSortDirection(direction);
    setPagination((prev) => ({ ...prev, page: 1 }));
  };

  const handleSelectAlert = (alertId: string) => {
    setSelectedAlerts((prev) =>
      prev.includes(alertId) ? prev.filter((id) => id !== alertId) : [...prev, alertId]
    );
  };

  const handleSelectAll = (selected: boolean) => {
    setSelectedAlerts(selected ? alerts.map((a) => a.id) : []);
  };

  const handleBulkDismiss = async () => {
    if (selectedAlerts.length === 0) return;
    try {
      await bulkDismissAlerts(selectedAlerts);
      setSelectedAlerts([]);
      loadAlerts();
    } catch (err) {
      console.error('Failed to dismiss alerts:', err);
    }
  };

  const handleExport = async () => {
    try {
      await exportAlerts({
        risk_level: filters.riskLevels,
        anomaly_type: filters.anomalyTypes,
        start_date: filters.startDate || undefined,
        end_date: filters.endDate || undefined,
        dismissed: filters.dismissed !== null ? filters.dismissed : undefined,
      });
    } catch (err) {
      console.error('Failed to export alerts:', err);
    }
  };

  return (
    <PageWrapper>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-heading font-bold mb-2">Alert History</h1>
            <p className="font-body text-navy-300">
              View and manage historical alerts with filtering and search
            </p>
          </div>
          <div className="flex gap-3">
            {selectedAlerts.length > 0 && (
              <button
                onClick={handleBulkDismiss}
                className="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-lg font-medium transition-colors"
              >
                Dismiss Selected ({selectedAlerts.length})
              </button>
            )}
            <button
              onClick={handleExport}
              className="px-4 py-2 bg-navy-600 hover:bg-navy-700 text-white rounded-lg font-medium transition-colors"
            >
              Export CSV
            </button>
          </div>
        </div>

        <FilterControls onFiltersChange={handleFiltersChange} />

        {loading && (
          <div className="space-y-4">
            <LoadingSkeleton className="h-12 w-full" />
            <LoadingSkeleton className="h-64 w-full" />
          </div>
        )}

        {error && (
          <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
            <p className="text-red-400 font-medium">Backend not connected</p>
            <p className="text-red-300 text-sm mt-1">{error}</p>
          </div>
        )}

        {!loading && !error && (
          <>
            <AlertsTable
              alerts={alerts}
              selectedAlerts={selectedAlerts}
              onSelectAlert={handleSelectAlert}
              onSelectAll={handleSelectAll}
              onSort={handleSort}
              onViewDetails={setSelectedAlert}
            />

            {totalPages > 1 && (
              <div className="flex justify-center items-center space-x-4">
                <button
                  onClick={() => handlePageChange(pagination.page - 1)}
                  disabled={pagination.page === 1}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
                >
                  Previous
                </button>
                <span className="text-navy-300">
                  Page {pagination.page} of {totalPages}
                </span>
                <button
                  onClick={() => handlePageChange(pagination.page + 1)}
                  disabled={pagination.page === totalPages}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
                >
                  Next
                </button>
              </div>
            )}

            <div className="text-sm text-navy-300 text-center">
              {totalCount === 0
                ? 'No alerts found'
                : `Showing ${(pagination.page - 1) * pagination.limit + 1}–${Math.min(
                    pagination.page * pagination.limit,
                    totalCount
                  )} of ${totalCount} alerts`}
            </div>
          </>
        )}

        <AlertDetailModal
          alert={selectedAlert}
          isOpen={selectedAlert !== null}
          onClose={() => setSelectedAlert(null)}
        />
      </div>
    </PageWrapper>
  );
}

export default Alerts;

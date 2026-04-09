import { StateCreator } from 'zustand';
import { StoreState } from './index';

export interface Alert {
  alert_id: string;
  session_id?: string;
  timestamp: string;
  anomaly_type: string;
  risk_level: 'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL';
  confidence_score: number;
  description: string;
  frame_snapshot_path?: string;
  snapshot?: string; // base64 image
  affected_persons?: number;
  location_x?: number;
  location_y?: number;
  is_dismissed?: boolean;
  dismissed_at?: string;
}

export interface AlertFilters {
  riskLevels: string[];
  anomalyTypes: string[];
  dateRange: { start: string | null; end: string | null };
  dismissed: boolean;
}

export interface AlertPagination {
  page: number;
  perPage: number;
  total: number;
}

export interface AlertState {
  activeAlerts: Alert[]; // Last 10 alerts for dashboard panel
  alertHistory: Alert[]; // For alerts page
  filters: AlertFilters;
  pagination: AlertPagination;
  selectedAlerts: string[]; // Alert IDs
}

export interface AlertSlice {
  alerts: AlertState;
  addAlert: (alert: Alert) => void;
  setAlertHistory: (alerts: Alert[]) => void;
  setAlertFilters: (filters: Partial<AlertFilters>) => void;
  setAlertPagination: (pagination: Partial<AlertPagination>) => void;
  toggleAlertSelection: (alertId: string) => void;
  clearAlertSelection: () => void;
  dismissAlert: (alertId: string) => void;
  dismissAlerts: (alertIds: string[]) => void;
}

const initialAlertState: AlertState = {
  activeAlerts: [],
  alertHistory: [],
  filters: {
    riskLevels: [],
    anomalyTypes: [],
    dateRange: { start: null, end: null },
    dismissed: false,
  },
  pagination: {
    page: 1,
    perPage: 20,
    total: 0,
  },
  selectedAlerts: [],
};

export const alertSlice: StateCreator<StoreState, [], [], AlertSlice> = (set) => ({
  alerts: initialAlertState,
  
  addAlert: (alert: Alert) =>
    set((state: StoreState) => ({
      alerts: {
        ...state.alerts,
        activeAlerts: [alert, ...state.alerts.activeAlerts].slice(0, 10),
      },
    })),
  
  setAlertHistory: (alerts: Alert[]) =>
    set((state: StoreState) => ({
      alerts: {
        ...state.alerts,
        alertHistory: alerts,
      },
    })),
  
  setAlertFilters: (filters: Partial<AlertFilters>) =>
    set((state: StoreState) => ({
      alerts: {
        ...state.alerts,
        filters: { ...state.alerts.filters, ...filters },
      },
    })),
  
  setAlertPagination: (pagination: Partial<AlertPagination>) =>
    set((state: StoreState) => ({
      alerts: {
        ...state.alerts,
        pagination: { ...state.alerts.pagination, ...pagination },
      },
    })),
  
  toggleAlertSelection: (alertId: string) =>
    set((state: StoreState) => {
      const isSelected = state.alerts.selectedAlerts.includes(alertId);
      return {
        alerts: {
          ...state.alerts,
          selectedAlerts: isSelected
            ? state.alerts.selectedAlerts.filter((id: string) => id !== alertId)
            : [...state.alerts.selectedAlerts, alertId],
        },
      };
    }),
  
  clearAlertSelection: () =>
    set((state: StoreState) => ({
      alerts: {
        ...state.alerts,
        selectedAlerts: [],
      },
    })),
  
  dismissAlert: (alertId: string) =>
    set((state: StoreState) => ({
      alerts: {
        ...state.alerts,
        activeAlerts: state.alerts.activeAlerts.map((alert: Alert) =>
          alert.alert_id === alertId
            ? { ...alert, is_dismissed: true, dismissed_at: new Date().toISOString() }
            : alert
        ),
        alertHistory: state.alerts.alertHistory.map((alert: Alert) =>
          alert.alert_id === alertId
            ? { ...alert, is_dismissed: true, dismissed_at: new Date().toISOString() }
            : alert
        ),
      },
    })),
  
  dismissAlerts: (alertIds: string[]) =>
    set((state: StoreState) => {
      const dismissedAt = new Date().toISOString();
      return {
        alerts: {
          ...state.alerts,
          activeAlerts: state.alerts.activeAlerts.map((alert: Alert) =>
            alertIds.includes(alert.alert_id)
              ? { ...alert, is_dismissed: true, dismissed_at: dismissedAt }
              : alert
          ),
          alertHistory: state.alerts.alertHistory.map((alert: Alert) =>
            alertIds.includes(alert.alert_id)
              ? { ...alert, is_dismissed: true, dismissed_at: dismissedAt }
              : alert
          ),
        },
      };
    }),
});

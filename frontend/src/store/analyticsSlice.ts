import { StateCreator } from 'zustand';
import { StoreState } from './index';

export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
}

export interface DensityDataPoint {
  timestamp: string;
  density: number;
  person_count: number;
}

export interface RiskDataPoint {
  timestamp: string;
  risk_score: number;
  risk_level: string;
}

export interface AlertFrequencyData {
  anomaly_type: string;
  count: number;
}

export interface PersonCountBin {
  range: string;
  count: number;
}

export interface KPIMetrics {
  averageDensity: number;
  totalAlerts: number;
  peakRiskScore: number;
  sessionDuration: number;
}

export interface AnalyticsState {
  densityTimeseries: DensityDataPoint[];
  riskTimeseries: RiskDataPoint[];
  alertFrequency: AlertFrequencyData[];
  personCountHistogram: PersonCountBin[];
  kpis: KPIMetrics;
}

export interface AnalyticsSlice {
  analytics: AnalyticsState;
  setDensityTimeseries: (data: DensityDataPoint[]) => void;
  setRiskTimeseries: (data: RiskDataPoint[]) => void;
  setAlertFrequency: (data: AlertFrequencyData[]) => void;
  setPersonCountHistogram: (data: PersonCountBin[]) => void;
  setKPIs: (kpis: Partial<KPIMetrics>) => void;
  resetAnalytics: () => void;
}

const initialAnalyticsState: AnalyticsState = {
  densityTimeseries: [],
  riskTimeseries: [],
  alertFrequency: [],
  personCountHistogram: [],
  kpis: {
    averageDensity: 0,
    totalAlerts: 0,
    peakRiskScore: 0,
    sessionDuration: 0,
  },
};

export const analyticsSlice: StateCreator<StoreState, [], [], AnalyticsSlice> = (set) => ({
  analytics: initialAnalyticsState,
  
  setDensityTimeseries: (data: DensityDataPoint[]) =>
    set((state: StoreState) => ({
      analytics: {
        ...state.analytics,
        densityTimeseries: data,
      },
    })),
  
  setRiskTimeseries: (data: RiskDataPoint[]) =>
    set((state: StoreState) => ({
      analytics: {
        ...state.analytics,
        riskTimeseries: data,
      },
    })),
  
  setAlertFrequency: (data: AlertFrequencyData[]) =>
    set((state: StoreState) => ({
      analytics: {
        ...state.analytics,
        alertFrequency: data,
      },
    })),
  
  setPersonCountHistogram: (data: PersonCountBin[]) =>
    set((state: StoreState) => ({
      analytics: {
        ...state.analytics,
        personCountHistogram: data,
      },
    })),
  
  setKPIs: (kpis: Partial<KPIMetrics>) =>
    set((state: StoreState) => ({
      analytics: {
        ...state.analytics,
        kpis: { ...state.analytics.kpis, ...kpis },
      },
    })),
  
  resetAnalytics: () =>
    set(() => ({
      analytics: initialAnalyticsState,
    })),
});

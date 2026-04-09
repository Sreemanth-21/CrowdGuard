import { StateCreator } from 'zustand';
import { StoreState } from './index';

export interface VideoSource {
  type: 'webcam' | 'upload';
  name: string;
}

export interface Anomaly {
  type: string;
  confidence: number;
  location?: { x: number; y: number };
}

export interface SessionStats {
  uptime: number;
  framesProcessed: number;
  totalPersons: number;
  totalAlerts: number;
  peakRiskScore: number;
  fps: number;
}

export interface VideoState {
  isProcessing: boolean;
  source: VideoSource | null;
  personCount: number;
  riskScore: number;
  riskLevel: 'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL';
  density: number;
  densityZone: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  anomalies: Anomaly[];
  sessionStats: SessionStats;
  currentFrame: string | null; // base64 image
  heatmapEnabled: boolean;
}

export interface VideoSlice {
  video: VideoState;
  setVideoState: (updates: Partial<VideoState>) => void;
  resetVideoState: () => void;
}

const initialVideoState: VideoState = {
  isProcessing: false,
  source: null,
  personCount: 0,
  riskScore: 0,
  riskLevel: 'SAFE',
  density: 0,
  densityZone: 'LOW',
  anomalies: [],
  sessionStats: {
    uptime: 0,
    framesProcessed: 0,
    totalPersons: 0,
    totalAlerts: 0,
    peakRiskScore: 0,
    fps: 0,
  },
  currentFrame: null,
  heatmapEnabled: true,
};

export const videoSlice: StateCreator<StoreState, [], [], VideoSlice> = (set) => ({
  video: initialVideoState,
  
  setVideoState: (updates) =>
    set((state) => ({
      video: { ...state.video, ...updates },
    })),
  
  resetVideoState: () =>
    set(() => ({
      video: initialVideoState,
    })),
});

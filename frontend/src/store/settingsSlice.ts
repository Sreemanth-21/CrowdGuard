import { StateCreator } from 'zustand';
import { StoreState } from './index';

export interface SettingsState {
  confidenceThreshold: number;
  modelVariant: 'nano' | 'small' | 'medium';
  highDensityThreshold: number;
  cooldownPeriod: number;
  heatmapOpacity: number;
  rapidMovementThreshold: number;
  crowdSurgeThreshold: number;
  stationaryCrowdThreshold: number;
  stationaryVelocityThreshold: number;
  fightingIouThreshold: number;
  fightingVelocityThreshold: number;
}

export interface SettingsSlice {
  settings: SettingsState;
  updateSettings: (updates: Partial<SettingsState>) => void;
  resetSettings: () => void;
}

const defaultSettings: SettingsState = {
  confidenceThreshold: 0.5,
  modelVariant: 'nano',
  highDensityThreshold: 0.7,
  cooldownPeriod: 10,
  heatmapOpacity: 0.6,
  rapidMovementThreshold: 25,
  crowdSurgeThreshold: 0.3,
  stationaryCrowdThreshold: 0.5,
  stationaryVelocityThreshold: 3,
  fightingIouThreshold: 0.3,
  fightingVelocityThreshold: 20,
};

export const settingsSlice: StateCreator<StoreState, [], [], SettingsSlice> = (set) => ({
  settings: defaultSettings,
  
  updateSettings: (updates) =>
    set((state) => ({
      settings: { ...state.settings, ...updates },
    })),
  
  resetSettings: () =>
    set(() => ({
      settings: defaultSettings,
    })),
});

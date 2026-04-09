import { create } from 'zustand';
import { videoSlice, VideoSlice } from './videoSlice';
import { alertSlice, AlertSlice } from './alertSlice';
import { analyticsSlice, AnalyticsSlice } from './analyticsSlice';
import { settingsSlice, SettingsSlice } from './settingsSlice';
import { wsSlice, WsSlice } from './wsSlice';

export type StoreState = VideoSlice & AlertSlice & AnalyticsSlice & SettingsSlice & WsSlice;

export const useStore = create<StoreState>()((...a) => ({
  ...videoSlice(...a),
  ...alertSlice(...a),
  ...analyticsSlice(...a),
  ...settingsSlice(...a),
  ...wsSlice(...a),
}));

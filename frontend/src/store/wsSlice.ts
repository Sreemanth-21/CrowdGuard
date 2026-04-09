import { StateCreator } from 'zustand';
import { StoreState } from './index';

export interface WebSocketMessage {
  type: string;
  payload: any;
}

export interface WsState {
  connected: boolean;
  reconnecting: boolean;
  lastMessage: WebSocketMessage | null;
  reconnectAttempts: number;
  backoffDelay: number;
  error: string | null;
}

export interface WsSlice {
  ws: WsState;
  setWsConnected: (connected: boolean) => void;
  setWsReconnecting: (reconnecting: boolean) => void;
  setWsLastMessage: (message: WebSocketMessage) => void;
  incrementReconnectAttempts: () => void;
  resetReconnectAttempts: () => void;
  setBackoffDelay: (delay: number) => void;
  setWsError: (error: string | null) => void;
  resetWsState: () => void;
}

const initialWsState: WsState = {
  connected: false,
  reconnecting: false,
  lastMessage: null,
  reconnectAttempts: 0,
  backoffDelay: 1000,
  error: null,
};

export const wsSlice: StateCreator<StoreState, [], [], WsSlice> = (set) => ({
  ws: initialWsState,
  
  setWsConnected: (connected) =>
    set((state) => ({
      ws: { ...state.ws, connected, reconnecting: false, error: null },
    })),
  
  setWsReconnecting: (reconnecting) =>
    set((state) => ({
      ws: { ...state.ws, reconnecting },
    })),
  
  setWsLastMessage: (message) =>
    set((state) => ({
      ws: { ...state.ws, lastMessage: message },
    })),
  
  incrementReconnectAttempts: () =>
    set((state) => ({
      ws: {
        ...state.ws,
        reconnectAttempts: state.ws.reconnectAttempts + 1,
        backoffDelay: Math.min(state.ws.backoffDelay * 2, 8000), // Cap at 8 seconds
      },
    })),
  
  resetReconnectAttempts: () =>
    set((state) => ({
      ws: {
        ...state.ws,
        reconnectAttempts: 0,
        backoffDelay: 1000,
      },
    })),
  
  setBackoffDelay: (delay) =>
    set((state) => ({
      ws: { ...state.ws, backoffDelay: delay },
    })),
  
  setWsError: (error) =>
    set((state) => ({
      ws: { ...state.ws, error },
    })),
  
  resetWsState: () =>
    set(() => ({
      ws: initialWsState,
    })),
});

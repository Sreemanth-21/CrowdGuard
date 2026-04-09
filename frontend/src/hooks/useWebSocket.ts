/**
 * WebSocket Hook for CrowdGuard
 * Connects directly to the FastAPI backend WebSocket (bypasses Vite proxy
 * to avoid ECONNABORTED on long-lived connections).
 * Implements unlimited exponential-backoff reconnection.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '../store';

// Always connect directly to the backend — never through the Vite proxy.
// The Vite proxy tears down long-lived WS connections (ECONNABORTED).
const WS_URL = 'ws://localhost:8000/ws';

// Backoff delays in ms; last value is reused indefinitely
const BACKOFF = [1000, 2000, 4000, 8000, 15000];

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const attemptsRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Prevent reconnect after intentional disconnect
  const stoppedRef = useRef(false);

  const isConnected = useStore((state) => state.ws.connected);
  const isReconnecting = useStore((state) => state.ws.reconnecting);

  const setVideoState = useStore((state) => state.setVideoState);
  const addAlert = useStore((state) => state.addAlert);
  const setWsError = useStore((state) => state.setWsError);
  const setWsConnected = useStore((state) => state.setWsConnected);
  const setWsReconnecting = useStore((state) => state.setWsReconnecting);

  const connect = useCallback(() => {
    if (stoppedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('[WS] Connected to', WS_URL);
        setWsConnected(true);
        setWsError(null);
        attemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          // Backend sends flat-key messages: { type, image, person_count, ... }
          // It also wraps some messages as { type, payload: { ... } }
          // Normalise both shapes here.
          const type: string = msg.type;
          const data = msg.payload ?? msg; // prefer payload, fall back to root

          switch (type) {
            case 'frame': {
              // image may be at root or inside payload
              const image = msg.image ?? data.image ?? null;
              setVideoState({
                currentFrame: image,
                personCount: data.person_count ?? msg.person_count ?? 0,
                riskScore: data.risk_score ?? msg.risk_score ?? 0,
                riskLevel: (data.risk_level ?? msg.risk_level ?? 'SAFE') as
                  'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL',
                anomalies: data.anomalies ?? msg.anomalies ?? [],
                density: data.density ?? msg.density ?? 0,
                isProcessing: true,
              });

              const stats = data.session_stats ?? msg.session_stats;
              if (stats) {
                setVideoState({
                  sessionStats: {
                    uptime: stats.uptime_seconds ?? stats.uptime ?? 0,
                    framesProcessed: stats.frames_processed ?? 0,
                    totalPersons: stats.total_persons ?? 0,
                    totalAlerts: stats.total_alerts ?? 0,
                    peakRiskScore: stats.peak_risk_score ?? 0,
                    fps: stats.fps ?? 0,
                  },
                });
              }
              break;
            }

            case 'alert': {
              addAlert({
                alert_id: data.alert_id ?? msg.id ?? '',
                timestamp: data.timestamp ?? msg.timestamp ?? new Date().toISOString(),
                anomaly_type: data.anomaly_type ?? msg.anomaly_type ?? '',
                risk_level: (data.risk_level ?? msg.risk_level ?? 'SAFE') as
                  'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL',
                confidence_score: data.confidence ?? msg.confidence ?? 0,
                affected_persons: data.affected_persons ?? msg.person_count ?? 0,
                description: data.description ?? msg.description ?? '',
                frame_snapshot_path: data.snapshot_path ?? msg.snapshot_path ?? '',
              });
              break;
            }

            case 'connected':
              console.log('[WS] Server acknowledged connection:', data);
              break;

            case 'status':
              console.log('[WS] Status:', data);
              break;

            case 'error':
              console.error('[WS] Server error:', data);
              setWsError(data.message ?? msg.message ?? 'Server error');
              break;

            default:
              console.warn('[WS] Unknown message type:', type, msg);
          }
        } catch (err) {
          console.error('[WS] Failed to parse message:', err, event.data);
        }
      };

      ws.onerror = (ev) => {
        console.error('[WS] Socket error:', ev);
        setWsError('WebSocket connection error');
      };

      ws.onclose = (ev) => {
        console.log(`[WS] Closed (code=${ev.code}, clean=${ev.wasClean})`);
        setWsConnected(false);
        wsRef.current = null;

        if (stoppedRef.current) return;

        // Schedule reconnect with capped exponential backoff
        const delay = BACKOFF[Math.min(attemptsRef.current, BACKOFF.length - 1)];
        attemptsRef.current += 1;
        setWsReconnecting(true);
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${attemptsRef.current})`);
        timerRef.current = setTimeout(connect, delay);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[WS] Failed to create WebSocket:', err);
      setWsError('Failed to connect to WebSocket');
    }
  }, [setVideoState, addAlert, setWsError, setWsConnected, setWsReconnecting]);

  const disconnect = useCallback(() => {
    stoppedRef.current = true;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsConnected(false);
    setWsReconnecting(false);
  }, [setWsConnected, setWsReconnecting]);

  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('[WS] Cannot send — not connected');
    }
  }, []);

  useEffect(() => {
    stoppedRef.current = false;
    connect();
    return () => {
      disconnect();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { isConnected, isReconnecting, send, disconnect, reconnect: connect };
}

/**
 * VideoFeed Component
 *
 * Renders the annotated base64 frame received from the backend WebSocket.
 * The backend already draws bounding boxes — we just display the image.
 * getUserMedia is NOT used here; webcam capture is handled server-side.
 */

import React, { useRef, useEffect } from 'react';
import { useStore } from '../../store';

interface VideoFeedProps {
  className?: string;
}

export const VideoFeed: React.FC<VideoFeedProps> = ({ className = '' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const currentFrame   = useStore((state) => state.video.currentFrame);
  const isProcessing   = useStore((state) => state.video.isProcessing);
  const isConnected    = useStore((state) => state.ws.connected);
  const isReconnecting = useStore((state) => state.ws.reconnecting);
  const wsError        = useStore((state) => state.ws.error);

  // Paint the annotated frame onto the canvas whenever it changes.
  // This is the ONLY rendering source — no getUserMedia stream touches this canvas.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentFrame) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    console.log('WS frame received:', currentFrame.substring(0, 50));

    const img = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
    };

    // Guard against the backend accidentally including the data: prefix already
    img.src = currentFrame.startsWith('data:')
      ? currentFrame
      : `data:image/jpeg;base64,${currentFrame}`;
  }, [currentFrame]);

  // ── Placeholder states ────────────────────────────────────────────────────

  if (!currentFrame) {
    if (!isConnected && !isReconnecting) {
      return (
        <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
          <div className="bg-slate-900 w-full h-64 rounded flex items-center justify-center">
            <div className="text-center space-y-2">
              <div className="text-5xl">⚠️</div>
              <p className="text-red-400 font-semibold">Backend not connected</p>
              <p className="text-slate-500 text-sm">
                {wsError ?? 'Make sure the FastAPI server is running on port 8000'}
              </p>
            </div>
          </div>
        </div>
      );
    }

    if (isReconnecting) {
      return (
        <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
          <div className="bg-slate-700 animate-pulse w-full h-64 rounded flex items-center justify-center">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-blue-400">Reconnecting to backend…</span>
            </div>
          </div>
        </div>
      );
    }

    // Connected, no session yet
    return (
      <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
        <div className="bg-slate-900 w-full h-64 rounded flex items-center justify-center">
          <div className="text-center space-y-2">
            <div className="text-5xl">📷</div>
            <p className="text-slate-400">No active session</p>
            <p className="text-slate-500 text-sm">Start a session to see the live feed</p>
          </div>
        </div>
      </div>
    );
  }

  // ── Live canvas ───────────────────────────────────────────────────────────

  return (
    <div className={`bg-slate-800 rounded-lg p-4 ${className}`}>
      <div className="relative">
        {/* canvasRef is the sole visual output — base64 annotated frame only */}
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded border border-slate-700"
          style={{ maxHeight: '480px' }}
        />

        {/* LIVE / PAUSED badge */}
        <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-0.5 rounded text-xs text-white">
          {isProcessing ? (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse inline-block" />
              LIVE
            </span>
          ) : 'PAUSED'}
        </div>

        {/* Connection dot */}
        <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/70 px-2 py-0.5 rounded text-xs text-white">
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
        </div>
      </div>
    </div>
  );
};

/**
 * Dashboard Page
 * Main dashboard with video feed, controls, risk meter, and alerts
 */

import { useEffect, useState } from 'react';
import { useWebSocket, useVideoControl } from '../hooks';
import { useStore } from '../store';
import { VideoFeed, VideoUploader, RiskMeter } from '../components/feed';
import { AlertPanel } from '../components/alert';

function Dashboard() {
  const [selectedSource, setSelectedSource] = useState<'webcam' | 'upload'>('webcam');
  const [selectedVideoFile, setSelectedVideoFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { isConnected, isReconnecting } = useWebSocket();
  const { startSession, stopSession, getStatus } = useVideoControl();

  const isProcessing = useStore((state) => state.video.isProcessing);
  const source = useStore((state) => state.video.source);
  const sessionStats = useStore((state) => state.video.sessionStats);
  const setVideoState = useStore((state) => state.setVideoState);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await getStatus();
        if (status.active) {
          setVideoState({
            isProcessing: true,
            source: {
              type: status.source_type as 'webcam' | 'upload',
              name: status.source_name || status.source_type || 'unknown',
            },
          });
        }
      } catch {
        // Backend unreachable — leave state as-is; WS indicator will show disconnected
      }
    };
    checkStatus();
  }, [getStatus, setVideoState]);

  const handleStartSession = async () => {
    setError(null);
    if (selectedSource === 'upload' && !selectedVideoFile) {
      setError('Please select a video file first');
      return;
    }
    try {
      const sourceName = selectedSource === 'upload' ? selectedVideoFile! : '0';
      await startSession(selectedSource, sourceName);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start session');
    }
  };

  const [isStopping, setIsStopping] = useState(false);

  const handleStopSession = async () => {
    if (isStopping) return; // prevent double-fire
    setIsStopping(true);
    setError(null);
    try {
      await stopSession();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop session');
    } finally {
      setIsStopping(false);
    }
  };

  const handleVideoUpload = (filename: string) => {
    setSelectedVideoFile(filename);
    setSelectedSource('upload');
  };

  const handleFileSelected = (filename: string) => {
    setSelectedVideoFile(filename);
    setSelectedSource('upload');
  };

  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">CrowdGuard Dashboard</h1>
          <p className="text-gray-400">Real-time crowd anomaly detection and monitoring</p>
        </div>

        {/* Backend not connected banner */}
        {!isConnected && !isReconnecting && (
          <div className="mb-6 flex items-center space-x-3 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
            <div className="w-2 h-2 bg-red-500 rounded-full" />
            <span className="text-red-400 text-sm font-medium">
              Backend not connected — make sure the FastAPI server is running on http://localhost:8000
            </span>
          </div>
        )}

        {/* Session Controls */}
        <div className="bg-slate-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-400">Source:</label>
                <select
                  value={selectedSource}
                  onChange={(e) => setSelectedSource(e.target.value as 'webcam' | 'upload')}
                  disabled={isProcessing}
                  className="bg-slate-700 text-white px-3 py-1 rounded border border-slate-600 focus:border-blue-500 focus:outline-none"
                >
                  <option value="webcam">Webcam</option>
                  <option value="upload">
                    Video File {selectedVideoFile ? `(${selectedVideoFile})` : '(Select below)'}
                  </option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                {!isProcessing ? (
                  <button
                    onClick={handleStartSession}
                    disabled={selectedSource === 'upload' && !selectedVideoFile}
                    className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded font-medium transition-colors"
                  >
                    {selectedSource === 'upload' && !selectedVideoFile
                      ? 'Select Video First'
                      : 'Start Session'}
                  </button>
                ) : (
                  <button
                    onClick={handleStopSession}
                    disabled={isStopping}
                    className="bg-red-600 hover:bg-red-700 disabled:bg-red-800 disabled:cursor-not-allowed text-white px-4 py-2 rounded font-medium transition-colors"
                  >
                    {isStopping ? 'Stopping...' : 'Stop Session'}
                  </button>
                )}
              </div>
            </div>

            {/* Session Status */}
            <div className="flex items-center space-x-4 text-sm">
              {source && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-green-400 font-medium">
                    {source.type === 'webcam' ? 'WEBCAM LIVE' : `VIDEO: ${source.name}`}
                  </span>
                </div>
              )}
              <div className="flex items-center space-x-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    isConnected ? 'bg-blue-500' : isReconnecting ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
                  }`}
                />
                <span
                  className={
                    isConnected ? 'text-blue-400' : isReconnecting ? 'text-yellow-400' : 'text-red-400'
                  }
                >
                  {isReconnecting ? 'Reconnecting...' : isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-3 p-3 bg-red-500/20 border border-red-500 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <VideoFeed className="w-full" />
            {!isProcessing && (
              <VideoUploader
                onUploadComplete={handleVideoUpload}
                onFileSelected={handleFileSelected}
                className="w-full"
              />
            )}
          </div>
          <div className="space-y-6">
            <RiskMeter />
            <AlertPanel />
          </div>
        </div>

        {/* Session Statistics */}
        {isProcessing && sessionStats && (
          <div className="mt-6 bg-slate-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Session Statistics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">{sessionStats.totalPersons}</div>
                <div className="text-sm text-gray-400">Total Persons</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-400">{sessionStats.totalAlerts}</div>
                <div className="text-sm text-gray-400">Total Alerts</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400">{sessionStats.peakRiskScore}</div>
                <div className="text-sm text-gray-400">Peak Risk</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {Math.floor(sessionStats.uptime / 60)}:
                  {(sessionStats.uptime % 60).toString().padStart(2, '0')}
                </div>
                <div className="text-sm text-gray-400">Uptime</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;

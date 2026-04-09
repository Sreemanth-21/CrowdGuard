# Zustand Store Documentation

This directory contains the Zustand state management store for the CrowdGuard frontend application.

## Store Structure

The store is organized into five slices, each managing a specific domain of the application state:

### 1. Video Slice (`videoSlice.ts`)
Manages the video processing state including:
- Current video source (webcam or uploaded file)
- Session status and statistics
- Current frame data (person count, risk score, density, anomalies)
- Heatmap toggle state

**Key Actions:**
- `setVideoState(updates)` - Update video state
- `resetVideoState()` - Reset to initial state

### 2. Alert Slice (`alertSlice.ts`)
Manages alert state including:
- Active alerts (10 most recent for dashboard panel)
- Alert history (for alerts page with pagination)
- Alert filters and pagination
- Selected alerts for bulk operations

**Key Actions:**
- `addAlert(alert)` - Add new alert to active alerts
- `setAlertHistory(alerts)` - Set alert history
- `setAlertFilters(filters)` - Update filters
- `dismissAlert(alertId)` - Dismiss single alert
- `dismissAlerts(alertIds)` - Bulk dismiss alerts

### 3. Analytics Slice (`analyticsSlice.ts`)
Manages analytics data including:
- Time series data (density, risk score, person count)
- Alert frequency by type
- Person count histogram
- KPI metrics

**Key Actions:**
- `setDensityTimeseries(data)` - Update density time series
- `setRiskTimeseries(data)` - Update risk time series
- `setAlertFrequency(data)` - Update alert frequency
- `setKPIs(kpis)` - Update KPI metrics

### 4. Settings Slice (`settingsSlice.ts`)
Manages all configuration parameters including:
- Detection thresholds
- Model variant selection
- Anomaly detection parameters
- Heatmap opacity

**Key Actions:**
- `updateSettings(updates)` - Update settings
- `resetSettings()` - Reset to defaults

### 5. WebSocket Slice (`wsSlice.ts`)
Manages WebSocket connection state including:
- Connection status
- Reconnection state with exponential backoff
- Last received message
- Error messages

**Key Actions:**
- `setWsConnected(connected)` - Set connection status
- `setWsReconnecting(reconnecting)` - Set reconnecting state
- `incrementReconnectAttempts()` - Increment reconnect attempts
- `resetReconnectAttempts()` - Reset reconnect attempts

## Usage Example

```typescript
import { useStore } from './store';

function MyComponent() {
  // Select specific state
  const personCount = useStore((state) => state.video.personCount);
  const riskLevel = useStore((state) => state.video.riskLevel);
  
  // Select actions
  const setVideoState = useStore((state) => state.setVideoState);
  const addAlert = useStore((state) => state.addAlert);
  
  // Use in component
  const handleUpdate = () => {
    setVideoState({ personCount: 42, riskLevel: 'WARNING' });
  };
  
  return (
    <div>
      <p>Person Count: {personCount}</p>
      <p>Risk Level: {riskLevel}</p>
      <button onClick={handleUpdate}>Update</button>
    </div>
  );
}
```

## Type Safety

All slices are fully typed with TypeScript interfaces. Import types as needed:

```typescript
import type { Alert, VideoState, SettingsState } from './store/alertSlice';
```

## Requirements Mapping

This store implementation satisfies the following requirements:
- **Requirement 13.1**: Real-time dashboard video feed state management
- **Requirement 14.1**: Real-time alert panel state management
- **Requirement 19.1**: Analytics time series data management
- **Requirement 24.1**: Configuration management state

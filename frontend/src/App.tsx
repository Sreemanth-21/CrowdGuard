import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar, TopBar } from './components/layout';
import Dashboard from './pages/Dashboard';
import Alerts from './pages/Alerts';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings.jsx';
import { useStore } from './store';
import './App.css';

function App() {
  // Pull live session stats from the Zustand store so Sidebar always reflects
  // real WebSocket data — no more "--" placeholders when backend is connected.
  const sessionStats = useStore((state) => state.video.sessionStats);
  const source       = useStore((state) => state.video.source);
  const isConnected  = useStore((state) => state.ws.connected);

  const sidebarStats = {
    uptimeSeconds:  sessionStats.uptime,
    totalPersons:   sessionStats.totalPersons,
    totalAlerts:    sessionStats.totalAlerts,
    peakRiskScore:  sessionStats.peakRiskScore,
  };

  return (
    <BrowserRouter>
      <Sidebar sessionStats={sidebarStats} />
      <TopBar videoSource={source ?? undefined} isConnected={isConnected} />
      <Routes>
        <Route path="/"         element={<Dashboard />} />
        <Route path="/alerts"   element={<Alerts />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

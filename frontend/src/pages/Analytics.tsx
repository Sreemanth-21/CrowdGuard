import { useState } from 'react';
import { PageWrapper } from '../components/layout';
import { 
  KPICards, 
  DensityChart, 
  RiskChart, 
  AlertFrequencyChart, 
  PersonCountHistogram 
} from '../components/analytics';
import { useStore } from '../store';

function Analytics() {
  const [timeRange, setTimeRange] = useState(60); // minutes
  const isSessionActive = useStore((state) => state.video.isProcessing);

  return (
    <PageWrapper>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-heading font-bold mb-2">Analytics Dashboard</h1>
            <p className="font-body text-navy-300">
              Real-time insights and historical trends for crowd monitoring
            </p>
          </div>
          
          {/* Time Range Selector */}
          <div className="flex items-center space-x-3">
            <label className="text-sm text-navy-300">Time Range:</label>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(Number(e.target.value))}
              className="bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
            >
              <option value={15}>Last 15 minutes</option>
              <option value={30}>Last 30 minutes</option>
              <option value={60}>Last 1 hour</option>
              <option value={180}>Last 3 hours</option>
              <option value={360}>Last 6 hours</option>
              <option value={720}>Last 12 hours</option>
              <option value={1440}>Last 24 hours</option>
            </select>
          </div>
        </div>

        {/* Session Status Indicator */}
        {isSessionActive && (
          <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-400 font-medium">Live Session Active</span>
              <span className="text-green-300 text-sm">- Charts auto-refresh every 30 seconds</span>
            </div>
          </div>
        )}

        {/* KPI Cards */}
        <KPICards 
          autoRefresh={isSessionActive}
          refreshInterval={10}
          timeRange={timeRange}
        />

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Density Chart */}
          <DensityChart
            timeRange={timeRange}
            autoRefresh={isSessionActive}
            refreshInterval={30}
          />

          {/* Risk Chart */}
          <RiskChart
            timeRange={timeRange}
            autoRefresh={isSessionActive}
            refreshInterval={30}
          />

          {/* Alert Frequency Chart */}
          <AlertFrequencyChart
            timeRange={timeRange}
            autoRefresh={isSessionActive}
            refreshInterval={30}
          />

          {/* Person Count Histogram */}
          <PersonCountHistogram
            timeRange={timeRange}
            autoRefresh={isSessionActive}
            refreshInterval={30}
          />
        </div>

        {/* Additional Info */}
        <div className="bg-slate-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Analytics Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-300">
            <div>
              <h4 className="font-medium text-white mb-2">Data Sources</h4>
              <ul className="space-y-1">
                <li>• Real-time video processing pipeline</li>
                <li>• ML-based anomaly detection engine</li>
                <li>• Historical session and alert data</li>
                <li>• Crowd density and movement metrics</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-white mb-2">Refresh Rates</h4>
              <ul className="space-y-1">
                <li>• KPI Cards: Every 10 seconds (live sessions)</li>
                <li>• Time Series Charts: Every 30 seconds (live sessions)</li>
                <li>• Historical Data: Manual refresh</li>
                <li>• Alert Frequency: Real-time updates</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}

export default Analytics;

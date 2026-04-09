import { Link, useLocation } from 'react-router-dom';
import { formatDuration } from '../../utils/formatters';

interface SidebarProps {
  sessionStats?: {
    uptimeSeconds: number;
    totalPersons: number;
    totalAlerts: number;
    peakRiskScore: number;
  };
}

function Sidebar({ sessionStats }: SidebarProps) {
  const location = useLocation();

  const navLinks = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/alerts', label: 'Alerts', icon: '🚨' },
    { path: '/analytics', label: 'Analytics', icon: '📈' },
    { path: '/settings', label: 'Settings', icon: '⚙️' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <aside 
      className="fixed left-0 top-0 h-screen w-60 bg-navy-800 border-r border-navy-700 flex flex-col"
      style={{ 
        position: 'fixed', 
        left: '0', 
        top: '0', 
        height: '100vh', 
        width: '240px', 
        backgroundColor: '#000e2a',
        borderRight: '1px solid #00153f',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 10
      }}
    >
      {/* Logo/Brand */}
      <div className="h-14 flex items-center px-6 border-b border-navy-700">
        <h1 className="text-xl font-heading font-bold text-white">CrowdGuard</h1>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-6">
        <ul className="space-y-1 px-3">
          {navLinks.map((link) => (
            <li key={link.path}>
              <Link
                to={link.path}
                className={`
                  flex items-center gap-3 px-3 py-2.5 rounded-lg
                  transition-colors duration-200
                  ${
                    isActive(link.path)
                      ? 'bg-navy-700 text-white'
                      : 'text-navy-300 hover:bg-navy-750 hover:text-white'
                  }
                `}
              >
                <span className="text-lg">{link.icon}</span>
                <span className="font-body font-medium">{link.label}</span>
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* Session Statistics */}
      <div className="border-t border-navy-700 p-4">
        <h2 className="text-xs font-heading font-semibold text-navy-400 uppercase tracking-wider mb-3">
          Session Stats
        </h2>
        <div className="space-y-2.5">
          <StatItem
            label="Uptime"
            value={sessionStats ? formatDuration(sessionStats.uptimeSeconds) : '--'}
          />
          <StatItem
            label="Persons Detected"
            value={sessionStats ? sessionStats.totalPersons.toLocaleString() : '--'}
          />
          <StatItem
            label="Alerts Generated"
            value={sessionStats ? sessionStats.totalAlerts.toLocaleString() : '--'}
          />
          <StatItem
            label="Peak Risk Score"
            value={sessionStats ? sessionStats.peakRiskScore.toFixed(1) : '--'}
          />
        </div>
      </div>
    </aside>
  );
}

interface StatItemProps {
  label: string;
  value: string;
}

function StatItem({ label, value }: StatItemProps) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs font-body text-navy-400">{label}</span>
      <span className="text-sm font-body font-semibold text-white">{value}</span>
    </div>
  );
}

export default Sidebar;

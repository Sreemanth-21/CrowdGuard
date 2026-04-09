# Layout Components

This directory contains the core layout components for the CrowdGuard application.

## Components

### Sidebar.tsx

Fixed-width sidebar (240px) with navigation links and session statistics.

**Requirements:** 17.1, 17.2, 17.3, 17.4, 17.5, 32.1

**Features:**
- Navigation links to all pages (Dashboard, Alerts, Analytics, Settings)
- Active route highlighting
- Real-time session statistics display:
  - Current session uptime (formatted as duration)
  - Total persons detected this session
  - Total alerts generated this session
  - Peak risk score recorded this session

**Props:**
```typescript
interface SidebarProps {
  sessionStats?: {
    uptimeSeconds: number;
    totalPersons: number;
    totalAlerts: number;
    peakRiskScore: number;
  };
}
```

**Usage:**
```tsx
import { Sidebar } from '@/components/layout';

<Sidebar sessionStats={sessionStats} />
```

### TopBar.tsx

Fixed-height topbar (56px) with video source indicator and connection status.

**Requirements:** 16.1, 16.2, 16.3, 16.4, 32.2

**Features:**
- Displays active video source name
- Shows "WEBCAM LIVE" with pulsing green indicator when webcam is active
- Shows filename with pulsing green indicator when uploaded video is active
- Shows "NO SOURCE" when no source is active
- Pulsing indicator dot animation for active connections

**Props:**
```typescript
interface TopBarProps {
  videoSource?: {
    type: 'webcam' | 'upload' | null;
    name: string;
  };
  isConnected?: boolean;
}
```

**Usage:**
```tsx
import { TopBar } from '@/components/layout';

<TopBar 
  videoSource={{ type: 'webcam', name: 'Webcam 0' }} 
  isConnected={true} 
/>
```

### PageWrapper.tsx

Container component for page content with proper spacing and layout.

**Requirements:** 32.1, 32.2

**Features:**
- Applies proper margins for sidebar (240px left) and topbar (56px top)
- Provides consistent padding for page content
- Ensures full viewport height coverage
- Dark theme background

**Props:**
```typescript
interface PageWrapperProps {
  children: ReactNode;
}
```

**Usage:**
```tsx
import { PageWrapper } from '@/components/layout';

function MyPage() {
  return (
    <PageWrapper>
      <h1>Page Title</h1>
      <p>Page content...</p>
    </PageWrapper>
  );
}
```

## Layout Structure

The layout follows a fixed sidebar + topbar pattern:

```
┌─────────────┬──────────────────────────────────┐
│             │         TopBar (56px)            │
│             ├──────────────────────────────────┤
│   Sidebar   │                                  │
│   (240px)   │                                  │
│             │        Page Content              │
│             │      (PageWrapper)               │
│             │                                  │
└─────────────┴──────────────────────────────────┘
```

## Dimensions

- **Sidebar Width:** 240px (fixed)
- **TopBar Height:** 56px (fixed)
- **Content Area:** Calculated as `calc(100vw - 240px)` width and `calc(100vh - 56px)` height

## Styling

All components use:
- TailwindCSS utility classes
- Dark theme with navy color palette (navy-900, navy-800, navy-700)
- Consistent spacing and typography from design system
- Smooth transitions (200ms duration)

## Integration

These components are integrated in `App.tsx` and wrap all page routes:

```tsx
function App() {
  return (
    <BrowserRouter>
      <Sidebar sessionStats={sessionStats} />
      <TopBar videoSource={videoSource} isConnected={isConnected} />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        {/* ... other routes */}
      </Routes>
    </BrowserRouter>
  );
}
```

All page components should use `PageWrapper` to ensure proper layout:

```tsx
function Dashboard() {
  return (
    <PageWrapper>
      {/* Page content */}
    </PageWrapper>
  );
}
```

## Future Enhancements

- Connect to WebSocket for real-time session stats updates
- Connect to state management (Zustand) for video source state
- Add responsive behavior for mobile/tablet portrait
- Add sidebar collapse/expand functionality
- Add notification badges for unread alerts

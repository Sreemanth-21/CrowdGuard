# CrowdGuard Utilities

This directory contains utility functions and constants used throughout the CrowdGuard frontend application.

## Files

### `constants.ts`

Centralized constants for the application including:

- **Risk Levels**: SAFE, CAUTION, WARNING, CRITICAL (Requirements 13.4, 14.4, 32.5)
- **Risk Level Colors**: Color mappings for each risk level with Tailwind classes and hex values
- **Anomaly Types**: All six anomaly types (HIGH_DENSITY, RAPID_MOVEMENT, etc.)
- **API Endpoints**: All backend API endpoints and WebSocket connections
- **File Upload Constraints**: Maximum file size and supported formats

#### Key Functions

- `getRiskColor(level)`: Get color mapping for a risk level
- `getRiskLevelFromScore(score)`: Determine risk level from a score (0-100)
- `getAnomalyLabel(type)`: Get human-readable label for anomaly type

#### Usage Example

```typescript
import { RISK_LEVELS, getRiskColor, getRiskLevelFromScore } from '@/utils/constants';

const level = getRiskLevelFromScore(78); // Returns 'CRITICAL'
const colors = getRiskColor(level); // Returns color object
console.log(colors.hex); // '#ef4444'
```

### `formatters.ts`

Data formatting utilities for display purposes:

#### Time Formatters (Requirement 14.6)

- `formatRelativeTime(timestamp)`: Format as "32 seconds ago", "2 minutes ago", etc.
- `formatAbsoluteTime(timestamp)`: Format as "Jan 15, 2024 10:30:22 AM"
- `formatTimeOnly(timestamp)`: Format as "10:30:22 AM"
- `formatDuration(seconds)`: Format duration as "2m 30s", "1h 15m", etc.

#### Number Formatters

- `formatNumber(value)`: Add thousands separators (1234 → "1,234")
- `formatDecimal(value, decimals)`: Fixed decimal places
- `formatPercentage(value, decimals)`: Format as percentage
- `formatFileSize(bytes)`: Human-readable file size

#### Domain-Specific Formatters

- `formatConfidence(confidence)`: Format confidence score as percentage
- `formatRiskScore(score)`: Format risk score with one decimal place
- `formatDensity(density)`: Format density as percentage
- `formatVelocity(velocity)`: Format velocity in pixels per frame
- `formatCoordinates(x, y)`: Format coordinate pair

#### Utility Functions

- `truncateText(text, maxLength)`: Truncate with ellipsis

#### Usage Example

```typescript
import { formatRelativeTime, formatRiskScore, formatDensity } from '@/utils/formatters';

const alertTime = formatRelativeTime('2024-01-15T10:35:22Z'); // "32 seconds ago"
const riskScore = formatRiskScore(45.678); // "45.7"
const density = formatDensity(0.42); // "42%"
```

### `index.ts`

Central export point for all utilities. Import from here for convenience:

```typescript
import { RISK_LEVELS, formatRelativeTime, getRiskColor } from '@/utils';
```

## Requirements Mapping

- **Requirement 13.4**: Risk level color coding (SAFE: Green, CAUTION: Amber, WARNING: Orange, CRITICAL: Red)
- **Requirement 14.4**: Risk level display with color coding
- **Requirement 14.6**: Relative timestamp formatting ("32 seconds ago", "2 minutes ago")
- **Requirement 32.5**: Risk level color coding consistency

## Testing

Run the demo file to verify all utilities:

```bash
npx tsx frontend/src/utils/demo.ts
```

## TypeScript Support

All utilities are fully typed with TypeScript. Import types as needed:

```typescript
import type { RiskLevel, AnomalyType, VideoSourceType, WSMessageType } from '@/utils';
```

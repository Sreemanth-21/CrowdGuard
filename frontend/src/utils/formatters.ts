/**
 * Data Formatting Utilities
 * 
 * Utilities for formatting timestamps, numbers, and other data
 * for display in the CrowdGuard application.
 */

/**
 * Format timestamp as relative human-readable string (Requirement 14.6)
 * Examples: "just now", "32 seconds ago", "2 minutes ago", "3 hours ago"
 * 
 * @param timestamp - ISO 8601 timestamp string or Date object
 * @returns Human-readable relative time string
 */
export function formatRelativeTime(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSeconds < 10) {
    return 'just now';
  } else if (diffSeconds < 60) {
    return `${diffSeconds} second${diffSeconds !== 1 ? 's' : ''} ago`;
  } else if (diffMinutes < 60) {
    return `${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago`;
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  } else if (diffDays < 7) {
    return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  } else {
    // For older dates, show absolute date
    return date.toLocaleDateString();
  }
}

/**
 * Format timestamp as absolute time string
 * Example: "Jan 15, 2024 10:30:22 AM"
 * 
 * @param timestamp - ISO 8601 timestamp string or Date object
 * @returns Formatted date and time string
 */
export function formatAbsoluteTime(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

/**
 * Format timestamp as time only
 * Example: "10:30:22 AM"
 * 
 * @param timestamp - ISO 8601 timestamp string or Date object
 * @returns Formatted time string
 */
export function formatTimeOnly(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  return date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

/**
 * Format duration in seconds as human-readable string
 * Examples: "45s", "2m 30s", "1h 15m", "2d 3h"
 * 
 * @param seconds - Duration in seconds
 * @returns Formatted duration string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    const remainingSeconds = Math.floor(seconds % 60);
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remainingMinutes = minutes % 60;
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
}

/**
 * Format number with thousands separators
 * Examples: 1234 -> "1,234", 1234567 -> "1,234,567"
 * 
 * @param value - Number to format
 * @returns Formatted number string
 */
export function formatNumber(value: number): string {
  return value.toLocaleString('en-US');
}

/**
 * Format decimal number with fixed precision
 * Examples: 0.12345 -> "0.12", 45.678 -> "45.68"
 * 
 * @param value - Number to format
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted number string
 */
export function formatDecimal(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

/**
 * Format percentage value
 * Examples: 0.75 -> "75%", 0.123 -> "12.3%"
 * 
 * @param value - Decimal value (0-1)
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted percentage string
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format file size in bytes to human-readable string
 * Examples: 1024 -> "1 KB", 1048576 -> "1 MB"
 * 
 * @param bytes - File size in bytes
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Format confidence score as percentage
 * Examples: 0.85 -> "85%", 0.123 -> "12%"
 * 
 * @param confidence - Confidence value (0-1)
 * @returns Formatted confidence string
 */
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

/**
 * Format risk score with one decimal place
 * Examples: 45.678 -> "45.7", 78 -> "78.0"
 * 
 * @param score - Risk score (0-100)
 * @returns Formatted risk score string
 */
export function formatRiskScore(score: number): string {
  return score.toFixed(1);
}

/**
 * Format density value as percentage
 * Examples: 0.42 -> "42%", 0.7 -> "70%"
 * 
 * @param density - Density value (0-1)
 * @returns Formatted density string
 */
export function formatDensity(density: number): string {
  return `${Math.round(density * 100)}%`;
}

/**
 * Format velocity in pixels per frame
 * Examples: 25.5 -> "25.5 px/f", 3 -> "3.0 px/f"
 * 
 * @param velocity - Velocity in pixels per frame
 * @returns Formatted velocity string
 */
export function formatVelocity(velocity: number): string {
  return `${velocity.toFixed(1)} px/f`;
}

/**
 * Truncate text to specified length with ellipsis
 * Examples: "Long text here" -> "Long te..."
 * 
 * @param text - Text to truncate
 * @param maxLength - Maximum length before truncation
 * @returns Truncated text with ellipsis if needed
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength - 3)}...`;
}

/**
 * Format coordinate pair
 * Examples: (512, 384) -> "(512, 384)"
 * 
 * @param x - X coordinate
 * @param y - Y coordinate
 * @returns Formatted coordinate string
 */
export function formatCoordinates(x: number, y: number): string {
  return `(${Math.round(x)}, ${Math.round(y)})`;
}

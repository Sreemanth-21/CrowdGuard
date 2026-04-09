/**
 * API Client for CrowdGuard Backend
 * Provides fetch wrapper with error handling, retry logic, and methods for all endpoints
 */

// Use empty string to leverage Vite proxy in development
// In production, set VITE_API_BASE_URL environment variable
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

interface FetchOptions extends RequestInit {
  retries?: number;
  retryDelay?: number;
}

interface ApiError extends Error {
  status?: number;
  data?: any;
}

/**
 * Fetch wrapper with error handling and retry logic
 */
async function apiFetch<T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<T> {
  const { retries = 3, retryDelay = 1000, ...fetchOptions } = options;
  
  const url = `${API_BASE_URL}${endpoint}`;
  
  const headers = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };
  
  let lastError: ApiError | null = null;
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        ...fetchOptions,
        headers,
      });
      
      if (!response.ok) {
        const error: ApiError = new Error(`HTTP ${response.status}: ${response.statusText}`);
        error.status = response.status;
        
        try {
          error.data = await response.json();
        } catch {
          // Response body is not JSON
        }
        
        if (response.status >= 500 && attempt < retries) {
          lastError = error;
          await new Promise(resolve => setTimeout(resolve, retryDelay * (attempt + 1)));
          continue;
        }
        
        throw error;
      }
      
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return response as any;
    } catch (error) {
      if (attempt === retries) {
        throw lastError || error;
      }
      lastError = error as ApiError;
      await new Promise(resolve => setTimeout(resolve, retryDelay * (attempt + 1)));
    }
  }
  
  throw lastError || new Error('Request failed');
}

// ============================================================================
// Video Management API
// ============================================================================

export const videoApi = {
  async upload(file: File): Promise<{ filename: string; size: number; duration?: number }> {
    const formData = new FormData();
    formData.append('file', file);
    
    // Don't set Content-Type header - let browser set it with boundary
    const url = `${API_BASE_URL}/api/video/upload`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        // No headers - browser will set Content-Type: multipart/form-data with boundary
      });
      
      if (!response.ok) {
        const error: ApiError = new Error(`HTTP ${response.status}: ${response.statusText}`);
        error.status = response.status;
        
        try {
          error.data = await response.json();
          // Extract error message from detail if available
          if (error.data?.detail) {
            if (typeof error.data.detail === 'string') {
              error.message = error.data.detail;
            } else if (error.data.detail.error) {
              error.message = error.data.detail.error;
            }
          }
        } catch {
          // Response body is not JSON
        }
        
        throw error;
      }
      
      return await response.json();
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  },
  
  async start(sourceType: 'webcam' | 'upload', sourceName?: string): Promise<{
    session_id: string;
    source_type: string;
    source_name: string;
    started_at: string;
  }> {
    return apiFetch('/api/video/start', {
      method: 'POST',
      body: JSON.stringify({ 
        source_type: sourceType, 
        source_name: sourceName || '0'
      }),
    });
  },
  
  async stop(): Promise<{
    session_id: string;
    ended_at: string;
    total_frames: number;
    total_alerts: number;
  }> {
    return apiFetch('/api/video/stop', {
      method: 'POST',
    });
  },
  
  async getStatus(): Promise<{
    active: boolean;
    session_id?: string;
    source_type?: string;
    source_name?: string;
    started_at?: string;
    uptime?: number;
  }> {
    return apiFetch('/api/video/status');
  },
  
  async getSources(): Promise<{
    webcams: Array<{ id: number; name: string; available: boolean }>;
    uploaded_files: Array<{ filename: string; size: number }>;
  }> {
    return apiFetch('/api/video/sources');
  },
};

// ============================================================================
// Alerts API
// ============================================================================

export interface Alert {
  alert_id: string;
  session_id: string;
  timestamp: string;
  anomaly_type: string;
  risk_level: string;
  confidence_score: number;
  person_count?: number;
  affected_persons?: number;
  density?: number;
  description: string;
  frame_snapshot_path?: string;
  snapshot_path?: string;
  is_dismissed: boolean;
  dismissed?: boolean;
}

export interface AlertFilters {
  anomaly_type?: string[];
  risk_level?: string[];
  start_date?: string;
  end_date?: string;
  dismissed?: boolean;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
}

export const alertsApi = {
  async list(
    filters: AlertFilters = {},
    pagination: PaginationParams = {}
  ): Promise<{
    alerts: Alert[];
    total: number;
    page: number;
    limit: number;
    pages: number;
  }> {
    const params = new URLSearchParams();
    
    if (pagination.page) params.append('page', pagination.page.toString());
    if (pagination.limit) params.append('limit', pagination.limit.toString());
    if (filters.dismissed !== undefined) params.append('dismissed', filters.dismissed.toString());
    if (filters.start_date) params.append('start_date', filters.start_date);
    if (filters.end_date) params.append('end_date', filters.end_date);
    if (filters.anomaly_type) {
      filters.anomaly_type.forEach(type => params.append('anomaly_type', type));
    }
    if (filters.risk_level) {
      filters.risk_level.forEach(level => params.append('risk_level', level));
    }
    
    return apiFetch(`/api/alerts?${params.toString()}`);
  },
  
  async get(alertId: string): Promise<Alert> {
    return apiFetch(`/api/alerts/${alertId}`);
  },
  
  async dismiss(alertId: string): Promise<{ success: boolean }> {
    return apiFetch(`/api/alerts/${alertId}/dismiss`, {
      method: 'PUT',
    });
  },
  
  async bulkDismiss(alertIds: string[]): Promise<{ dismissed_count: number }> {
    return apiFetch('/api/alerts/bulk-dismiss', {
      method: 'POST',
      body: JSON.stringify({ alert_ids: alertIds }),
    });
  },
  
  async getSummary(): Promise<{
    total_alerts: number;
    by_type: Record<string, number>;
    by_risk_level: Record<string, number>;
  }> {
    return apiFetch('/api/alerts/summary');
  },
  
  async export(filters: AlertFilters = {}): Promise<Blob> {
    const params = new URLSearchParams();
    
    if (filters.start_date) params.append('start_date', filters.start_date);
    if (filters.end_date) params.append('end_date', filters.end_date);
    if (filters.anomaly_type) {
      filters.anomaly_type.forEach(type => params.append('anomaly_type', type));
    }
    if (filters.risk_level) {
      filters.risk_level.forEach(level => params.append('risk_level', level));
    }
    
    const response = await fetch(`${API_BASE_URL}/api/alerts/export?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }
    return response.blob();
  },
  
  getSnapshotUrl(alertId: string): string {
    return `${API_BASE_URL}/api/alerts/${alertId}/snapshot`;
  },
};

// ============================================================================
// Analytics API
// ============================================================================

export interface DensityDataPoint {
  timestamp: string;
  density: number;
  person_count: number;
}

export interface RiskDataPoint {
  timestamp: string;
  risk_score: number;
  risk_level: string;
}

export interface AlertFrequency {
  anomaly_type: string;
  count: number;
}

export interface SessionStats {
  session_id: string;
  started_at: string;
  ended_at?: string;
  duration: number;
  total_frames: number;
  total_alerts: number;
  avg_density: number;
  peak_risk_score: number;
}

export interface KPIs {
  average_density: number;
  total_alerts: number;
  peak_risk_score: number;
  session_duration_minutes: number;
  session_id?: string;
}

export const analyticsApi = {
  async getDensityTimeseries(minutes: number = 60): Promise<DensityDataPoint[]> {
    const result = await apiFetch<any>(`/api/analytics/density-timeseries?minutes=${minutes}`);
    return Array.isArray(result) ? result : result?.data || [];
  },
  
  async getRiskTimeseries(minutes: number = 60): Promise<RiskDataPoint[]> {
    const result = await apiFetch<any>(`/api/analytics/risk-timeseries?minutes=${minutes}`);
    return Array.isArray(result) ? result : result?.data || [];
  },
  
  async getAlertFrequency(): Promise<AlertFrequency[]> {
    const result = await apiFetch<any>('/api/analytics/alert-frequency');
    return Array.isArray(result) ? result : result?.data || [];
  },
  
  async getSessionStats(sessionId?: string): Promise<SessionStats> {
    const params = sessionId ? `?session_id=${sessionId}` : '';
    return apiFetch(`/api/analytics/session-stats${params}`);
  },
  
  async getKPIs(minutes = 60): Promise<KPIs> {
    return apiFetch(`/api/analytics/kpis?minutes=${minutes}`);
  },
};

// ============================================================================
// Settings API
// ============================================================================

export interface Settings {
  confidence_threshold: number;
  model_variant: string;
  high_density_threshold: number;
  rapid_movement_threshold: number;
  sudden_dispersal_threshold: number;
  crowd_surge_threshold: number;
  stationary_duration: number;
  fighting_iou_threshold: number;
  alert_cooldown: number;
  heatmap_opacity: number;
}

export const settingsApi = {
  async get(): Promise<Settings> {
    return apiFetch('/api/settings');
  },
  
  async update(settings: Partial<Settings>): Promise<Settings> {
    return apiFetch('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    });
  },
  
  async reset(): Promise<{ reset: boolean; settings: Settings }> {
    return apiFetch('/api/settings/reset', {
      method: 'POST',
    });
  },
};

export default {
  video: videoApi,
  alerts: alertsApi,
  analytics: analyticsApi,
  settings: settingsApi,
};
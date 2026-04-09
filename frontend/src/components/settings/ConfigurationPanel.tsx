/**
 * ConfigurationPanel Component
 * Provides configuration controls for system settings
 */

import React, { useState, useEffect } from 'react';
import { settingsApi, Settings } from '../../utils/api';

interface ConfigurationPanelProps {
  className?: string;
}

interface ValidationErrors {
  [key: string]: string;
}

export const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({
  className = '',
}) => {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});

  // Load settings on component mount
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await settingsApi.get();
      setSettings(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const validateSettings = (settingsToValidate: Settings): ValidationErrors => {
    const errors: ValidationErrors = {};

    // Requirement 24.1: Confidence threshold validation (0.3-0.9)
    if (settingsToValidate.confidence_threshold < 0.3 || settingsToValidate.confidence_threshold > 0.9) {
      errors.confidence_threshold = 'Confidence threshold must be between 0.3 and 0.9';
    }

    // Model variant validation (nano, small, medium)
    const allowedVariants = ['nano', 'small', 'medium'];
    if (!allowedVariants.includes(settingsToValidate.model_variant)) {
      errors.model_variant = 'Model variant must be nano, small, or medium';
    }

    // Requirement 24.3: High density threshold validation (0.5-0.9)
    if (settingsToValidate.high_density_threshold < 0.5 || settingsToValidate.high_density_threshold > 0.9) {
      errors.high_density_threshold = 'High density threshold must be between 0.5 and 0.9';
    }

    // Rapid movement threshold validation (must be positive)
    if (settingsToValidate.rapid_movement_threshold < 0) {
      errors.rapid_movement_threshold = 'Rapid movement threshold must be positive';
    }

    // Sudden dispersal threshold validation (0.0-1.0)
    if (settingsToValidate.sudden_dispersal_threshold < 0.0 || settingsToValidate.sudden_dispersal_threshold > 1.0) {
      errors.sudden_dispersal_threshold = 'Sudden dispersal threshold must be between 0.0 and 1.0';
    }

    // Crowd surge threshold validation (0.0-1.0)
    if (settingsToValidate.crowd_surge_threshold < 0.0 || settingsToValidate.crowd_surge_threshold > 1.0) {
      errors.crowd_surge_threshold = 'Crowd surge threshold must be between 0.0 and 1.0';
    }

    // Stationary duration validation (must be positive)
    if (settingsToValidate.stationary_duration < 0) {
      errors.stationary_duration = 'Stationary duration must be positive';
    }

    // Fighting IoU threshold validation (0.0-1.0)
    if (settingsToValidate.fighting_iou_threshold < 0.0 || settingsToValidate.fighting_iou_threshold > 1.0) {
      errors.fighting_iou_threshold = 'Fighting IoU threshold must be between 0.0 and 1.0';
    }

    // Requirement 24.4: Alert cooldown validation (5-60 seconds)
    if (settingsToValidate.alert_cooldown < 5 || settingsToValidate.alert_cooldown > 60) {
      errors.alert_cooldown = 'Alert cooldown must be between 5 and 60 seconds';
    }

    // Requirement 24.5: Heatmap opacity validation (0.0-1.0)
    if (settingsToValidate.heatmap_opacity < 0.0 || settingsToValidate.heatmap_opacity > 1.0) {
      errors.heatmap_opacity = 'Heatmap opacity must be between 0.0 and 1.0';
    }

    return errors;
  };

  const handleInputChange = (field: keyof Settings, value: number | string) => {
    if (!settings) return;

    const updatedSettings = { ...settings, [field]: value };
    setSettings(updatedSettings);

    // Clear success message when user makes changes
    setSuccessMessage(null);

    // Validate the specific field
    const errors = validateSettings(updatedSettings);
    setValidationErrors(errors);
  };

  const handleSave = async () => {
    if (!settings) return;

    const errors = validateSettings(settings);
    setValidationErrors(errors);

    if (Object.keys(errors).length > 0) {
      return;
    }

    try {
      setSaving(true);
      setError(null);
      await settingsApi.update(settings);
      setSuccessMessage('Settings saved successfully! Changes will be applied to active sessions within 1 second.');
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    try {
      setSaving(true);
      setError(null);
      const resetResponse = await settingsApi.reset();
      setSettings(resetResponse.settings); // Extract settings from response
      setValidationErrors({});
      setSuccessMessage('Settings reset to defaults successfully!');
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset settings');
    } finally {
      setSaving(false);
    }
  };

  const isValid = Object.keys(validationErrors).length === 0;

  if (loading) {
    return (
      <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-slate-700 rounded w-1/3 mb-6"></div>
          <div className="space-y-4">
            {[...Array(6)].map((_, i) => (
              <div key={i}>
                <div className="h-4 bg-slate-700 rounded w-1/4 mb-2"></div>
                <div className="h-10 bg-slate-700 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error && !settings) {
    return (
      <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
        <h3 className="text-lg font-semibold text-white mb-4">Configuration</h3>
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-400 font-medium">Failed to load settings</p>
          <p className="text-red-300 text-sm mt-1">{error}</p>
          <button
            onClick={loadSettings}
            className="mt-3 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!settings) return null;

  return (
    <div className={`bg-slate-800 rounded-lg p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-white mb-6">Configuration</h3>

      {/* Success Message */}
      {successMessage && (
        <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4 mb-6">
          <p className="text-green-400 font-medium">{successMessage}</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 mb-6">
          <p className="text-red-400 font-medium">{error}</p>
        </div>
      )}

      <div className="space-y-6">
        {/* Detection Settings */}
        <div>
          <h4 className="text-md font-medium text-white mb-4">Detection Settings</h4>
          
          {/* Confidence Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Confidence Threshold: {settings.confidence_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.3"
              max="0.9"
              step="0.01"
              value={settings.confidence_threshold}
              onChange={(e) => handleInputChange('confidence_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0.3</span>
              <span>0.9</span>
            </div>
            {validationErrors.confidence_threshold && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.confidence_threshold}</p>
            )}
          </div>

          {/* Model Variant */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Model Variant
            </label>
            <select
              value={settings.model_variant}
              onChange={(e) => handleInputChange('model_variant', e.target.value)}
              className="w-full bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
            >
              <option value="nano">YOLOv8 Nano (fastest)</option>
              <option value="small">YOLOv8 Small (balanced)</option>
              <option value="medium">YOLOv8 Medium (accurate)</option>
            </select>
          </div>
        </div>

        {/* Anomaly Detection Thresholds */}
        <div>
          <h4 className="text-md font-medium text-white mb-4">Anomaly Detection Thresholds</h4>
          
          {/* High Density Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              High Density Threshold: {settings.high_density_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.01"
              value={settings.high_density_threshold}
              onChange={(e) => handleInputChange('high_density_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0.5</span>
              <span>0.9</span>
            </div>
            {validationErrors.high_density_threshold && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.high_density_threshold}</p>
            )}
          </div>

          {/* Rapid Movement Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Rapid Movement Threshold: {settings.rapid_movement_threshold} px/frame
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={settings.rapid_movement_threshold}
              onChange={(e) => handleInputChange('rapid_movement_threshold', parseInt(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0</span>
              <span>100</span>
            </div>
            {validationErrors.rapid_movement_threshold && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.rapid_movement_threshold}</p>
            )}
          </div>

          {/* Sudden Dispersal Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Sudden Dispersal Threshold: {(settings.sudden_dispersal_threshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={settings.sudden_dispersal_threshold}
              onChange={(e) => handleInputChange('sudden_dispersal_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0%</span>
              <span>100%</span>
            </div>
            {validationErrors.sudden_dispersal_threshold && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.sudden_dispersal_threshold}</p>
            )}
          </div>

          {/* Crowd Surge Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Crowd Surge Threshold: {(settings.crowd_surge_threshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={settings.crowd_surge_threshold}
              onChange={(e) => handleInputChange('crowd_surge_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0%</span>
              <span>100%</span>
            </div>
            {validationErrors.crowd_surge_threshold && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.crowd_surge_threshold}</p>
            )}
          </div>

          {/* Stationary Duration */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Stationary Duration: {settings.stationary_duration} seconds
            </label>
            <input
              type="range"
              min="0"
              max="120"
              step="1"
              value={settings.stationary_duration}
              onChange={(e) => handleInputChange('stationary_duration', parseInt(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0s</span>
              <span>120s</span>
            </div>
            {validationErrors.stationary_duration && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.stationary_duration}</p>
            )}
          </div>

          {/* Fighting IoU Threshold */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Fighting IoU Threshold: {settings.fighting_iou_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={settings.fighting_iou_threshold}
              onChange={(e) => handleInputChange('fighting_iou_threshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0.0</span>
              <span>1.0</span>
            </div>
            {validationErrors.fighting_iou_threshold && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.fighting_iou_threshold}</p>
            )}
          </div>
        </div>

        {/* Alert Settings */}
        <div>
          <h4 className="text-md font-medium text-white mb-4">Alert Settings</h4>
          
          {/* Alert Cooldown */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Alert Cooldown: {settings.alert_cooldown} seconds
            </label>
            <input
              type="range"
              min="5"
              max="60"
              step="1"
              value={settings.alert_cooldown}
              onChange={(e) => handleInputChange('alert_cooldown', parseInt(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>5s</span>
              <span>60s</span>
            </div>
            {validationErrors.alert_cooldown && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.alert_cooldown}</p>
            )}
          </div>
        </div>

        {/* Display Settings */}
        <div>
          <h4 className="text-md font-medium text-white mb-4">Display Settings</h4>
          
          {/* Heatmap Opacity */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Heatmap Opacity: {(settings.heatmap_opacity * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={settings.heatmap_opacity}
              onChange={(e) => handleInputChange('heatmap_opacity', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0%</span>
              <span>100%</span>
            </div>
            {validationErrors.heatmap_opacity && (
              <p className="text-red-400 text-sm mt-1">{validationErrors.heatmap_opacity}</p>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4 pt-4 border-t border-slate-700">
          <button
            onClick={handleSave}
            disabled={!isValid || saving}
            className={`
              px-6 py-2 rounded-lg font-medium transition-colors
              ${isValid && !saving
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }
            `}
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
          
          <button
            onClick={handleReset}
            disabled={saving}
            className="px-6 py-2 bg-slate-600 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  );
};
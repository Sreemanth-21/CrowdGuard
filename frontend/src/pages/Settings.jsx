/**
 * Settings Page Component
 * Main settings page that serves as container for ConfigurationPanel and FederatedLearningDemo
 * Handles loading states and integrates with settings API
 * 
 * Requirements: 24.1, 29.7
 */

import React, { useState, useEffect } from 'react';
import { PageWrapper } from '../components/layout';
import { ConfigurationPanel } from '../components/settings/ConfigurationPanel';
import FederatedLearningDemo from '../components/federated/FederatedLearningDemo';
import { settingsApi } from '../utils/api';

function Settings() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize settings loading
    const initializeSettings = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Pre-load settings to ensure they're available
        await settingsApi.get();
        
        // Small delay to show loading state briefly
        await new Promise(resolve => setTimeout(resolve, 300));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load settings');
      } finally {
        setLoading(false);
      }
    };

    initializeSettings();
  }, []);

  if (loading) {
    return (
      <PageWrapper>
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-heading font-bold mb-6">Settings</h1>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Configuration Panel Loading Skeleton */}
            <div className="lg:col-span-1">
              <div className="bg-slate-800 rounded-lg p-6">
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
                  <div className="flex space-x-4 pt-4 border-t border-slate-700 mt-6">
                    <div className="h-10 bg-slate-700 rounded w-24"></div>
                    <div className="h-10 bg-slate-700 rounded w-32"></div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Federated Learning Demo Loading Skeleton */}
            <div className="lg:col-span-1">
              <div className="bg-slate-800 rounded-lg p-6">
                <div className="animate-pulse">
                  <div className="h-6 bg-slate-700 rounded w-1/2 mb-4"></div>
                  <div className="h-4 bg-slate-700 rounded w-3/4 mb-4"></div>
                  <div className="flex space-x-4 mb-4">
                    <div className="h-10 bg-slate-700 rounded w-32"></div>
                    <div className="h-10 bg-slate-700 rounded w-24"></div>
                  </div>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    {[...Array(3)].map((_, i) => (
                      <div key={i} className="h-24 bg-slate-700 rounded"></div>
                    ))}
                  </div>
                  <div className="h-32 bg-slate-700 rounded"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper>
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-heading font-bold mb-6">Settings</h1>
          
          <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-red-400 mb-2">
              Failed to Load Settings
            </h3>
            <p className="text-red-300 mb-4">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </PageWrapper>
    );
  }

  return (
    <PageWrapper>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-heading font-bold mb-6">Settings</h1>
        
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="xl:col-span-1">
            <ConfigurationPanel />
          </div>
          
          {/* Federated Learning Demo */}
          <div className="xl:col-span-2">
            <div className="bg-slate-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">
                Federated Learning Demo
              </h3>
              <p className="text-gray-300 text-sm mb-6">
                Experience privacy-preserving machine learning with our federated learning simulation. 
                Watch as multiple virtual nodes collaborate to train a global model without sharing raw data.
              </p>
              
              <FederatedLearningDemo />
            </div>
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}

export default Settings;
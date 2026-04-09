/**
 * ErrorBoundary Component
 * Catches component render errors and displays fallback UI
 */

import React, { ReactNode, ReactElement } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactElement {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="flex items-center justify-center min-h-screen bg-slate-900">
            <div className="bg-slate-800 rounded-lg p-8 max-w-md border border-red-500">
              <h2 className="text-xl font-bold text-red-400 mb-2">Something went wrong</h2>
              <p className="text-gray-300 mb-4 text-sm">
                {this.state.error?.message || 'An unexpected error occurred'}
              </p>
              <details className="mb-4 text-xs text-gray-400">
                <summary className="cursor-pointer hover:text-gray-300">Error details</summary>
                <pre className="mt-2 bg-slate-900 p-2 rounded overflow-auto max-h-40">
                  {this.state.error?.stack}
                </pre>
              </details>
              <button
                onClick={this.handleRetry}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors"
              >
                Try again
              </button>
            </div>
          </div>
        )
      );
    }

    return this.props.children as ReactElement;
  }
}

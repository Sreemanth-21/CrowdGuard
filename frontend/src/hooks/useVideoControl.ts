/**
 * Video Control Hook for CrowdGuard
 * Manages video session control and file uploads
 */

import { useCallback } from 'react';
import { videoApi } from '../utils/api';
import { useStore } from '../store';

export function useVideoControl() {
  const setVideoState = useStore((state) => state.setVideoState);
  const setWsError = useStore((state) => state.setWsError);

  const startSession = useCallback(
    async (sourceType: 'webcam' | 'upload', sourceName?: string) => {
      try {
        setVideoState({ isProcessing: true });
        const response = await videoApi.start(sourceType, sourceName);
        setVideoState({ 
          isProcessing: true,
          source: { type: sourceType, name: sourceName || sourceType }
        });
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to start session';
        setWsError(message);
        throw error;
      }
    },
    [setVideoState, setWsError]
  );

  const stopSession = useCallback(async () => {
    try {
      const response = await videoApi.stop();
      setVideoState({ 
        isProcessing: false,
        source: null,
        currentFrame: null
      });
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to stop session';
      setWsError(message);
      throw error;
    }
  }, [setVideoState, setWsError]);

  const uploadVideo = useCallback(
    async (file: File) => {
      try {
        const response = await videoApi.upload(file);
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to upload video';
        setWsError(message);
        throw error;
      }
    },
    [setWsError]
  );

  const getStatus = useCallback(async () => {
    try {
      return await videoApi.getStatus();
    } catch (error) {
      console.error('Failed to get session status:', error);
      throw error;
    }
  }, []);

  const getSources = useCallback(async () => {
    try {
      return await videoApi.getSources();
    } catch (error) {
      console.error('Failed to get video sources:', error);
      throw error;
    }
  }, []);

  return {
    startSession,
    stopSession,
    uploadVideo,
    getStatus,
    getSources,
  };
}

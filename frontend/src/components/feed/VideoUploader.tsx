import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useVideoControl } from '../../hooks';
import { FILE_UPLOAD } from '../../utils/constants';

interface UploadedFile {
  filename: string;
  size: number;
}

interface VideoUploaderProps {
  onUploadComplete?: (filename: string) => void;
  onFileSelected?: (filename: string) => void;
  className?: string;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({
  onUploadComplete,
  onFileSelected,
  className = '',
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { uploadVideo, getSources } = useVideoControl();

  const fetchUploadedFiles = useCallback(async () => {
    try {
      setIsLoadingFiles(true);
      const data = await getSources();
      setUploadedFiles(data.uploaded_files || []);
    } catch (err) {
      console.error('Failed to fetch uploaded files:', err);
      setUploadedFiles([]);
    } finally {
      setIsLoadingFiles(false);
    }
  }, [getSources]);

  useEffect(() => {
    fetchUploadedFiles();
  }, [fetchUploadedFiles]);

  const validateFile = (file: File): string | null => {
    if (file.size > FILE_UPLOAD.MAX_SIZE_BYTES) {
      return `File size exceeds ${FILE_UPLOAD.MAX_SIZE_MB}MB limit`;
    }
    const extension = file.name.split('.').pop()?.toLowerCase();
    const supportedFormats = FILE_UPLOAD.SUPPORTED_FORMATS as readonly string[];
    if (!extension || !supportedFormats.includes(extension)) {
      return `Unsupported format. Supported: ${FILE_UPLOAD.SUPPORTED_FORMATS.join(', ')}`;
    }
    const supportedMimeTypes = FILE_UPLOAD.SUPPORTED_MIME_TYPES as readonly string[];
    if (!supportedMimeTypes.includes(file.type)) {
      return `Invalid file type: ${file.type}`;
    }
    return null;
  };

  const handleFileUpload = useCallback(async (file: File) => {
    setError(null);
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    try {
      setIsUploading(true);
      setUploadProgress(0);
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);
      const response = await uploadVideo(file);
      clearInterval(progressInterval);
      setUploadProgress(100);
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
        onUploadComplete?.(response.filename);
        fetchUploadedFiles();
        setSelectedFile(response.filename);
      }, 500);
    } catch (error) {
      setIsUploading(false);
      setUploadProgress(0);
      setError(error instanceof Error ? error.message : 'Upload failed');
    }
  }, [uploadVideo, onUploadComplete, fetchUploadedFiles]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  const handleSelectUploadedFile = (filename: string) => {
    setSelectedFile(filename);
    onFileSelected?.(filename);
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="bg-slate-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Upload Video</h3>
        <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${isDragOver ? 'border-blue-500 bg-blue-500 bg-opacity-10' : 'border-slate-600 hover:border-slate-500'} ${isUploading ? 'pointer-events-none opacity-50' : ''}`} onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onClick={openFilePicker}>
          <input ref={fileInputRef} type="file" accept={FILE_UPLOAD.SUPPORTED_MIME_TYPES.join(',')} onChange={handleFileSelect} className="hidden" />
          {isUploading ? (
            <div className="space-y-4">
              <div className="text-blue-400">
                <svg className="w-12 h-12 mx-auto animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              <div>
                <p className="text-white font-medium">Uploading...</p>
                <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
                  <div className="bg-blue-500 h-2 rounded-full transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div>
                </div>
                <p className="text-sm text-gray-400 mt-1">{uploadProgress}%</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-gray-400">
                <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <p className="text-white font-medium">Drop video file here</p>
                <p className="text-gray-400 text-sm">or click to browse</p>
              </div>
              <div className="text-xs text-gray-500">
                <p>Supported formats: {FILE_UPLOAD.SUPPORTED_FORMATS.join(', ').toUpperCase()}</p>
                <p>Max size: {FILE_UPLOAD.MAX_SIZE_MB}MB</p>
              </div>
            </div>
          )}
        </div>
        {error && (
          <div className="mt-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 rounded-lg">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
      </div>
      <div className="bg-slate-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Uploaded Videos</h3>
          <button onClick={fetchUploadedFiles} disabled={isLoadingFiles} className="flex items-center space-x-1 text-sm text-blue-400 hover:text-blue-300 disabled:opacity-50 transition-colors">
            <svg className={`w-4 h-4 ${isLoadingFiles ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>{isLoadingFiles ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
        {isLoadingFiles ? (
          <div className="flex justify-center py-8">
            <div className="text-gray-400 flex items-center space-x-2">
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Loading files...</span>
            </div>
          </div>
        ) : uploadedFiles.length > 0 ? (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {uploadedFiles.map((file) => (
              <div key={file.filename} onClick={() => handleSelectUploadedFile(file.filename)} className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${selectedFile === file.filename ? 'border-blue-500 bg-blue-500 bg-opacity-10' : 'border-slate-700 bg-slate-700 hover:border-slate-600'}`}>
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium truncate">{file.filename}</p>
                    <p className="text-sm text-gray-400">{formatFileSize(file.size)}</p>
                  </div>
                  {selectedFile === file.filename && (
                    <div className="text-blue-400 ml-3 flex-shrink-0">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-gray-500 mb-2">
              <svg className="w-12 h-12 mx-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
              </svg>
            </div>
            <p className="text-gray-400">No uploaded videos yet</p>
            <p className="text-sm text-gray-500 mt-1">Upload a video to get started</p>
          </div>
        )}
      </div>
    </div>
  );
};

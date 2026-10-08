/**
 * Veritus — Analysis State Hook (src/hooks/useAnalysis.js)
 *
 * Manages the state and polling lifecycle for autonomous hypothesis discovery runs.
 */

import { useState, useRef, useCallback } from 'react';
import { analyzeDataset, getAnalysisStatus } from '../api/client';

export function useAnalysis() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);

  const pollIntervalRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const runAnalysis = useCallback(
    async (file, numHypotheses = 6, enableRefinement = true) => {
      if (!file) {
        setError('Please select or upload a CSV file before running analysis.');
        return;
      }

      const jobId = 'job_' + Math.random().toString(36).substring(2, 11) + Date.now();
      setActiveJobId(jobId);
      setIsAnalyzing(true);
      setProgress(0.05);
      setProgressMessage('Uploading dataset and launching AI discovery loop...');
      setError(null);
      setResults(null);

      // Start polling status
      pollIntervalRef.current = setInterval(async () => {
        try {
          const statusData = await getAnalysisStatus(jobId);
          if (statusData) {
            if (typeof statusData.progress === 'number') {
              setProgress(statusData.progress);
            }
            if (statusData.message) {
              setProgressMessage(statusData.message);
            }
            if (statusData.status === 'completed' && statusData.result) {
              stopPolling();
              setResults(statusData.result);
              setIsAnalyzing(false);
              setProgress(1.0);
            } else if (statusData.status === 'failed') {
              stopPolling();
              setIsAnalyzing(false);
              setError(statusData.error || 'Pipeline execution failed.');
            }
          }
        } catch {
          // Polling probe fails temporarily while job starts or network hiccups; keep going
        }
      }, 1000);

      try {
        // Trigger pipeline synchronously from API perspective (returns result upon completion)
        const data = await analyzeDataset(file, numHypotheses, enableRefinement, jobId, false);
        stopPolling();
        setResults(data);
        setProgress(1.0);
        setProgressMessage('Analysis completed successfully!');
        setIsAnalyzing(false);
      } catch (err) {
        stopPolling();
        setIsAnalyzing(false);
        setError(err.message || 'An error occurred during dataset analysis.');
      }
    },
    [stopPolling]
  );

  const loadHistoricalResults = useCallback((historicalData) => {
    stopPolling();
    setIsAnalyzing(false);
    setError(null);
    setResults(historicalData);
    setProgress(1.0);
    setProgressMessage('Loaded historical dataset run.');
  }, [stopPolling]);

  const resetAnalysis = useCallback(() => {
    stopPolling();
    setIsAnalyzing(false);
    setProgress(0);
    setProgressMessage('');
    setResults(null);
    setError(null);
    setActiveJobId(null);
  }, [stopPolling]);

  return {
    isAnalyzing,
    progress,
    progressMessage,
    results,
    error,
    activeJobId,
    runAnalysis,
    loadHistoricalResults,
    resetAnalysis,
  };
}

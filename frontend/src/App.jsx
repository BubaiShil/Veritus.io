import React, { useState, useEffect, useCallback } from 'react';
import Papa from 'papaparse';
import Sidebar from './components/Sidebar';
import DatasetPreview from './components/DatasetPreview';
import RunAnalysisButton from './components/RunAnalysisButton';
import ProgressTracker from './components/ProgressTracker';
import ResultsTabs from './components/ResultsTabs';
import { useAnalysis } from './hooks/useAnalysis';
import { getSystemStatus, getHistory, getHistoryDetail } from './api/client';
import { AlertTriangle, Sparkles, Brain, CheckCircle, Database } from 'lucide-react';

export default function App() {
  // System and History State
  const [systemStatus, setSystemStatus] = useState({ ollama: false, database: false });
  const [statusLoading, setStatusLoading] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState(null);

  // Dataset Input State
  const [file, setFile] = useState(null);
  const [datasetPreviewData, setDatasetPreviewData] = useState(null);
  const [numHypotheses, setNumHypotheses] = useState(6);
  const [enableRefinement, setEnableRefinement] = useState(true);

  // Analysis Hook
  const {
    isAnalyzing,
    progress,
    progressMessage,
    results,
    error,
    runAnalysis,
    loadHistoricalResults,
    resetAnalysis,
  } = useAnalysis();

  // Load Initial Status and Run History
  const fetchStatus = useCallback(async () => {
    setStatusLoading(true);
    try {
      const data = await getSystemStatus();
      setSystemStatus(data);
    } catch {
      setSystemStatus({ ollama: false, database: false });
    } finally {
      setStatusLoading(false);
    }
  }, []);

  const fetchHistoryList = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const data = await getHistory();
      if (data?.datasets) {
        setHistoryList(data.datasets);
      }
    } catch {
      setHistoryList([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchHistoryList();
  }, [fetchStatus, fetchHistoryList]);

  // Handle CSV File Selection & Instant Client-Side Preview
  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setActiveDatasetId(null);
    resetAnalysis();

    Papa.parse(selectedFile, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const rows = results.data;
        const columns = results.meta.fields || [];

        // Compute preview metrics
        let missingCount = 0;
        let numericCount = 0;

        columns.forEach((col) => {
          let hasNumeric = false;
          rows.forEach((row) => {
            const val = row[col];
            if (val === null || val === undefined || val === '') {
              missingCount++;
            } else if (typeof val === 'number') {
              hasNumeric = true;
            }
          });
          if (hasNumeric) numericCount++;
        });

        setDatasetPreviewData({
          rows,
          columns,
          totalRows: rows.length,
          totalColumns: columns.length,
          missingCount,
          numericCount,
        });
      },
      error: () => {
        setDatasetPreviewData(null);
      },
    });
  };

  const handleFileClear = () => {
    setFile(null);
    setDatasetPreviewData(null);
    resetAnalysis();
    setActiveDatasetId(null);
  };

  // Run Analysis Handler
  const handleRunAnalysis = async () => {
    if (!file) return;
    setActiveDatasetId(null);
    await runAnalysis(file, numHypotheses, enableRefinement);
    // Refresh history list upon completion
    fetchHistoryList();
  };

  // Select Historical Run Handler
  const handleSelectHistoricalRun = async (datasetId) => {
    setActiveDatasetId(datasetId);
    try {
      const detail = await getHistoryDetail(datasetId);
      loadHistoricalResults(detail);
      setFile(null);
      setDatasetPreviewData(null);
    } catch {
      // Failed to fetch historical run details
    }
  };

  return (
    <div className="app-layout">
      {/* Left Sidebar */}
      <Sidebar
        systemStatus={systemStatus}
        statusLoading={statusLoading}
        onRefreshStatus={fetchStatus}
        file={file}
        onFileSelect={handleFileSelect}
        onFileClear={handleFileClear}
        numHypotheses={numHypotheses}
        onNumHypothesesChange={setNumHypotheses}
        enableRefinement={enableRefinement}
        onEnableRefinementChange={setEnableRefinement}
        historyList={historyList}
        historyLoading={historyLoading}
        onRefreshHistory={fetchHistoryList}
        onSelectHistoricalRun={handleSelectHistoricalRun}
        activeDatasetId={activeDatasetId}
      />

      {/* Main Content Area */}
      <main className="main-content">
        {/* Header */}
        <header className="header-section">
          <h1 className="vt-title">🔬 Veritus</h1>
          <p className="vt-subtitle">
            Autonomous AI Data Scientist & Hypothesis Discovery Engine
          </p>
        </header>

        {/* Global Error Banner */}
        {error && (
          <div className="alert-box error" role="alert">
            <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Analysis Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Service Warning Banner (if Ollama or Postgres are offline) */}
        {(!systemStatus.ollama || !systemStatus.database) && !error && (
          <div className="alert-box info" style={{ background: '#FFFBEB', borderColor: '#FDE68A', color: '#92400E' }}>
            <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Service Notice:</strong>{' '}
              {!systemStatus.ollama && !systemStatus.database
                ? 'Both Ollama and PostgreSQL appear to be offline. Start Ollama and Docker to run discoveries.'
                : !systemStatus.ollama
                ? 'Ollama is offline. Start your local Ollama server (or configure GROQ_API_KEY) for hypothesis generation.'
                : 'PostgreSQL is unreachable. Start the database container to persist findings and run history.'}
            </div>
          </div>
        )}

        {/* Welcome Hero / AHGT Explainer Grid when no dataset or results active */}
        {!datasetPreviewData && !results && !isAnalyzing && (
          <>
            <div className="vt-hero">
              <h3>Scientific Discovery on Autopilot</h3>
              <p>
                Veritus uploads your dataset, profiles statistical distributions, autonomously formulates
                falsifiable scientific hypotheses using an LLM, validates them against rigorous frequentist tests,
                and self-corrects failed hypotheses via an iterative refinement loop.
              </p>
            </div>

            <div className="vt-grid">
              <div className="vt-feature">
                <span className="n">1</span>
                <div className="t">Distribution Profiling</div>
                <div className="d">
                  Extracts column types, variance, and candidates for statistical analysis.
                </div>
              </div>

              <div className="vt-feature">
                <span className="n">2</span>
                <div className="t">Hypothesis Formulation</div>
                <div className="d">
                  Generates domain-grounded testable propositions mapped to suitable statistical tests.
                </div>
              </div>

              <div className="vt-feature">
                <span className="n">3</span>
                <div className="t">Rigorous Testing</div>
                <div className="d">
                  Runs Welch’s t-test, ANOVA, Mann-Whitney, Pearson/Spearman, and Chi-Square with α=0.05.
                </div>
              </div>

              <div className="vt-feature">
                <span className="n">4</span>
                <div className="t">Self-Correcting Loop</div>
                <div className="d">
                  Failed hypotheses are iteratively reformulated with narrower criteria and re-tested.
                </div>
              </div>
            </div>
          </>
        )}

        {/* Dataset Preview & Action Button */}
        {datasetPreviewData && !results && !isAnalyzing && (
          <>
            <DatasetPreview
              data={datasetPreviewData}
              filename={file ? file.name : 'Uploaded Dataset'}
            />

            <RunAnalysisButton
              onRun={handleRunAnalysis}
              loading={isAnalyzing}
              disabled={!file}
              numHypotheses={numHypotheses}
              refinementEnabled={enableRefinement}
            />
          </>
        )}

        {/* Live Progress Tracker during analysis */}
        {isAnalyzing && (
          <ProgressTracker progress={progress} message={progressMessage} />
        )}

        {/* Results View */}
        {results && (
          <section aria-label="Analysis Results">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#0F172A', margin: 0 }}>
                  Analysis Report:{' '}
                  <span style={{ color: '#4F46E5' }}>
                    {results.dataset_name || (file ? file.name : 'Dataset Run')}
                  </span>
                </h2>
                <div style={{ fontSize: '0.85rem', color: '#64748B', marginTop: '2px' }}>
                  Lineage saved in PostgreSQL with full statistical test metrics and plain-English explanations.
                </div>
              </div>

              {/* Run Another Dataset button */}
              <button
                className="btn-primary"
                style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                onClick={() => {
                  resetAnalysis();
                  setFile(null);
                  setDatasetPreviewData(null);
                  setActiveDatasetId(null);
                }}
              >
                + Analyze New Dataset
              </button>
            </div>

            <ResultsTabs results={results} />
          </section>
        )}
      </main>
    </div>
  );
}

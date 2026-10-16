import React, { useRef } from 'react';
import SystemStatusBadge from './SystemStatusBadge';
import {
  Upload,
  FileSpreadsheet,
  Sliders,
  History,
  CheckCircle2,
  XCircle,
  X,
  RefreshCw,
} from 'lucide-react';

export default function Sidebar({
  systemStatus,
  statusLoading,
  onRefreshStatus,
  file,
  onFileSelect,
  onFileClear,
  numHypotheses,
  onNumHypothesesChange,
  enableRefinement,
  onEnableRefinementChange,
  historyList,
  historyLoading,
  onRefreshHistory,
  onSelectHistoricalRun,
  activeDatasetId,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        onFileSelect(droppedFile);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <aside className="sidebar">
      {/* Sidebar Top / Title */}
      <div className="sidebar-header">
        <span style={{ fontSize: '1.5rem' }}>🔬</span>
        <div>
          <div className="sidebar-title">Veritus</div>
          <div style={{ fontSize: '0.72rem', color: '#64748B' }}>AHGT Discovery Engine</div>
        </div>
      </div>

      {/* System Status Badges */}
      <SystemStatusBadge
        status={systemStatus}
        loading={statusLoading}
        onRefresh={onRefreshStatus}
      />

      {/* Dataset Upload Area */}
      <div>
        <div className="section-label">Dataset Input</div>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".csv"
          style={{ display: 'none' }}
          id="csv-file-input"
        />

        {!file ? (
          <div
            className="upload-dropzone"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <Upload size={22} className="upload-icon" />
            <div className="upload-text">Upload CSV Dataset</div>
            <div className="upload-hint">Drag & drop or click to browse</div>
          </div>
        ) : (
          <div className="file-pill">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden' }}>
              <FileSpreadsheet size={16} color="#18181B" style={{ flexShrink: 0 }} />
              <span
                style={{
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {file.name}
              </span>
            </div>
            <button
              onClick={onFileClear}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: '#64748B',
                padding: '2px',
                display: 'flex',
              }}
              title="Remove file"
            >
              <X size={14} />
            </button>
          </div>
        )}
      </div>

      {/* Agent Settings Controls */}
      <div className="control-group">
        <div className="section-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Sliders size={13} />
          Agent Settings
        </div>

        {/* Hypothesis Count Slider */}
        <div className="slider-container">
          <div className="slider-header">
            <span>Hypotheses to Formulate</span>
            <span className="slider-val">{numHypotheses}</span>
          </div>
          <input
            type="range"
            min="3"
            max="12"
            step="1"
            value={numHypotheses}
            onChange={(e) => onNumHypothesesChange(Number(e.target.value))}
            id="hypotheses-count-slider"
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#94A3B8' }}>
            <span>3</span>
            <span>12</span>
          </div>
        </div>

        {/* Refinement Loop Switch */}
        <div style={{ marginTop: '0.5rem' }}>
          <label className="toggle-row">
            <span>Self-Correction Loop</span>
            <div className="switch">
              <input
                type="checkbox"
                checked={enableRefinement}
                onChange={(e) => onEnableRefinementChange(e.target.checked)}
                id="refinement-toggle"
              />
              <span className="slider-toggle"></span>
            </div>
          </label>
          <div style={{ fontSize: '0.72rem', color: '#64748B', marginTop: '0.2rem' }}>
            Iteratively reformulates failed hypotheses with narrower conditions
          </div>
        </div>
      </div>

      {/* Run History List */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem',
          }}
        >
          <div className="section-label" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <History size={13} />
            Run History
          </div>
          <button
            onClick={onRefreshHistory}
            disabled={historyLoading}
            style={{
              background: 'none',
              border: 'none',
              cursor: historyLoading ? 'not-allowed' : 'pointer',
              color: '#64748B',
              padding: 2,
              display: 'flex',
            }}
            title="Refresh history"
          >
            <RefreshCw size={12} className={historyLoading ? 'animate-spin' : ''} />
          </button>
        </div>

        <div className="history-list">
          {historyList.length === 0 ? (
            <div style={{ fontSize: '0.78rem', color: '#94A3B8', textAlign: 'center', padding: '1rem 0' }}>
              No historical runs logged yet
            </div>
          ) : (
            historyList.map((item) => {
              const isSelected = activeDatasetId === item.id;
              return (
                <div
                  key={item.id}
                  className={`history-item ${isSelected ? 'active' : ''}`}
                  onClick={() => onSelectHistoricalRun(item.id)}
                >
                  <div className="history-name" title={item.name}>
                    {item.name}
                  </div>
                  <div className="history-meta">
                    <span>{formatDate(item.uploaded_at)}</span>
                    <span style={{ display: 'flex', gap: '6px' }}>
                      <span style={{ color: '#166534', display: 'flex', alignItems: 'center', gap: '2px' }}>
                        <CheckCircle2 size={11} /> {item.validated_count || 0}
                      </span>
                      <span style={{ color: '#991B1B', display: 'flex', alignItems: 'center', gap: '2px' }}>
                        <XCircle size={11} /> {item.rejected_count || 0}
                      </span>
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}

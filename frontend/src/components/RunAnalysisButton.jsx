import React from 'react';
import { Play, Loader2 } from 'lucide-react';

export default function RunAnalysisButton({
  onRun,
  loading,
  disabled,
  numHypotheses,
  refinementEnabled,
}) {
  return (
    <div className="action-banner">
      <div>
        <div style={{ fontWeight: 700, fontSize: '0.98rem', color: '#0F172A' }}>
          Execute Autonomous Analysis
        </div>
        <div style={{ fontSize: '0.82rem', color: '#64748B', marginTop: '2px' }}>
          Formulate up to {numHypotheses} hypotheses with frequentist testing &{' '}
          {refinementEnabled ? 'iterative self-correction' : 'standard testing'}
        </div>
      </div>

      <button
        className="btn-primary"
        onClick={onRun}
        disabled={disabled || loading}
        id="btn-run-analysis"
      >
        {loading ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Analyzing Dataset...
          </>
        ) : (
          <>
            <Play size={18} fill="currentColor" />
            🚀 Run Analysis
          </>
        )}
      </button>
    </div>
  );
}

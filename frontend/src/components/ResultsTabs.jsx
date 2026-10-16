import React, { useState } from 'react';
import FindingCard from './FindingCard';
import { CheckCircle2, XCircle, Sparkles, Activity } from 'lucide-react';

export default function ResultsTabs({ results }) {
  const [activeTab, setActiveTab] = useState('validated');

  if (!results) return null;

  const validated = results.validated || [];
  const rejected = results.rejected || [];
  const allHypotheses = results.all_hypotheses || [...validated, ...rejected];
  const refinedCount = allHypotheses.filter((h) => h.is_refined).length;

  return (
    <div style={{ marginTop: '1.5rem' }}>
      {/* Metrics Row */}
      <div className="metrics-row">
        <div className="metric-tile">
          <div className="metric-tile-label">Total Tested</div>
          <div className="metric-tile-val">
            <Activity size={22} color="#18181B" />
            {results.total_hypotheses ?? allHypotheses.length}
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Validated Discoveries</div>
          <div className="metric-tile-val valid">
            <CheckCircle2 size={22} color="#166534" />
            {results.validated_count ?? validated.length}
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Rejected Hypotheses</div>
          <div className="metric-tile-val reject">
            <XCircle size={22} color="#991B1B" />
            {results.rejected_count ?? rejected.length}
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Self-Corrections (Refined)</div>
          <div className="metric-tile-val" style={{ color: '#6B21A8' }}>
            <Sparkles size={22} color="#A855F7" />
            {refinedCount}
          </div>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="tabs-nav">
        <button
          className={`tab-btn valid ${activeTab === 'validated' ? 'active' : ''}`}
          onClick={() => setActiveTab('validated')}
          id="tab-validated"
        >
          <CheckCircle2 size={18} />
          Validated Discoveries
          <span className="tab-badge valid">{validated.length}</span>
        </button>

        <button
          className={`tab-btn reject ${activeTab === 'rejected' ? 'active' : ''}`}
          onClick={() => setActiveTab('rejected')}
          id="tab-rejected"
        >
          <XCircle size={18} />
          Rejected Hypotheses
          <span className="tab-badge reject">{rejected.length}</span>
        </button>
      </div>

      {/* Tab Panels */}
      <div>
        {activeTab === 'validated' && (
          <div>
            {validated.length === 0 ? (
              <div className="empty-state">
                <p>No hypotheses reached statistical significance in this analysis run.</p>
              </div>
            ) : (
              validated.map((hypo, idx) => (
                <FindingCard key={hypo.id || `val-${idx}`} hypothesis={hypo} />
              ))
            )}
          </div>
        )}

        {activeTab === 'rejected' && (
          <div>
            {rejected.length === 0 ? (
              <div className="empty-state">
                <p>No rejected hypotheses recorded for this dataset.</p>
              </div>
            ) : (
              rejected.map((hypo, idx) => (
                <FindingCard key={hypo.id || `rej-${idx}`} hypothesis={hypo} />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}

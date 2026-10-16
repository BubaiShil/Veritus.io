import React, { useState, useMemo } from 'react';
import {
  MessageSquare,
  Sparkles,
  Send,
  Loader2,
  CheckCircle2,
  XCircle,
  HelpCircle,
  BarChart2,
  TrendingUp,
  ChevronRight,
  Info,
  Layers,
  Table,
} from 'lucide-react';
import { askDatasetQuestion } from '../api/client';

export default function AskVeritus({ file, datasetId, columns = [] }) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Generate dynamic smart suggestion chips based on available columns
  const suggestions = useMemo(() => {
    if (!columns || columns.length === 0) {
      return [
        'How many rows and columns are in this data?',
        'Which feature has the strongest link to the outcome?',
        'Are there any missing values or nulls?',
      ];
    }

    const lowerCols = columns.map((c) => ({ orig: c, low: c.toLowerCase() }));
    // Try to find candidate target
    const target =
      lowerCols.find((c) =>
        ['progression', 'target', 'outcome', 'disease', 'churn', 'price', 'sales', 'y'].some(
          (k) => c.low.includes(k)
        )
      )?.orig || columns[columns.length - 1];

    const nonTarget = columns.filter((c) => c !== target);
    const colA = nonTarget[0] || columns[0];
    const colB = nonTarget[1] || columns[1] || columns[0];

    const chips = [];
    chips.push('How many rows and columns are in this dataset?');
    if (target) {
      chips.push(`Which feature has the strongest link to ${target}?`);
    }
    if (colA && colB && colA !== colB) {
      chips.push(`Does ${colA} significantly correlate with ${colB}?`);
    }
    const catCol = lowerCols.find((c) => ['sex', 'gender', 'type', 'group', 'class', 'status'].some(k => c.low.includes(k)))?.orig;
    if (catCol && (target || colA)) {
      chips.push(`Compare ${target || colA} across ${catCol} categories`);
    }
    if (colA) {
      chips.push(`What is the average and distribution of ${colA}?`);
    }

    return chips.slice(0, 4);
  }, [columns]);

  const handleAsk = async (queryToRun) => {
    const q = (queryToRun || question).trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setQuestion(q);

    try {
      const data = await askDatasetQuestion(q, datasetId, file);
      if (data && data.success) {
        setResult(data);
        setHistory((prev) => [
          { question: q, answer: data.answer, verdict: data.verdict, timestamp: Date.now() },
          ...prev.filter((h) => h.question !== q).slice(0, 4),
        ]);
      } else {
        setError(data?.error || 'Unable to analyze question against dataset.');
      }
    } catch (err) {
      setError(err.message || 'Failed to process query with Veritus engine.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="card-panel ask-veritus-panel" style={{ marginTop: '1.5rem' }}>
      {/* Header */}
      <div className="ask-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div className="ask-icon-badge">
            <Sparkles size={20} color="#18181B" />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', color: '#0F172A', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              Ask Veritus
              <span className="badge-pill" style={{ background: '#F4F4F5', color: '#18181B', fontSize: '0.72rem', fontWeight: 600 }}>
                On-Demand Statistical Q&A
              </span>
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748B' }}>
              Ask questions or test custom hypotheses in plain English (powered by Gemini / AI & frequentist stats)
            </div>
          </div>
        </div>
      </div>

      {/* Suggestion Chips */}
      <div className="ask-chips-container">
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748B' }}>Quick questions:</span>
        {suggestions.map((chip, idx) => (
          <button
            key={idx}
            className="ask-chip-btn"
            onClick={() => handleAsk(chip)}
            disabled={loading}
          >
            <ChevronRight size={13} />
            {chip}
          </button>
        ))}
      </div>

      {/* Input Field */}
      <div className="ask-input-wrapper">
        <input
          type="text"
          className="ask-input"
          placeholder='Ask any question about your data... (e.g. "Average BP by gender", "Top predictors of target", "Missing values?")'
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          id="ask-veritus-input"
        />
        <button
          className="ask-submit-btn"
          onClick={() => handleAsk()}
          disabled={loading || !question.trim()}
          id="ask-veritus-btn"
        >
          {loading ? (
            <>
              <Loader2 size={16} className="spin-icon" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Send size={15} />
              <span>Ask</span>
            </>
          )}
        </button>
      </div>

      {/* Error Notice */}
      {error && (
        <div className="alert-box error" style={{ marginTop: '1rem', padding: '0.6rem 0.9rem', fontSize: '0.85rem' }}>
          <Info size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Result Card */}
      {result && (
        <div className="ask-result-card">
          <div className="ask-result-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              {/* Verdict badge */}
              {result.verdict === 'validated' && (
                <span className="verdict-tag valid">
                  <CheckCircle2 size={14} /> Validated (p &lt; 0.05)
                </span>
              )}
              {result.verdict === 'rejected' && (
                <span className="verdict-tag reject">
                  <XCircle size={14} /> Rejected (p &ge; 0.05)
                </span>
              )}
              {result.verdict === 'informative' && (
                <span className="verdict-tag" style={{ background: '#F4F4F5', color: '#18181B', borderColor: '#E4E4E7' }}>
                  <BarChart2 size={14} /> Empirical Finding
                </span>
              )}

              {result.test_method && (
                <span className="method-pill">{result.test_method}</span>
              )}

              {result.p_value !== undefined && result.p_value !== null && (
                <span className="metric-pill">
                  p = {result.p_value < 0.0001 ? '< 0.0001' : result.p_value.toFixed(4)}
                </span>
              )}

              {result.effect_size !== undefined && result.effect_size !== null && (
                <span className="metric-pill">
                  {result.effect_size_name || 'Effect Size'}: {typeof result.effect_size === 'number' ? result.effect_size.toFixed(3) : result.effect_size}
                </span>
              )}
            </div>

            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
              {result.dataset_name || 'Active Dataset'}
            </span>
          </div>

          {/* Interpreted Hypothesis */}
          {result.interpreted_hypothesis && (
            <div className="ask-hypothesis-banner">
              <strong>Interpreted Hypothesis / Action:</strong> &ldquo;{result.interpreted_hypothesis}&rdquo;
            </div>
          )}

          {/* Plain-English Scientific Answer */}
          <div className="ask-answer-body">
            <div className="ask-answer-title">Answer & Empirical Findings:</div>
            <p className="ask-answer-text">{result.answer}</p>
          </div>

          {/* Dataset Overview Grid */}
          {result.action === 'dataset_overview' && result.overview && (
            <div className="ask-overview-grid">
              <div className="ask-overview-card">
                <div className="ask-overview-num">{result.overview.num_rows?.toLocaleString()}</div>
                <div className="ask-overview-label">Total Rows</div>
              </div>
              <div className="ask-overview-card">
                <div className="ask-overview-num">{result.overview.num_cols}</div>
                <div className="ask-overview-label">Total Columns</div>
              </div>
              <div className="ask-overview-card">
                <div className="ask-overview-num">{result.overview.numeric_cols_count}</div>
                <div className="ask-overview-label">Numeric Features</div>
              </div>
              <div className="ask-overview-card">
                <div className="ask-overview-num">{result.overview.categorical_cols_count}</div>
                <div className="ask-overview-label">Categorical Features</div>
              </div>
              <div className="ask-overview-card">
                <div className="ask-overview-num" style={{ color: result.overview.total_missing > 0 ? '#EF4444' : '#166534' }}>
                  {result.overview.total_missing?.toLocaleString()}
                </div>
                <div className="ask-overview-label">Missing Values</div>
              </div>
            </div>
          )}

          {/* Group Comparison Table */}
          {result.action === 'group_comparison' && result.groups && result.groups.length > 0 && (
            <div style={{ marginTop: '0.85rem' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#18181B', marginBottom: '0.4rem' }}>
                Group Breakdown ({result.metric_col} by {result.group_col}):
              </div>
              <table className="ask-group-table">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Observations (n)</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                  </tr>
                </thead>
                <tbody>
                  {result.groups.map((g, idx) => (
                    <tr key={idx}>
                      <td style={{ fontWeight: 600 }}>{g.group}</td>
                      <td>{g.count}</td>
                      <td style={{ fontWeight: 700, color: '#18181B' }}>{g.mean}</td>
                      <td>{g.median}</td>
                      <td>{g.std}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Comparative Ranking Table */}
          {result.action === 'comparative_ranking' && result.ranking && result.ranking.length > 0 && (
            <div className="ask-ranking-section">
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#18181B', marginBottom: '0.5rem' }}>
                Feature Relationship Strength with &lsquo;{result.target_col}&rsquo; (Ranked):
              </div>
              <div className="ask-ranking-table">
                {result.ranking.map((row, idx) => {
                  const pct = Math.min(100, Math.round((row.abs_effect || 0) * 100));
                  return (
                    <div key={idx} className="ask-rank-row">
                      <div className="ask-rank-num">#{idx + 1}</div>
                      <div className="ask-rank-col">{row.column}</div>
                      <div className="ask-rank-bar-wrap">
                        <div
                          className="ask-rank-bar"
                          style={{
                            width: `${pct}%`,
                            background: row.significant
                              ? 'linear-gradient(90deg, #18181B, #52525B)'
                              : '#CBD5E1',
                          }}
                        />
                      </div>
                      <div className="ask-rank-metric">
                        {row.metric_name}: <strong>{row.metric_value}</strong>
                      </div>
                      <div className="ask-rank-p">
                        p {row.p_value < 0.0001 ? '< 0.0001' : `= ${row.p_value.toFixed(4)}`}
                      </div>
                      <span
                        className="badge-pill"
                        style={{
                          fontSize: '0.7rem',
                          background: row.significant ? '#DCFCE7' : '#F4F4F5',
                          color: row.significant ? '#166534' : '#71717A',
                        }}
                      >
                        {row.significant ? 'Significant' : 'Non-significant'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Descriptive Stats Grid */}
          {result.action === 'descriptive_summary' && result.details && (
            <div className="metrics-row" style={{ marginTop: '0.75rem' }}>
              {Object.entries(result.details)
                .filter(([k]) => typeof result.details[k] === 'number')
                .slice(0, 6)
                .map(([k, val]) => (
                  <div key={k} className="metric-tile" style={{ padding: '0.5rem 0.8rem' }}>
                    <div className="metric-tile-label" style={{ textTransform: 'capitalize' }}>
                      {k}
                    </div>
                    <div className="metric-tile-val" style={{ fontSize: '1.1rem' }}>
                      {val}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}

      {/* Query History */}
      {history.length > 1 && (
        <div className="ask-history-section">
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94A3B8', marginBottom: '0.4rem' }}>
            Recent queries:
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {history.slice(1).map((h, i) => (
              <button
                key={i}
                className="ask-history-chip"
                onClick={() => handleAsk(h.question)}
              >
                <MessageSquare size={12} />
                <span>{h.question}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

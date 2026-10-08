import React from 'react';
import { CheckCircle2, XCircle, Sparkles, GitBranch } from 'lucide-react';

export default function FindingCard({ hypothesis }) {
  const {
    id,
    hypothesis_text,
    test_method,
    p_value,
    effect_size,
    effect_size_name,
    verdict,
    explanation,
    summary,
    parent_id,
    is_refined,
  } = hypothesis;

  const isValid = verdict === 'validated';
  const cardType = is_refined ? 'refined' : isValid ? 'valid' : 'reject';

  // Format p-value nicely
  const formatPVal = (p) => {
    if (p === null || p === undefined) return 'N/A';
    if (p < 0.0001) return '< 0.0001';
    if (p < 0.001) return p.toExponential(2);
    return Number(p).toFixed(4);
  };

  // Format effect size nicely
  const formatEffect = (eff) => {
    if (eff === null || eff === undefined) return null;
    return Number(eff).toFixed(3);
  };

  return (
    <div className={`finding-card ${cardType}`} id={`finding-${id}`}>
      {/* Title with Status Icon */}
      <div className="finding-card-title" style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem' }}>
        <span style={{ marginTop: '2px', flexShrink: 0 }}>
          {isValid ? (
            <CheckCircle2 size={19} color="#166534" />
          ) : (
            <XCircle size={19} color="#991B1B" />
          )}
        </span>
        <div style={{ flex: 1 }}>
          <span>{hypothesis_text}</span>
        </div>
      </div>

      {/* Badges Row */}
      <div className="finding-badges">
        {/* Refined Badge */}
        {is_refined && (
          <span className="finding-badge b-refined" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <Sparkles size={11} /> REFINED
          </span>
        )}

        {/* Lineage Badge */}
        {parent_id && (
          <span className="finding-badge b-lineage" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <GitBranch size={11} /> Parent #{parent_id}
          </span>
        )}

        {/* Statistical Test Method */}
        {test_method && (
          <span className="finding-badge b-method">
            {test_method}
          </span>
        )}

        {/* P-Value */}
        <span className={`finding-badge ${isValid ? 'b-ok' : 'b-fail'}`}>
          p = {formatPVal(p_value)} {isValid ? '(p < 0.05)' : '(p ≥ 0.05)'}
        </span>

        {/* Effect Size */}
        {effect_size !== null && effect_size !== undefined && (
          <span className="finding-badge b-effect">
            Effect Size: {formatEffect(effect_size)} {effect_size_name ? `(${effect_size_name})` : ''}
          </span>
        )}
      </div>

      {/* Explanation Box */}
      <div className="finding-explanation">
        <div className="finding-explanation-label">
          {isValid ? 'Scientific Explanation' : 'Statistical Verdict & Summary'}
        </div>
        <p style={{ margin: 0 }}>
          {explanation || summary || (isValid ? 'Hypothesis is statistically supported.' : 'Failed to reach statistical significance.')}
        </p>
      </div>
    </div>
  );
}

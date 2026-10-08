import React from 'react';
import { Loader2, Sparkles } from 'lucide-react';

export default function ProgressTracker({ progress, message }) {
  const pct = Math.min(100, Math.max(5, Math.round((progress || 0) * 100)));

  return (
    <div className="progress-card">
      <div className="progress-header">
        <div className="progress-title">
          <Loader2 size={18} className="animate-spin" />
          <span>AHGT Autonomous Discovery Loop Active</span>
        </div>
        <div className="progress-pct">{pct}%</div>
      </div>

      <div className="progress-bar-bg">
        <div
          className="progress-bar-fill"
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="progress-status-msg">
        <Sparkles size={14} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle', color: '#4F46E5' }} />
        {message || 'AI agent processing dataset schema and statistics...'}
      </div>
    </div>
  );
}

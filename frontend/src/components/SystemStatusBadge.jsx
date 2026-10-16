import React from 'react';
import { Cpu, Database, RefreshCw } from 'lucide-react';

export default function SystemStatusBadge({ status, loading, onRefresh }) {
  const ollamaOnline = Boolean(status?.ollama);
  const dbOnline = Boolean(status?.database);

  return (
    <div className="status-badge-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
        <span className="section-label" style={{ margin: 0 }}>System Services</span>
        <button
          onClick={onRefresh}
          disabled={loading}
          style={{
            background: 'none',
            border: 'none',
            cursor: loading ? 'not-allowed' : 'pointer',
            color: '#64748B',
            padding: 2,
            display: 'flex',
            alignItems: 'center'
          }}
          title="Refresh service status"
          aria-label="Refresh service status"
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="status-badge-row">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#475569' }}>
          <Cpu size={14} /> AI Engine ({status?.llm_provider === 'gemini' ? 'Gemini' : status?.llm_provider === 'groq' ? 'Groq' : 'Ollama'})
        </span>
        <span className={`status-pill ${ollamaOnline ? 'online' : 'offline'}`}>
          <span style={{ fontSize: '0.65rem' }}>{ollamaOnline ? '●' : '○'}</span>
          {ollamaOnline ? 'Online' : 'Offline'}
        </span>
      </div>

      <div className="status-badge-row">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#475569' }}>
          <Database size={14} /> PostgreSQL
        </span>
        <span className={`status-pill ${dbOnline ? 'online' : 'offline'}`}>
          <span style={{ fontSize: '0.65rem' }}>{dbOnline ? '●' : '○'}</span>
          {dbOnline ? 'Online' : 'Offline'}
        </span>
      </div>
    </div>
  );
}

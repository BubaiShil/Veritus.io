import React from 'react';
import { Table, Layers, FileSpreadsheet, AlertCircle } from 'lucide-react';

export default function DatasetPreview({ data, filename }) {
  if (!data || !data.rows || data.rows.length === 0) {
    return null;
  }

  const { rows, columns, totalRows, totalColumns, missingCount, numericCount } = data;
  const previewRows = rows.slice(0, 10);

  return (
    <div className="card-panel">
      <div className="card-panel-header">
        <div className="card-panel-title">
          <FileSpreadsheet size={20} color="#4F46E5" />
          <span>Dataset Preview: <strong>{filename}</strong></span>
        </div>
        <span style={{ fontSize: '0.8rem', color: '#64748B' }}>
          Showing top {Math.min(10, rows.length)} of {totalRows.toLocaleString()} rows
        </span>
      </div>

      {/* Dataset Summary Metrics */}
      <div className="metrics-row" style={{ marginBottom: '1.25rem' }}>
        <div className="metric-tile" style={{ padding: '0.75rem 1rem' }}>
          <div className="metric-tile-label">Total Rows</div>
          <div className="metric-tile-val" style={{ fontSize: '1.4rem' }}>
            {totalRows.toLocaleString()}
          </div>
        </div>

        <div className="metric-tile" style={{ padding: '0.75rem 1rem' }}>
          <div className="metric-tile-label">Columns</div>
          <div className="metric-tile-val" style={{ fontSize: '1.4rem' }}>
            {totalColumns}
          </div>
        </div>

        <div className="metric-tile" style={{ padding: '0.75rem 1rem' }}>
          <div className="metric-tile-label">Numeric Features</div>
          <div className="metric-tile-val" style={{ fontSize: '1.4rem' }}>
            {numericCount}
          </div>
        </div>

        <div className="metric-tile" style={{ padding: '0.75rem 1rem' }}>
          <div className="metric-tile-label">Missing Values</div>
          <div className="metric-tile-val" style={{ fontSize: '1.4rem', color: missingCount > 0 ? '#B45309' : '#166534' }}>
            {missingCount}
          </div>
        </div>
      </div>

      {/* Table Preview */}
      <div className="table-wrapper">
        <table className="preview-table">
          <thead>
            <tr>
              <th style={{ width: '40px', textAlign: 'center' }}>#</th>
              {columns.map((col, idx) => (
                <th key={idx}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewRows.map((row, rIdx) => (
              <tr key={rIdx}>
                <td style={{ textAlign: 'center', color: '#94A3B8' }}>{rIdx + 1}</td>
                {columns.map((col, cIdx) => {
                  const val = row[col];
                  const isNull = val === null || val === undefined || val === '';
                  return (
                    <td key={cIdx} style={{ color: isNull ? '#94A3B8' : undefined }}>
                      {isNull ? '—' : String(val)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

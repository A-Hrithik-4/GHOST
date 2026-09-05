import React from 'react';
import { Layers } from 'lucide-react';

const SIDEBAR_ITEMS = [
  { id: 'overview', num: '01', title: 'OVERVIEW' },
  { id: 'problem_objective', num: '02', title: 'PROBLEM & OBJECTIVE' },
  { id: 'solution', num: '03', title: 'SOLUTION' },
  { id: 'system_architecture', num: '04', title: 'SYSTEM ARCHITECTURE' },
  { id: 'data_feature_engineering', num: '05', title: 'DATA & FEATURE ENGINEERING' },
  { id: 'ml_model', num: '06', title: 'ML MODEL' },
  { id: 'navigation_ekf', num: '07', title: 'NAVIGATION & EKF' },
  { id: 'experiments_results', num: '08', title: 'EXPERIMENTS & RESULTS' },
  { id: 'implementation_code', num: '09', title: 'IMPLEMENTATION & CODE' },
  { id: 'limitations_future_scope', num: '10', title: 'LIMITATIONS & FUTURE SCOPE' },
];

export default function ProjectSidebar({ activeSection, onSelectSection, onSwitchMode }) {
  return (
    <aside className="project-sidebar">
      <div className="sidebar-header-box" style={{ padding: '16px 16px 14px 16px', borderBottom: '1.5px solid var(--border-light)' }}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '15px', fontWeight: '900', color: 'var(--graphite-dark)', letterSpacing: '1.5px' }}>
          GHOST
        </div>
        <div style={{ fontSize: '10px', fontWeight: '800', color: 'var(--amber-primary)', letterSpacing: '0.5px', marginTop: '3px', textTransform: 'uppercase' }}>
          GNSS-Free Hybrid Onboard Sensor Tracker
        </div>
      </div>

      <div className="sidebar-scroll-content" style={{ padding: '12px 0' }}>
        <div className="sidebar-group">
          {SIDEBAR_ITEMS.map((sec) => {
            const isActive = activeSection === sec.id;
            return (
              <div
                key={sec.id}
                className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelectSection(sec.id)}
              >
                <span className="nav-idx">{sec.num}</span>
                <span className="nav-title">{sec.title}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ height: '1px', backgroundColor: 'var(--border-light)', margin: '8px 16px' }} />

      <div style={{ padding: '12px 16px' }}>
        <button 
          className="sidebar-proto-btn"
          onClick={() => onSwitchMode('prototype')}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '10px 14px',
            backgroundColor: 'var(--bg-card)',
            border: '1.5px solid var(--amber-border)',
            borderRadius: '6px',
            color: 'var(--amber-primary)',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px',
            fontWeight: '800',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            boxShadow: '0 2px 4px rgba(200, 138, 0, 0.08)'
          }}
        >
          <Layers size={13} />
          <span>OPEN PROTOTYPE</span>
        </button>
      </div>
    </aside>
  );
}

import React from 'react';
import { Layers, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function ProjectNavbar({ activeMode, onSwitchMode }) {
  return (
    <header className="global-header">
      <div className="header-brand-box">
        <span className="header-logo">GHOST</span>
        <div className="header-divider" />
        <div className="header-sub-info">
          <h1>GNSS-Free Hybrid Sensor Tracker</h1>
          <p>SIH 26168 · ISRO / Department of Space</p>
        </div>
      </div>

      <div className="header-badges">
        <div className="tech-indicator-pill">
          <span className="indicator-dot dot-amber" />
          <span>SYSTEM DOCUMENTATION</span>
        </div>
        <div className="tech-indicator-pill">
          <span className="indicator-dot dot-green" />
          <span>MODEL VALIDATED (2.40% DRIFT)</span>
        </div>
      </div>

      <div className="mode-toggle-group">
        <button 
          className={`mode-btn ${activeMode === 'prototype' ? 'active-proto' : ''}`}
          onClick={() => onSwitchMode('prototype')}
        >
          <Layers size={13} />
          <span>PROTOTYPE</span>
        </button>
        <button 
          className={`mode-btn ${activeMode === 'project' ? 'active-project' : ''}`}
          onClick={() => onSwitchMode('project')}
        >
          <ShieldCheck size={13} />
          <span>PROJECT MODE</span>
        </button>
      </div>
    </header>
  );
}

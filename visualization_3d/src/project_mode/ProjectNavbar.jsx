import React from 'react';

/**
 * ProjectNavbar.jsx
 * 
 * Top header with persistent global mode switcher: [ 🚗 Prototype ] [ 📊 Project ]
 */

export default function ProjectNavbar({ activeMode, onSwitchMode }) {
  return (
    <header className="global-header">
      <div className="header-brand-box">
        <div className="header-logo">GHOST</div>
        <div className="header-divider"></div>
        <div className="header-sub-info">
          <h1>GNSS-FREE HYBRID ONBOARD SENSOR TRACKER</h1>
          <p>Smart India Hackathon 2026 · SIH26168 · ISRO</p>
        </div>
      </div>

      {/* Global Persistent Mode Switcher */}
      <div className="mode-toggle-group">
        <button
          className={`mode-btn ${activeMode === 'prototype' ? 'active-proto' : ''}`}
          onClick={() => onSwitchMode('prototype')}
        >
          Prototype
        </button>

        <button
          className={`mode-btn ${activeMode === 'project' ? 'active-project' : ''}`}
          onClick={() => onSwitchMode('project')}
        >
          Project
        </button>
      </div>
    </header>
  );
}

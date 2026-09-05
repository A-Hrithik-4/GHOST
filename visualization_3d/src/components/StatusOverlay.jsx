import React from 'react';
import { Layers, FileText } from 'lucide-react';

/**
 * StatusOverlay.jsx
 * 
 * Top overlay header displaying GHOST branding, simulation mode, GNSS status badges,
 * and the outage event banner at t = 30s.
 */

export default function StatusOverlay({ sample, currentTime, activeMode, onSwitchMode }) {
  const isOutage = sample ? sample.state === 'OUTAGE' : (currentTime >= 30.0 && currentTime < 90.0);
  const isRestored = sample ? sample.state === 'RESTORED' : currentTime >= 90.0;
  const isJustLost = currentTime >= 30.0 && currentTime < 33.5;

  let gnssStatusClass = "status-gnss";
  let gnssStatusText = "GNSS: AVAILABLE";
  let dotClass = "dot-green";

  if (isOutage) {
    gnssStatusClass = "status-gnss-lost";
    gnssStatusText = "GNSS: LOST";
    dotClass = "dot-red";
  } else if (isRestored) {
    gnssStatusClass = "status-gnss-restored";
    gnssStatusText = "GNSS: RESTORED";
    dotClass = "dot-blue";
  }

  let ghostStatusClass = "status-ghost-standby";
  let ghostStatusText = "GHOST: STANDBY";
  let ghostDotClass = "dot-yellow";

  if (isOutage) {
    ghostStatusClass = "status-ghost-active";
    ghostStatusText = "GHOST: ACTIVE";
    ghostDotClass = "dot-cyan";
  } else if (isRestored) {
    ghostStatusClass = "status-ghost-realigned";
    ghostStatusText = "GHOST: RE-ALIGNED";
    ghostDotClass = "dot-green";
  }

  return (
    <>
      {/* Top Header Bar */}
      <header className="header-overlay">
        <div className="brand-section">
          <div className="brand-logo">GHOST</div>
          <div className="brand-divider"></div>
          <div className="brand-info">
            <h1>GNSS-FREE HYBRID ONBOARD SENSOR TRACKER</h1>
            <p>Smart India Hackathon (SIH PS 26168) — ISRO</p>
          </div>
          <div className="badge-dev">EXPERIMENTAL 60S GNSS OUTAGE SIMULATION</div>
        </div>

        <div className="status-section">
          <div className={`status-badge ${gnssStatusClass}`}>
            <span className={`status-dot ${dotClass}`}></span>
            {gnssStatusText}
          </div>
          <div className={`status-badge ${ghostStatusClass}`}>
            <span className={`status-dot ${ghostDotClass}`}></span>
            {ghostStatusText}
          </div>

          {/* Mode Switcher Toggle Pill */}
          {onSwitchMode && (
            <div className="mode-toggle-group">
              <button
                className={`mode-btn ${activeMode === 'prototype' ? 'active-proto' : ''}`}
                onClick={() => onSwitchMode('prototype')}
              >
                <Layers size={13} />
                <span>Prototype</span>
              </button>
              <button
                className={`mode-btn ${activeMode === 'project' ? 'active-project' : ''}`}
                onClick={() => onSwitchMode('project')}
              >
                <FileText size={13} />
                <span>Project</span>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Outage Event Notification Banner (t = 30.0s) */}
      {isJustLost && (
        <div className="outage-event-banner">
          <div className="banner-icon">ALERT</div>
          <div className="banner-text">
            <h2>GNSS SIGNAL LOST</h2>
            <p>GHOST AI Fallback Navigation Engine Activated</p>
          </div>
        </div>
      )}
    </>
  );
}

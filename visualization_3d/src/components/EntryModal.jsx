import React from 'react';
import { Layers, FileText, ArrowRight, ShieldCheck, Activity } from 'lucide-react';

export default function EntryModal({ onSelectMode }) {
  return (
    <div className="entry-modal-overlay">
      <div className="entry-modal-card">
        <div className="entry-header">
          <div className="entry-brand-badge">SIH PS 26168 · ISRO / DEPARTMENT OF SPACE</div>
          <h1 className="entry-title">GHOST</h1>
          <p className="entry-subtitle">GNSS-Free Hybrid Onboard Sensor Tracker</p>
          <div className="entry-tagline">“GPS disappeared. But Ghost didn’t.”</div>
        </div>

        <div className="entry-prompt">
          SELECT SYSTEM EXPLORATION MODE
        </div>

        <div className="entry-options-grid">
          {/* Option 1: Prototype Mode */}
          <div className="entry-option-card" onClick={() => onSelectMode('prototype')}>
            <div className="option-header-row">
              <div className="option-icon-box amber-box">
                <Activity size={22} color="#C88A00" />
              </div>
              <span className="option-badge badge-amber">3D SIMULATION</span>
            </div>
            <div className="option-content">
              <h3>PROTOTYPE MODE</h3>
              <p>
                Interactive 3D real-time vehicle trajectory playback during a 60-second GNSS outage.
                Compare raw dead-reckoning drift against GHOST AI-calibrated EKF fusion.
              </p>
              <ul className="option-bullets">
                <li>Dual-vehicle comparative 3D canvas</li>
                <li>Real-time telemetry & outage controls</li>
                <li>Visualized drift accumulation (2.40% vs 69.73%)</li>
              </ul>
            </div>
            <button className="btn-entry-action action-amber">
              <span>LAUNCH PROTOTYPE</span>
              <ArrowRight size={14} />
            </button>
          </div>

          {/* Option 2: Project Mode */}
          <div className="entry-option-card" onClick={() => onSelectMode('project')}>
            <div className="option-header-row">
              <div className="option-icon-box graphite-box">
                <FileText size={22} color="#20251F" />
              </div>
              <span className="option-badge badge-graphite">TECHNICAL DOSSIER</span>
            </div>
            <div className="option-content">
              <h3>PROJECT MODE</h3>
              <p>
                Comprehensive 10-section technical documentation dossier covering system architecture, ML pipeline, EKF state space, and experimental results.
              </p>
              <ul className="option-bullets">
                <li>1D-CNN & Isotonic Speed Calibration</li>
                <li>5-State EKF & OSM Heading Constraints</li>
                <li>Interactive Python code viewers & audit logs</li>
              </ul>
            </div>
            <button className="btn-entry-action action-graphite">
              <span>OPEN PROJECT DOSSIER</span>
              <ArrowRight size={14} />
            </button>
          </div>
        </div>

        <div className="entry-modal-footer">
          <ShieldCheck size={13} color="#16803C" />
          <span>You can switch between Project Mode and Prototype Mode at any time using the mode toggle in the top navigation bar.</span>
        </div>
      </div>
    </div>
  );
}

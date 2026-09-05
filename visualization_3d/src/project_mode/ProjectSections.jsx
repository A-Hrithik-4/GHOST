import React, { useState } from 'react';
import { Layers } from 'lucide-react';
import CodeViewer from './CodeViewer';

// Reusable Footer Component across all Project Mode pages
export function FinalProjectFooter({ onSwitchToPrototype }) {
  return (
    <footer className="project-footer" style={{
      marginTop: '40px',
      paddingTop: '20px',
      borderTop: '1.5px solid var(--border-light)',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontSize: '12px',
      color: 'var(--text-secondary)'
    }}>
      <div>
        <strong style={{ color: 'var(--graphite-dark)' }}>GHOST</strong> — GNSS-Free Hybrid Onboard Sensor Tracker
        <span style={{ marginLeft: '10px', fontFamily: 'JetBrains Mono, monospace', fontWeight: '800', color: 'var(--amber-primary)' }}>
          SIH 26168
        </span>
        <div style={{ fontStyle: 'italic', marginTop: '2px', color: 'var(--text-muted)' }}>
          “GPS disappeared. But Ghost didn’t.”
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span className="tech-indicator-pill">PROJECT MODE</span>
        <button 
          onClick={onSwitchToPrototype}
          style={{
            background: 'var(--amber-soft-bg)',
            border: '1px solid var(--amber-border)',
            borderRadius: '4px',
            padding: '6px 12px',
            color: 'var(--amber-primary)',
            fontWeight: '800',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Layers size={13} />
          <span>OPEN PROTOTYPE</span>
        </button>
      </div>
    </footer>
  );
}

// 01 — OVERVIEW
export function OverviewSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">01 · OVERVIEW</span>
        <h2 className="section-title">OVERVIEW</h2>
        <div className="section-subtitle">GHOST — GNSS-Free Hybrid Onboard Sensor Tracker</div>
      </div>

      {/* HERO STATEMENT */}
      <div className="hero-statement-card" style={{
        background: 'var(--bg-card)',
        border: '1.5px solid var(--amber-border)',
        borderRadius: '8px',
        padding: '24px',
        marginBottom: '24px',
        boxShadow: '0 4px 12px rgba(200, 138, 0, 0.06)'
      }}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '22px', fontWeight: '900', color: 'var(--amber-primary)', letterSpacing: '0.5px' }}>
          “GPS disappeared. But Ghost didn’t.”
        </div>
        <p style={{ marginTop: '12px', fontSize: '14.5px', color: 'var(--text-primary)', lineHeight: '1.6' }}>
          GHOST is an AI-assisted dead-reckoning navigation system designed to maintain road-level positioning continuity during temporary GNSS outages.
          The system combines inertial sensing, machine-learning-based speed estimation, speed calibration, Extended Kalman Filtering, and road-heading constraints to limit positional drift when GNSS becomes unavailable.
        </p>
      </div>

      {/* HERO RESULT CARD */}
      <div className="hero-result-card" style={{
        background: 'var(--bg-card)',
        border: '1.5px solid var(--border-light)',
        borderRadius: '8px',
        padding: '24px',
        marginBottom: '24px'
      }}>
        <div className="grid-4-cols" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
          <div className="metric-box highlighted" style={{ borderLeft: '4px solid var(--amber-primary)', paddingLeft: '12px' }}>
            <span className="metric-val mono" style={{ fontSize: '32px', fontWeight: '900', color: 'var(--amber-primary)' }}>2.40%</span>
            <span className="metric-lbl" style={{ fontSize: '11px', fontWeight: '800', color: 'var(--graphite-dark)' }}>BEST DEMONSTRATED POSITIONAL DRIFT</span>
          </div>
          <div className="metric-box" style={{ paddingLeft: '12px' }}>
            <span className="metric-val mono" style={{ fontSize: '24px', fontWeight: '800', color: 'var(--graphite-dark)' }}>27.95 m</span>
            <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>FINAL POSITION ERROR</span>
          </div>
          <div className="metric-box" style={{ paddingLeft: '12px' }}>
            <span className="metric-val mono" style={{ fontSize: '24px', fontWeight: '800', color: 'var(--graphite-dark)' }}>60 s</span>
            <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>GNSS OUTAGE</span>
          </div>
          <div className="metric-box" style={{ paddingLeft: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="metric-val mono" style={{ fontSize: '24px', fontWeight: '800', color: 'var(--graphite-dark)' }}>≤ 10%</span>
              <span className="badge badge-green" style={{ background: 'var(--green-soft-bg)', color: 'var(--green-primary)', padding: '2px 8px', borderRadius: '4px', fontWeight: '900', fontSize: '11px' }}>PASS</span>
            </div>
            <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>SIH TARGET BENCHMARK</span>
          </div>
        </div>

        <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid var(--border-light)', fontSize: '12.5px', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
          The demonstrated 60-second GNSS outage produced 2.40% positional drift over 1162.5 m of travel.
        </div>
      </div>

      {/* OVERVIEW STATEMENT */}
      <div className="content-card" style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', letterSpacing: '0.5px' }}>WHAT GHOST DOES</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          When GNSS is available, the system uses GNSS observations to support navigation.
          <br /><br />
          When GNSS becomes unavailable, GHOST continues estimating the vehicle’s position using onboard motion information and learned speed estimates.
          <br /><br />
          The objective is not to replace GNSS permanently. The objective is to provide positioning continuity during temporary GNSS outages.
        </p>
      </div>

      {/* KEY CAPABILITIES */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '14px' }}>KEY CAPABILITIES</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
          <div className="content-card">
            <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800', fontSize: '12px' }}>01</span>
            <h4 style={{ fontSize: '13px', fontWeight: '800', marginTop: '4px' }}>GNSS OUTAGE HANDLING</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Continues navigation when GNSS measurements become unavailable.</p>
          </div>
          <div className="content-card">
            <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800', fontSize: '12px' }}>02</span>
            <h4 style={{ fontSize: '13px', fontWeight: '800', marginTop: '4px' }}>AI-ASSISTED SPEED ESTIMATION</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>A 1D-CNN estimates vehicle speed from motion-related sensor features.</p>
          </div>
          <div className="content-card">
            <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800', fontSize: '12px' }}>03</span>
            <h4 style={{ fontSize: '13px', fontWeight: '800', marginTop: '4px' }}>SPEED CALIBRATION</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Leakage-free isotonic calibration improves the speed estimate used by navigation.</p>
          </div>
          <div className="content-card">
            <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800', fontSize: '12px' }}>04</span>
            <h4 style={{ fontSize: '13px', fontWeight: '800', marginTop: '4px' }}>DEAD-RECKONING NAVIGATION</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>A 5-state Extended Kalman Filter combines motion information to estimate position.</p>
          </div>
          <div className="content-card">
            <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800', fontSize: '12px' }}>05</span>
            <h4 style={{ fontSize: '13px', fontWeight: '800', marginTop: '4px' }}>ROAD-HEADING CONSTRAINT</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>OpenStreetMap-derived road heading provides an additional directional constraint.</p>
          </div>
        </div>
      </div>

      {/* KEY TAKEAWAY */}
      <div className="callout-box callout-amber" style={{ background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '6px', padding: '14px 18px' }}>
        <strong>KEY TAKEAWAY:</strong> GHOST maintains positioning continuity during temporary GNSS outages.
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 02 — PROBLEM & OBJECTIVE
export function ProblemObjectiveSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">02 · PROBLEM & OBJECTIVE</span>
        <h2 className="section-title">PROBLEM & OBJECTIVE</h2>
      </div>

      {/* THE PROBLEM */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>WHEN GNSS DISAPPEARS</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          GNSS is a critical source of positioning information for modern navigation systems.
          <br /><br />
          However, GNSS signals can temporarily become unreliable or unavailable because of:
        </p>
        <ul style={{ paddingLeft: '20px', marginTop: '8px', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          <li>signal obstruction</li>
          <li>urban environments</li>
          <li>tunnels</li>
          <li>interference</li>
          <li>multipath effects</li>
          <li>temporary signal loss</li>
        </ul>
        <p style={{ marginTop: '10px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
          During an outage, a vehicle must rely on onboard sensors to estimate its movement. This is known as <strong>dead reckoning</strong>.
        </p>
      </div>

      {/* THE DEAD-RECKONING PROBLEM */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>WHY DRIFT HAPPENS</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          Dead reckoning integrates motion over time. Small errors in:
        </p>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', margin: '10px 0' }}>
          {['speed', 'heading', 'acceleration', 'sensor bias', 'orientation'].map((err, i) => (
            <span key={i} className="mono" style={{ background: 'var(--bg-main)', border: '1px solid var(--border-light)', padding: '4px 10px', borderRadius: '4px', fontSize: '12px', color: 'var(--graphite-dark)' }}>
              {err}
            </span>
          ))}
        </div>
        <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>
          accumulate over the outage period. As a result, the estimated position gradually moves away from the actual vehicle trajectory.
        </p>

        {/* Visual concept */}
        <div style={{ marginTop: '16px', padding: '16px', background: 'var(--bg-main)', border: '1px solid var(--border-light)', borderRadius: '6px', textAlign: 'center' }}>
          <div style={{ fontWeight: '800', color: 'var(--green-primary)' }}>GNSS AVAILABLE</div>
          <div className="arch-arrow">↓ Accurate position correction</div>
          <div style={{ fontWeight: '800', color: 'var(--status-error)', marginTop: '8px' }}>GNSS LOST</div>
          <div className="arch-arrow">↓ Dead reckoning</div>
          <div className="arch-arrow">↓ Accumulating error</div>
          <div style={{ fontWeight: '900', color: 'var(--amber-primary)', marginTop: '6px' }}>POSITION DRIFT</div>
        </div>
      </div>

      {/* OBJECTIVE */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>GHOST OBJECTIVE</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          Develop an AI-assisted dead-reckoning system capable of maintaining positioning continuity during a temporary GNSS outage while keeping positional drift within the SIH benchmark.
        </p>
        <div style={{ marginTop: '12px', padding: '10px 14px', background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '4px', fontWeight: '800', color: 'var(--amber-primary)', fontFamily: 'JetBrains Mono, monospace' }}>
          TARGET: ≤ 10% positional drift during the demonstrated GNSS outage.
        </div>
      </div>

      {/* IMPORTANT SCOPE */}
      <div className="callout-box callout-amber" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: '6px', padding: '16px' }}>
        <strong style={{ color: 'var(--graphite-dark)' }}>IMPORTANT SCOPE NOTE:</strong>
        <p style={{ marginTop: '6px', fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          GHOST is designed for temporary GNSS outage handling. It is NOT presented as:
        </p>
        <ul style={{ paddingLeft: '18px', marginTop: '4px', fontSize: '12px', color: 'var(--text-secondary)' }}>
          <li>a permanent GNSS replacement</li>
          <li>a lane-level positioning system</li>
          <li>a universal accuracy guarantee</li>
          <li>a production-certified navigation system</li>
        </ul>
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 03 — SOLUTION
export function SolutionSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">03 · SOLUTION</span>
        <h2 className="section-title">SOLUTION</h2>
        <div className="section-subtitle">A hybrid AI + sensor-fusion approach to reduce dead-reckoning drift.</div>
      </div>

      {/* PIPELINE FLOWCHART */}
      <div className="arch-diagram-box" style={{ marginBottom: '24px' }}>
        <div className="arch-flow-node">IMU / SENSOR DATA</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-flow-node">FEATURE ENGINEERING</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-flow-node node-amber">1D-CNN SPEED ESTIMATION</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-flow-node node-amber">ISOTONIC SPEED CALIBRATION</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-flow-node node-amber">5-STATE EKF</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-flow-node">OSM ROAD-HEADING CONSTRAINT</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-flow-node node-green">POSITION ESTIMATE</div>
      </div>

      {/* COMPONENTS */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        <div className="content-card">
          <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800' }}>COMPONENT 01</span>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginTop: '4px' }}>IMU / SENSOR DATA</h4>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.5' }}>
            The system starts with motion-related onboard sensor measurements. These measurements provide information about vehicle movement even when GNSS is unavailable.
          </p>
        </div>

        <div className="content-card">
          <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800' }}>COMPONENT 02</span>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginTop: '4px' }}>FEATURE ENGINEERING</h4>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.5' }}>
            Raw sensor observations are transformed into model-ready features capturing velocity, acceleration, heading, position changes, rate-of-change measurements, and temporal motion behaviour.
          </p>
        </div>

        <div className="content-card">
          <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800' }}>COMPONENT 03</span>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginTop: '4px' }}>1D-CNN SPEED ESTIMATION</h4>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.5' }}>
            A 1D Convolutional Neural Network learns temporal patterns from sequential sensor features to estimate speed without requiring a large recurrent architecture.
          </p>
        </div>

        <div className="content-card">
          <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800' }}>COMPONENT 04</span>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginTop: '4px' }}>ISOTONIC SPEED CALIBRATION</h4>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.5' }}>
            The raw CNN speed estimate is calibrated using isotonic regression, mapping model outputs toward a reliable speed estimate while preserving monotonic relationships (1D-CNN + Leakage-Free Isotonic Calibration).
          </p>
        </div>

        <div className="content-card">
          <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800' }}>COMPONENT 05</span>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginTop: '4px' }}>5-STATE EKF</h4>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.5' }}>
            The calibrated speed estimate is incorporated into the frozen Phase 6 Extended Kalman Filter to estimate vehicle state while managing sensor uncertainty.
          </p>
        </div>

        <div className="content-card">
          <span className="mono" style={{ color: 'var(--amber-primary)', fontWeight: '800' }}>COMPONENT 06</span>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginTop: '4px' }}>OSM ROAD-HEADING CONSTRAINT</h4>
          <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: '1.5' }}>
            OpenStreetMap road information provides a directional constraint, constraining the estimated vehicle heading toward the road direction rather than allowing unrestricted heading drift.
          </p>
        </div>
      </div>

      {/* KEY TAKEAWAY */}
      <div className="callout-box callout-amber" style={{ background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '6px', padding: '14px 18px' }}>
        <strong>KEY TAKEAWAY:</strong> AI estimates speed; sensor fusion turns that estimate into position.
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 04 — SYSTEM ARCHITECTURE
export function SystemArchitectureSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">04 · SYSTEM ARCHITECTURE</span>
        <h2 className="section-title">SYSTEM ARCHITECTURE</h2>
        <div className="section-subtitle">End-to-end architecture of the GHOST navigation pipeline.</div>
      </div>

      {/* ARCHITECTURE DIAGRAM */}
      <div className="arch-diagram-box" style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '16px' }}>ARCHITECTURE DIAGRAM</h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
          <div className="arch-flow-node" style={{ width: '90%' }}>
            <strong>SENSOR INPUTS</strong> (IMU Accelerometer/Gyroscope + Speed Measurements + GNSS)
          </div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node" style={{ width: '90%' }}>DATA PREPROCESSING</div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node" style={{ width: '90%' }}>FEATURE ENGINEERING</div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node node-amber" style={{ width: '90%' }}>1D-CNN SPEED ESTIMATOR</div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node node-amber" style={{ width: '90%' }}>ISOTONIC CALIBRATION</div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node node-amber" style={{ width: '90%' }}>
            5-STATE EKF <span style={{ fontSize: '11px', color: 'var(--cyan-tech)', marginLeft: '8px' }}>[+ OSM ROAD HEADING]</span>
          </div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node node-green" style={{ width: '90%' }}>POSITION ESTIMATE</div>
          <div className="arch-arrow">│ ▼</div>
          <div className="arch-flow-node node-green" style={{ width: '90%' }}>NAVIGATION OUTPUT</div>
        </div>
      </div>

      {/* MODES COMPARISON */}
      <div className="grid-2-cols" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        <div className="content-card" style={{ borderLeft: '4px solid var(--green-primary)' }}>
          <h4 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--green-primary)' }}>GNSS AVAILABLE MODE</h4>
          <div style={{ marginTop: '12px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
            Sensor Data + GNSS Position
            <br />↓<br />
            EKF Update
            <br />↓<br />
            <strong>Corrected Position</strong>
          </div>
        </div>

        <div className="content-card" style={{ borderLeft: '4px solid var(--amber-primary)' }}>
          <h4 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--amber-primary)' }}>GNSS OUTAGE MODE</h4>
          <div style={{ marginTop: '12px', fontSize: '12.5px', color: 'var(--graphite-dark)' }}>
            IMU / Motion Data
            <br />↓<br />
            1D-CNN Speed
            <br />↓<br />
            Isotonic Calibration
            <br />↓<br />
            EKF Prediction + Road Heading Constraint
            <br />↓<br />
            <strong>Continuous Position Estimate</strong>
          </div>
        </div>
      </div>

      {/* ARCHITECTURE NOTE */}
      <div className="callout-box callout-amber" style={{ background: 'var(--bg-card)', border: '1.5px solid var(--border-light)', borderRadius: '6px', padding: '16px' }}>
        <strong style={{ color: 'var(--graphite-dark)' }}>IMPORTANT ARCHITECTURE NOTE:</strong>
        <p style={{ marginTop: '6px', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          The ML model does not directly predict the vehicle’s position. The ML component estimates speed. The navigation layer then uses the calibrated speed inside the sensor-fusion pipeline. This separation keeps learning and navigation responsibilities clearly defined.
        </p>
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 05 — DATA & FEATURE ENGINEERING
export function DataFeatureEngineeringSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">05 · DATA & FEATURE ENGINEERING</span>
        <h2 className="section-title">DATA & FEATURE ENGINEERING</h2>
      </div>

      {/* DATASET SUMMARY */}
      <div className="grid-4-cols" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '14px', marginBottom: '24px' }}>
        <div className="metric-box">
          <span className="metric-val mono" style={{ fontSize: '26px', fontWeight: '900', color: 'var(--graphite-dark)' }}>50,000</span>
          <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>ORIGINAL RECORDS</span>
        </div>
        <div className="metric-box">
          <span className="metric-val mono" style={{ fontSize: '26px', fontWeight: '900', color: 'var(--graphite-dark)' }}>50</span>
          <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>TRACKS</span>
        </div>
        <div className="metric-box highlighted">
          <span className="metric-val mono" style={{ fontSize: '26px', fontWeight: '900', color: 'var(--amber-primary)' }}>49,950</span>
          <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--graphite-dark)' }}>ML-READY RECORDS</span>
        </div>
        <div className="metric-box">
          <span className="metric-val mono" style={{ fontSize: '26px', fontWeight: '900', color: 'var(--graphite-dark)' }}>15</span>
          <span className="metric-lbl" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>FEATURES USED</span>
        </div>
      </div>

      {/* WHY 49,950? */}
      <div className="content-card" style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>WHY 50 RECORDS ARE REMOVED</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          Each of the 50 tracks has a first observation. The first observation does not contain previous history required to calculate derived temporal features such as: RangeRate, HeadingRate, VelocityChange, PositionChange.
        </p>
        <div style={{ marginTop: '12px', padding: '10px 14px', background: 'var(--bg-main)', border: '1px solid var(--border-light)', borderRadius: '4px', fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: 'var(--graphite-dark)' }}>
          50,000 total records − 50 first observations = <strong>49,950 ML-ready records</strong>
        </div>
      </div>

      {/* FEATURE ENGINEERING */}
      <div className="content-card" style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '14px' }}>ENGINEERED FEATURES (15 SENSOR-DERIVED FEATURES)</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
          <div style={{ background: 'var(--bg-main)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-light)' }}>
            <span className="mono" style={{ fontSize: '11px', fontWeight: '800', color: 'var(--amber-primary)' }}>ACCELERATION (9)</span>
            <div className="mono" style={{ fontSize: '12px', color: 'var(--graphite-dark)', marginTop: '6px', lineHeight: '1.6' }}>
              acc_x_clean, acc_y_clean, acc_z_clean, longitudinal_acc, lateral_acc, vertical_acc, acc_magnitude, acc_mean, acc_std
            </div>
          </div>

          <div style={{ background: 'var(--bg-main)', padding: '12px', borderRadius: '6px', border: '1px solid var(--border-light)' }}>
            <span className="mono" style={{ fontSize: '11px', fontWeight: '800', color: 'var(--amber-primary)' }}>HEADING & ROTATION (6)</span>
            <div className="mono" style={{ fontSize: '12px', color: 'var(--graphite-dark)', marginTop: '6px', lineHeight: '1.6' }}>
              gyro_x_clean, gyro_y_clean, gyro_z_clean, yaw_rate, gyro_magnitude, gyro_std
            </div>
          </div>
        </div>
      </div>

      {/* DATA PROCESSING FLOW */}
      <div className="content-card">
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '12px' }}>DATA PROCESSING FLOW</h3>
        <div className="arch-diagram-box" style={{ padding: '12px' }}>
          RAW LOG DATA → PARSING → CLEANING → TRACK IDENTIFICATION → TEMPORAL FEATURE ENGINEERING → ML-READY DATA
        </div>
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 06 — ML MODEL
export function MlModelSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">06 · ML MODEL</span>
        <h2 className="section-title">ML MODEL</h2>
        <div className="section-subtitle">1D-CNN speed estimation with leakage-free isotonic calibration.</div>
      </div>

      {/* MODEL OVERVIEW */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>1D-CNN SPEED ESTIMATOR</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          GHOST uses a one-dimensional convolutional neural network to estimate vehicle speed from sequential motion-related features.
          The model learns temporal patterns in the sensor data and produces a predicted speed.
        </p>
      </div>

      {/* MODEL FLOW */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '12px' }}>MODEL FLOW</h3>
        <div className="arch-diagram-box" style={{ padding: '14px' }}>
          INPUT FEATURES → 1D CONVOLUTION → ACTIVATION → FEATURE EXTRACTION → DENSE LAYERS → SPEED PREDICTION
        </div>
      </div>

      {/* WHY 1D-CNN? */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '12px' }}>WHY 1D-CNN?</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
          <div className="content-card">
            <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>TEMPORAL PATTERNS</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Learns local patterns across sequential sensor observations.</p>
          </div>
          <div className="content-card">
            <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>COMPUTATIONAL EFFICIENCY</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Provides a relatively lightweight architecture for sequential data.</p>
          </div>
          <div className="content-card">
            <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>FEATURE LEARNING</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Learns useful representations directly from engineered motion features.</p>
          </div>
        </div>
      </div>

      {/* ISOTONIC CALIBRATION */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>LEAKAGE-FREE SPEED CALIBRATION</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          The raw 1D-CNN speed prediction is passed through isotonic regression. Isotonic regression learns a monotonic mapping between predicted and reference speed values. The calibration stage improves the speed estimate used by the navigation filter.
        </p>
      </div>

      {/* LEAKAGE PREVENTION CARD */}
      <div className="content-card" style={{ borderLeft: '4px solid var(--green-primary)', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--green-primary)' }}>LEAKAGE AUDIT</h3>
          <span className="badge badge-green" style={{ background: 'var(--green-soft-bg)', color: 'var(--green-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900' }}>PASS</span>
        </div>
        <p style={{ marginTop: '8px', fontSize: '13px', color: 'var(--text-secondary)' }}>
          Calibration was evaluated using a leakage-free methodology.
          <br />
          <em>Important: Do not imply that future outage data was used to train the calibration.</em>
        </p>
      </div>

      {/* MODEL OUTPUT */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '10px' }}>MODEL OUTPUT FLOW</h3>
        <div className="arch-diagram-box" style={{ padding: '12px' }}>
          RAW CNN SPEED → ISOTONIC CALIBRATION → CALIBRATED SPEED → EKF
        </div>
      </div>

      {/* KEY TAKEAWAY */}
      <div className="callout-box callout-amber" style={{ background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '6px', padding: '14px 18px' }}>
        <strong>KEY TAKEAWAY:</strong> Isotonic calibration improved the speed estimate used by the navigation system.
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 07 — NAVIGATION & EKF
export function NavigationEkfSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">07 · NAVIGATION & EKF</span>
        <h2 className="section-title">NAVIGATION & EKF</h2>
        <div className="section-subtitle">Sensor fusion converts calibrated motion estimates into continuous positioning.</div>
      </div>

      {/* 5-STATE EKF */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>5-STATE EKF STATE VECTOR</h3>
        
        <div className="mono" style={{ fontSize: '20px', fontWeight: '900', color: 'var(--amber-primary)', padding: '14px', background: 'var(--bg-main)', borderRadius: '6px', textAlign: 'center', margin: '14px 0', border: '1px solid var(--border-light)' }}>
          x = [ x, y, v_x, v_y, ψ ]ᵀ
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px', marginTop: '12px' }}>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}><strong>x</strong> = position along X</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}><strong>y</strong> = position along Y</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}><strong>v_x</strong> = velocity along X</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}><strong>v_y</strong> = velocity along Y</div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}><strong>ψ</strong> = heading / yaw</div>
        </div>
      </div>

      {/* EKF PIPELINE */}
      <div className="grid-2-cols" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px', marginBottom: '20px' }}>
        <div className="content-card">
          <h4 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--amber-primary)' }}>PREDICTION</h4>
          <p style={{ marginTop: '8px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
            Sensor measurements → Motion model → Predicted state
          </p>
        </div>
        <div className="content-card">
          <h4 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--green-primary)' }}>UPDATE</h4>
          <p style={{ marginTop: '8px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
            Available measurement / constraint → Kalman update → Corrected state
          </p>
        </div>
      </div>

      {/* AVAILABILITY VS OUTAGE */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '14px', fontWeight: '800', color: 'var(--graphite-dark)' }}>GNSS AVAILABILITY VS OUTAGE BEHAVIOUR</h3>
        <div style={{ marginTop: '10px', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          <strong>DURING GNSS AVAILABILITY:</strong> GNSS measurements can correct the estimated state.
          <br /><br />
          <strong>DURING GNSS OUTAGE:</strong> The system relies on: IMU motion information, calibrated speed, heading estimation, EKF state propagation, and road-heading constraint to continue estimating position.
        </div>
      </div>

      {/* ROAD-HEADING CONSTRAINT */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>OSM ROAD HEADING CONSTRAINT</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          OpenStreetMap road information provides an external directional constraint.
          Instead of allowing heading to drift freely, the navigation layer can use the road direction as additional information about the expected vehicle heading.
        </p>
        <div style={{ marginTop: '12px', padding: '12px', background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '4px', fontSize: '12.5px', color: 'var(--graphite-dark)' }}>
          <strong>WHY THIS MATTERS:</strong> Small heading errors can create large positional errors during dead reckoning. A road-heading constraint therefore helps limit uncontrolled directional drift.
        </div>
      </div>

      {/* KEY TAKEAWAY */}
      <div className="callout-box callout-amber" style={{ background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '6px', padding: '14px 18px' }}>
        <strong>KEY TAKEAWAY:</strong> The EKF combines motion information and directional constraints to limit drift.
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 08 — EXPERIMENTS & RESULTS
export function ExperimentsResultsSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">08 · EXPERIMENTS & RESULTS</span>
        <h2 className="section-title">EXPERIMENTS & RESULTS</h2>
        <div className="section-subtitle">Progression from the Phase 6 baseline to the final calibrated model.</div>
      </div>

      {/* EXPERIMENT TABLE */}
      <div className="content-card" style={{ marginBottom: '24px', overflowX: 'auto' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '14px' }}>EXPERIMENTAL COMPARISON</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
          <thead>
            <tr style={{ background: 'var(--bg-main)', borderBottom: '2px solid var(--border-light)', textTransform: 'uppercase', fontSize: '11px', color: 'var(--text-secondary)' }}>
              <th style={{ padding: '10px', textAlign: 'left' }}>Configuration</th>
              <th style={{ padding: '10px', textAlign: 'right' }}>Final Position Error</th>
              <th style={{ padding: '10px', textAlign: 'right' }}>Positional Drift</th>
              <th style={{ padding: '10px', textAlign: 'center' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '12px', fontWeight: '700' }}>Phase 6 Raw CNN</td>
              <td style={{ padding: '12px', textAlign: 'right' }} className="mono">217.67 m</td>
              <td style={{ padding: '12px', textAlign: 'right' }} className="mono">18.72%</td>
              <td style={{ padding: '12px', textAlign: 'center' }}><span style={{ color: 'var(--status-error)', fontWeight: '800' }}>EXCEEDS TARGET</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '12px', fontWeight: '700' }}>Experiment 1 — Linear Calibration</td>
              <td style={{ padding: '12px', textAlign: 'right' }} className="mono">52.80 m</td>
              <td style={{ padding: '12px', textAlign: 'right' }} className="mono">4.54%</td>
              <td style={{ padding: '12px', textAlign: 'center' }}><span style={{ color: 'var(--status-warning)', fontWeight: '800' }}>IMPROVED</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
              <td style={{ padding: '12px', fontWeight: '700' }}>Experiment 4A — Soft Gate</td>
              <td style={{ padding: '12px', textAlign: 'right' }} className="mono">65.09 m</td>
              <td style={{ padding: '12px', textAlign: 'right' }} className="mono">5.60%</td>
              <td style={{ padding: '12px', textAlign: 'center' }}><span style={{ color: 'var(--status-warning)', fontWeight: '800' }}>ALTERNATIVE</span></td>
            </tr>
            <tr style={{ background: 'var(--amber-soft-bg)', borderBottom: '2px solid var(--amber-border)' }}>
              <td style={{ padding: '12px', fontWeight: '900', color: 'var(--amber-primary)' }}>Experiment 3 — Isotonic Calibration</td>
              <td style={{ padding: '12px', textAlign: 'right', fontWeight: '900' }} className="mono">27.95 m</td>
              <td style={{ padding: '12px', textAlign: 'right', fontWeight: '900', color: 'var(--green-primary)' }} className="mono">2.40%</td>
              <td style={{ padding: '12px', textAlign: 'center' }}>
                <span className="badge badge-green" style={{ background: 'var(--green-primary)', color: '#FFF', padding: '3px 8px', borderRadius: '4px', fontWeight: '900', fontSize: '10px' }}>
                  SELECTED FINAL CONFIGURATION
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* DETAILED EXPERIMENTS */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px', marginBottom: '24px' }}>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800' }}>PHASE 6 BASELINE</h4>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '800', color: 'var(--status-error)', marginTop: '4px' }}>217.67 m (18.72%)</div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Raw CNN baseline exceeded the SIH benchmark of ≤10%.</p>
        </div>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800' }}>EXPERIMENT 1: LINEAR CALIBRATION</h4>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '800', color: 'var(--graphite-dark)', marginTop: '4px' }}>52.80 m (4.54%)</div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Linear calibration stage substantially reduced drift.</p>
        </div>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800' }}>EXPERIMENT 4A: SOFT GATE</h4>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '800', color: 'var(--graphite-dark)', marginTop: '4px' }}>65.09 m (5.60%)</div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Soft gating improved baseline but did not beat isotonic.</p>
        </div>
        <div className="content-card" style={{ border: '1.5px solid var(--green-border)' }}>
          <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--green-primary)' }}>EXPERIMENT 3: ISOTONIC CALIBRATION</h4>
          <div className="mono" style={{ fontSize: '18px', fontWeight: '900', color: 'var(--green-primary)', marginTop: '4px' }}>27.95 m (2.40%)</div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>Produced the best demonstrated result.</p>
        </div>
      </div>

      {/* FINAL BENCHMARK CARD & BENCHMARK MARGIN */}
      <div className="hero-result-card" style={{ background: 'var(--bg-card)', border: '1.5px solid var(--amber-border)', borderRadius: '8px', padding: '20px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div className="mono" style={{ fontSize: '36px', fontWeight: '900', color: 'var(--amber-primary)' }}>2.40% DRIFT</div>
            <div style={{ fontSize: '13px', fontWeight: '800', color: 'var(--graphite-dark)', marginTop: '2px' }}>
              SIH TARGET: ≤10% | STATUS: <span style={{ color: 'var(--green-primary)', fontWeight: '900' }}>PASS</span>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              60-second GNSS outage · 1162.5 m outage distance · 27.95 m final position error
            </div>
          </div>
          <div style={{ background: 'var(--green-soft-bg)', border: '1px solid var(--green-border)', padding: '14px 18px', borderRadius: '6px' }}>
            <div style={{ fontSize: '13px', fontWeight: '800', color: 'var(--green-primary)' }}>BENCHMARK MARGIN</div>
            <div className="mono" style={{ fontSize: '14px', fontWeight: '900', color: 'var(--graphite-dark)', marginTop: '2px' }}>
              7.60 percentage points below 10% limit
            </div>
            <div className="mono" style={{ fontSize: '13px', fontWeight: '800', color: 'var(--green-primary)', marginTop: '2px' }}>
              76% below maximum allowable drift limit
            </div>
            <div style={{ fontSize: '10.5px', color: 'var(--text-secondary)', marginTop: '4px', fontStyle: 'italic' }}>
              (Note: 76% below 10% threshold; not "76% accuracy")
            </div>
          </div>
        </div>
      </div>

      {/* FINAL MODEL DEFINITION */}
      <div className="content-card" style={{ marginBottom: '24px' }}>
        <h4 style={{ fontSize: '12px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase' }}>FINAL SELECTED SYSTEM CONFIGURATION</h4>
        <div className="mono" style={{ fontSize: '13px', fontWeight: '800', color: 'var(--graphite-dark)', marginTop: '6px', lineHeight: '1.5' }}>
          GHOST 1D-CNN Speed Estimator + Leakage-Free Isotonic Speed Calibration + Frozen Phase 6 5-State EKF + OSM Road-Heading Constraint
        </div>
        <div style={{ marginTop: '8px' }}>
          <span className="badge badge-green" style={{ background: 'var(--green-soft-bg)', color: 'var(--green-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900', fontSize: '11px' }}>
            TECHNICAL READINESS: PASS
          </span>
        </div>
      </div>

      {/* SUBSECTION: ERROR ANALYSIS */}
      <div className="content-card" style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)' }}>ERROR ANALYSIS</h3>
        <p style={{ marginTop: '8px', fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          The dominant dead-reckoning challenge is accumulated error over time.
          Potential contributors include: speed estimation error, heading error, sensor noise, sensor bias, and accumulated integration error.
          <br /><br />
          The calibration stage reduces speed-estimation error before the estimate reaches the navigation filter.
          The road-heading constraint provides additional directional information.
        </p>

        {/* IMPORTANT DIAGNOSTIC */}
        <div style={{ marginTop: '14px', padding: '14px', background: 'var(--bg-main)', border: '1px solid var(--border-light)', borderRadius: '6px' }}>
          <div style={{ fontSize: '12px', fontWeight: '800', color: 'var(--graphite-dark)' }}>GROUND-TRUTH SPEED DIAGNOSTIC REFERENCE</div>
          <div className="mono" style={{ fontSize: '14px', fontWeight: '800', color: 'var(--cyan-tech)', marginTop: '4px' }}>
            44.80 m final error | 3.85% final error
          </div>
          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px', fontStyle: 'italic' }}>
            IMPORTANT: This is a diagnostic reference only and is NOT the deployable/ranked GHOST configuration. Do not present it as the final model result.
          </p>
        </div>
      </div>

      {/* KEY TAKEAWAY */}
      <div className="callout-box callout-amber" style={{ background: 'var(--amber-soft-bg)', border: '1px solid var(--amber-border)', borderRadius: '6px', padding: '14px 18px' }}>
        <strong>KEY TAKEAWAY:</strong> The final demonstrated configuration achieved 2.40% positional drift over a 60-second GNSS outage.
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 09 — IMPLEMENTATION & CODE
export function ImplementationCodeSection({ onSwitchToPrototype }) {
  const [activeTab, setActiveTab] = useState('ml_model');

  const codeSnippets = {
    ml_model: {
      filename: 'preprocessing/speed_prediction.py',
      language: 'python',
      code: `import torch
import torch.nn as nn

class SpeedCNN1D(nn.Module):
    """
    1D-CNN Speed Estimator for GHOST Dead-Reckoning Pipeline.
    Estimates longitudinal vehicle speed from sequential 15-feature sensor windows.
    """
    def __init__(self, input_features=15, seq_len=30):
        super(SpeedCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features) -> transpose to (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        speed_pred = self.fc2(x)
        return speed_pred`
    },
    calibration: {
      filename: 'preprocessing/phase7_exp3_speed_regime_calibration.py',
      language: 'python',
      code: `from sklearn.isotonic import IsotonicRegression
import numpy as np

def train_leakage_free_isotonic_calibration(y_val_pred, y_val_true):
    """
    Leakage-Free Isotonic Speed Calibration.
    Maps 1D-CNN raw predicted speed toward ground-truth reference monotonically.
    Must be fit exclusively on Validation split.
    """
    iso_calibrator = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso_calibrator.fit(y_val_pred, y_val_true)
    return iso_calibrator

def apply_calibration(iso_calibrator, raw_speed_preds):
    """
    Applies calibrated speed transformation before passing into EKF.
    """
    calibrated_speed = iso_calibrator.predict(raw_speed_preds)
    return np.maximum(0.0, calibrated_speed)`
    },
    ekf: {
      filename: 'preprocessing/phase6_ekf_fusion.py',
      language: 'python',
      code: `import numpy as np

def ekf_predict(state, cov, speed_calibrated, yaw_rate, dt):
    """
    Frozen Phase 6 5-State Extended Kalman Filter Prediction Step.
    State vector x = [px, py, vx, vy, yaw]^T
    """
    px, py, vx, vy, yaw = state
    
    # Kinematic propagation using calibrated speed and IMU yaw rate
    new_yaw = yaw + yaw_rate * dt
    new_px = px + speed_calibrated * np.cos(new_yaw) * dt
    new_py = py + speed_calibrated * np.sin(new_yaw) * dt
    new_vx = speed_calibrated * np.cos(new_yaw)
    new_vy = speed_calibrated * np.sin(new_yaw)
    
    new_state = np.array([new_px, new_py, new_vx, new_vy, new_yaw])
    
    # Jacobian matrix calculation & covariance propagation
    F = np.eye(5)
    F[0, 2] = dt
    F[1, 3] = dt
    F[2, 4] = -speed_calibrated * np.sin(new_yaw) * dt
    F[3, 4] = speed_calibrated * np.cos(new_yaw) * dt
    
    Q = np.diag([0.05, 0.05, 0.1, 0.1, 0.001])
    new_cov = F @ cov @ F.T + Q
    
    return new_state, new_cov`
    },
    evaluation: {
      filename: 'preprocessing/phase7_final_validation_audit.py',
      language: 'python',
      code: `import numpy as np

def verify_sih_benchmark(est_positions, gt_positions, outage_distance=1162.5):
    """
    Evaluates final positional drift against SIH target (<= 10%).
    """
    final_err = np.linalg_norm(est_positions[-1] - gt_positions[-1])
    drift_percent = (final_err / outage_distance) * 100.0
    
    sih_target = 10.0
    is_pass = drift_percent <= sih_target
    
    return {
        'final_position_error_m': round(final_err, 2),
        'drift_percentage': round(drift_percent, 2),
        'sih_target': sih_target,
        'status': 'PASS' if is_pass else 'FAIL'
    }`
    },
    data_processing: {
      filename: 'preprocessing/signal_pipeline.py',
      language: 'python',
      code: `import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff_hz=3.0, fs_hz=10.0):
    """
    2nd-order Butterworth low-pass filter to smooth accelerometer noise.
    """
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(2, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def clean_and_extract_features(df):
    """
    Extracts 15 permitted motion features from 10 Hz raw sensor observations.
    """
    acc_x = lowpass_filter(df['acc_x'].values)
    acc_y = lowpass_filter(df['acc_y'].values)
    acc_z = lowpass_filter(df['acc_z'].values)
    
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    # Return feature dictionary
    return {
        'acc_x_clean': acc_x,
        'acc_y_clean': acc_y,
        'acc_z_clean': acc_z,
        'acc_magnitude': acc_magnitude
    }`
    }
  };

  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">09 · IMPLEMENTATION & CODE</span>
        <h2 className="section-title">IMPLEMENTATION & CODE</h2>
        <div className="section-subtitle">The implementation is available as part of the GHOST project repository.</div>
      </div>

      {/* PROJECT STRUCTURE TREE */}
      <div className="content-card" style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '12px' }}>PROJECT REPOSITORY STRUCTURE</h3>
        <pre className="mono" style={{ background: 'var(--code-bg)', color: '#D8D2C6', padding: '16px', borderRadius: '6px', fontSize: '12px', lineHeight: '1.5', overflowX: 'auto' }}>
{`GHOST/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
├── visualization_3d/
│   ├── public/
│   │   └── models/
│   ├── src/
│   │   ├── components/
│   │   ├── data/
│   │   ├── project_mode/
│   │   ├── styles/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.js
│
├── README.md
└── .gitignore`}
        </pre>
      </div>

      {/* IMPLEMENTATION AREAS */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px', marginBottom: '24px' }}>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>DATA PIPELINE</h4>
          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>Parsing · Cleaning · Track processing · Feature engineering</p>
        </div>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>ML PIPELINE</h4>
          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>1D-CNN · Training · Prediction · Evaluation</p>
        </div>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>CALIBRATION</h4>
          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>Isotonic regression · Leakage-free calibration · Speed transformation</p>
        </div>
        <div className="content-card">
          <h4 style={{ fontSize: '13px', fontWeight: '800', color: 'var(--amber-primary)' }}>NAVIGATION</h4>
          <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>5-state EKF · State propagation · Measurement updates · Road constraint</p>
        </div>
      </div>

      {/* CODE VIEWER WITH TABS */}
      <div className="content-card">
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '14px' }}>SOURCE CODE EXPLORER</h3>
        
        {/* Tab switcher */}
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
          {[
            { id: 'ml_model', label: 'ML MODEL' },
            { id: 'calibration', label: 'CALIBRATION' },
            { id: 'ekf', label: 'EKF' },
            { id: 'evaluation', label: 'EVALUATION' },
            { id: 'data_processing', label: 'DATA PROCESSING' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="mono"
              style={{
                padding: '6px 12px',
                borderRadius: '4px',
                border: '1px solid var(--border-light)',
                background: activeTab === tab.id ? 'var(--amber-primary)' : 'var(--bg-main)',
                color: activeTab === tab.id ? '#FFF' : 'var(--graphite-dark)',
                fontSize: '11px',
                fontWeight: '800',
                cursor: 'pointer'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <CodeViewer
          filename={codeSnippets[activeTab].filename}
          language={codeSnippets[activeTab].language}
          code={codeSnippets[activeTab].code}
        />
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

// 10 — LIMITATIONS & FUTURE SCOPE
export function LimitationsFutureScopeSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">10 · LIMITATIONS & FUTURE SCOPE</span>
        <h2 className="section-title">LIMITATIONS & FUTURE SCOPE</h2>
      </div>

      {/* CURRENT LIMITATIONS */}
      <div style={{ marginBottom: '28px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '14px' }}>CURRENT LIMITATIONS</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
          <div className="content-card">
            <h4 style={{ fontSize: '13.5px', fontWeight: '800', color: 'var(--status-warning)' }}>TEMPORARY GNSS OUTAGE</h4>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.5' }}>
              The demonstrated benchmark covers a 60-second GNSS outage. Longer outages may accumulate additional error.
            </p>
          </div>
          <div className="content-card">
            <h4 style={{ fontSize: '13.5px', fontWeight: '800', color: 'var(--status-warning)' }}>DATASET / ENVIRONMENT</h4>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.5' }}>
              Performance is demonstrated on the available project dataset and evaluation scenario. Broader real-world environments require additional validation.
            </p>
          </div>
          <div className="content-card">
            <h4 style={{ fontSize: '13.5px', fontWeight: '800', color: 'var(--status-warning)' }}>SENSOR DEPENDENCY</h4>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.5' }}>
              Dead reckoning remains dependent on the quality and reliability of onboard sensor measurements.
            </p>
          </div>
          <div className="content-card">
            <h4 style={{ fontSize: '13.5px', fontWeight: '800', color: 'var(--status-warning)' }}>NOT A GNSS REPLACEMENT</h4>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.5' }}>
              GHOST is intended to provide continuity during temporary GNSS outages rather than permanently replace GNSS.
            </p>
          </div>
          <div className="content-card">
            <h4 style={{ fontSize: '13.5px', fontWeight: '800', color: 'var(--status-warning)' }}>NOT LANE-LEVEL ACCURACY</h4>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.5' }}>
              The project demonstrates positioning continuity and drift reduction. It does not claim lane-level positioning accuracy.
            </p>
          </div>
        </div>
      </div>

      {/* FUTURE SCOPE TIMELINE */}
      <div className="content-card">
        <h3 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--graphite-dark)', marginBottom: '16px' }}>FUTURE SCOPE</h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
            <span className="mono" style={{ background: 'var(--amber-soft-bg)', color: 'var(--amber-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900', fontSize: '12px' }}>01</span>
            <div>
              <strong style={{ fontSize: '13px', color: 'var(--graphite-dark)' }}>LONGER OUTAGE VALIDATION</strong>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>Evaluate performance across longer GNSS outage durations.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
            <span className="mono" style={{ background: 'var(--amber-soft-bg)', color: 'var(--amber-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900', fontSize: '12px' }}>02</span>
            <div>
              <strong style={{ fontSize: '13px', color: 'var(--graphite-dark)' }}>REAL VEHICLE TESTING</strong>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>Validate the system using real onboard hardware and vehicle motion.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
            <span className="mono" style={{ background: 'var(--amber-soft-bg)', color: 'var(--amber-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900', fontSize: '12px' }}>03</span>
            <div>
              <strong style={{ fontSize: '13px', color: 'var(--graphite-dark)' }}>MORE DIVERSE ENVIRONMENTS</strong>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>Evaluate across different road types, speeds, weather conditions, and urban environments.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
            <span className="mono" style={{ background: 'var(--amber-soft-bg)', color: 'var(--amber-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900', fontSize: '12px' }}>04</span>
            <div>
              <strong style={{ fontSize: '13px', color: 'var(--graphite-dark)' }}>HARDWARE OPTIMIZATION</strong>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>Explore deployment on embedded/edge hardware.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
            <span className="mono" style={{ background: 'var(--amber-soft-bg)', color: 'var(--amber-primary)', padding: '4px 10px', borderRadius: '4px', fontWeight: '900', fontSize: '12px' }}>05</span>
            <div>
              <strong style={{ fontSize: '13px', color: 'var(--graphite-dark)' }}>ROBUST SENSOR FUSION</strong>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>Investigate additional sensor sources and stronger adaptive fusion strategies.</p>
            </div>
          </div>
        </div>
      </div>

      <FinalProjectFooter onSwitchToPrototype={onSwitchToPrototype} />
    </div>
  );
}

import React from 'react';

/**
 * ProjectSections.jsx
 * 
 * Contains all 12 technical story sections for GHOST Project Mode.
 * Zero emojis, clean typography, structured for high-end technical presentation.
 */

// 01. Overview Section
export function OverviewSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="hero-box">
        <div className="section-tag">01 · OVERVIEW</div>
        <div className="hero-headline">WHEN GNSS DISAPPEARS, GHOST KEEPS THE VEHICLE MOVING.</div>
        <div className="hero-subtitle">
          GHOST is an AI-assisted dead-reckoning system designed to maintain positioning continuity during temporary GNSS outages.
        </div>
        <p style={{ marginTop: '14px', fontSize: '14.5px', color: '#5B6575', lineHeight: '1.65' }}>
          It combines onboard IMU signals, an AI-based speed estimator, speed calibration, sensor fusion, and road-network constraints to estimate vehicle motion when GNSS is temporarily unavailable.
        </p>
      </div>

      <div className="section-header" style={{ marginTop: '32px' }}>
        <span className="section-tag">DEMONSTRATED RESULT</span>
      </div>

      <div className="grid-5-cols">
        <div className="metric-card accent-green">
          <span className="val-large val-green">2.40%</span>
          <span className="lbl-card">Positional drift</span>
        </div>
        <div className="metric-card accent-green">
          <span className="val-large val-green">27.95 m</span>
          <span className="lbl-card">Final position error</span>
        </div>
        <div className="metric-card accent-blue">
          <span className="val-large val-blue">60 s</span>
          <span className="lbl-card">GNSS outage duration</span>
        </div>
        <div className="metric-card accent-blue">
          <span className="val-large val-blue">1162.5 m</span>
          <span className="lbl-card">Distance travelled</span>
        </div>
        <div className="metric-card accent-blue">
          <span className="val-large val-blue">≤10%</span>
          <span className="lbl-card">SIH benchmark</span>
        </div>
      </div>

      <p className="disclaimer-pill" style={{ marginBottom: '28px' }}>
        Best demonstrated result on the evaluated 60-second GNSS outage trajectory.
      </p>

      <div className="content-card">
        <h3>THE GOAL</h3>
        <p>
          GHOST is not designed to permanently replace GNSS.
        </p>
        <p style={{ marginTop: '8px' }}>
          Its purpose is simpler:
        </p>
        <p style={{ marginTop: '8px', fontWeight: '700', color: '#0B1220' }}>
          When GNSS temporarily disappears, maintain a reliable estimate of where the vehicle is going.
        </p>
      </div>

      <div style={{ marginTop: '24px' }}>
        <button className="btn-cta" onClick={onSwitchToPrototype}>
          LAUNCH LIVE PROTOTYPE →
        </button>
      </div>
    </div>
  );
}

// 02. Problem Section
export function ProblemSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">02 · THE PROBLEM</span>
        <h2 className="section-title">GNSS IS NOT ALWAYS AVAILABLE.</h2>
        <p className="section-desc">
          Vehicle navigation systems normally depend heavily on GNSS for absolute positioning.
        </p>
      </div>

      <div className="content-card">
        <h3>Temporary signal loss can occur because of:</h3>
        <ul className="clean-list" style={{ marginTop: '12px', paddingLeft: '20px', color: '#5B6575', fontSize: '14.5px', lineHeight: '1.8' }}>
          <li>Tunnels and underground environments</li>
          <li>Urban obstruction and signal blockage</li>
          <li>Interference</li>
          <li>Poor satellite visibility</li>
          <li>Challenging operating environments</li>
        </ul>
        <p style={{ marginTop: '16px', fontWeight: '600', color: '#0B1220' }}>
          During these outages, the vehicle continues moving even though the absolute position update disappears.
        </p>
      </div>

      <div className="content-card">
        <h3>THE CHALLENGE</h3>
        <p>
          Without GNSS updates, small errors in speed, acceleration, heading, and gyroscope measurements accumulate over time.
        </p>
        <p style={{ marginTop: '8px', fontWeight: '700', color: '#DC2626' }}>
          That accumulation becomes positional drift.
        </p>
      </div>

      <div className="content-card" style={{ borderLeft: '4px solid #2563EB' }}>
        <h3>SIH REQUIREMENT</h3>
        <p>The system must keep cumulative positional drift within:</p>
        <div className="val-large val-blue" style={{ fontSize: '28px', margin: '10px 0' }}>
          ≤ 10% OF DISTANCE TRAVELLED
        </div>
        <p style={{ fontSize: '14px', color: '#5B6575' }}>
          For the evaluated trajectory: 1162.5 m travelled → maximum allowable drift = 116.25 m
        </p>
        <div style={{ marginTop: '12px', padding: '12px 16px', background: '#EEF2F7', borderRadius: '8px', display: 'inline-block' }}>
          <span style={{ fontSize: '13px', fontWeight: '700', color: '#059669' }}>
            GHOST’s demonstrated final error: 27.95 m (2.40% Drift)
          </span>
        </div>
      </div>
    </div>
  );
}

// 03. Approach Section (The Idea)
export function ApproachSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">03 · THE IDEA</span>
        <h2 className="section-title">TURN MOTION INTO POSITION.</h2>
        <p className="section-desc">
          Instead of depending on GNSS continuously, GHOST estimates vehicle motion using sensors that remain available onboard.
        </p>
      </div>

      <div className="vertical-flow">
        <div className="flow-step-card">
          <div className="step-tag">INPUT</div>
          <h4>IMU</h4>
          <p>Accelerometer + gyroscope measurements</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card accent-violet">
          <div className="step-tag tag-violet">UNDERSTAND MOTION — AI</div>
          <h4>1D-CNN</h4>
          <p>Learns vehicle speed from temporal IMU patterns</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card accent-violet">
          <div className="step-tag tag-violet">CORRECT SPEED — CALIBRATION</div>
          <h4>Isotonic Calibration</h4>
          <p>Corrects systematic speed-dependent prediction bias</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card">
          <div className="step-tag">FUSE EVERYTHING</div>
          <h4>5-State EKF</h4>
          <p>Combines acceleration, heading, calibrated speed and motion history</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card">
          <div className="step-tag">USE THE ROAD</div>
          <h4>OSM Road Heading</h4>
          <p>Provides a geometric heading constraint</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card highlight-green">
          <div className="step-tag tag-green">OUTPUT</div>
          <h4 style={{ color: '#059669' }}>Continuous Position Estimate</h4>
          <p style={{ color: '#059669' }}>A GNSS-free position trajectory during the outage</p>
        </div>
      </div>
    </div>
  );
}

// 04. How It Works Section
export function HowItWorksSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">04 · HOW GHOST WORKS</span>
        <h2 className="section-title">A HYBRID SYSTEM — NOT JUST AN AI MODEL</h2>
        <p className="section-desc">
          GHOST is built as a chain of complementary components.
        </p>
      </div>

      <div className="grid-2-cols">
        <div className="content-card">
          <span className="step-idx-pill">01</span>
          <h3>MOTION SENSING</h3>
          <p>The onboard IMU records vehicle motion at 10 Hz. The system uses acceleration and yaw-rate information to understand how the vehicle is moving.</p>
        </div>

        <div className="content-card">
          <span className="step-idx-pill">02</span>
          <h3>TEMPORAL FEATURE EXTRACTION</h3>
          <p>A 30-step sliding window provides the neural network with recent motion history rather than relying on a single sensor reading.</p>
        </div>

        <div className="content-card accent-card-violet">
          <span className="step-idx-pill tag-violet">03 — AI</span>
          <h3>AI SPEED ESTIMATION</h3>
          <p>A 1D-Convolutional Neural Network estimates instantaneous vehicle speed from IMU motion dynamics.</p>
        </div>

        <div className="content-card accent-card-violet">
          <span className="step-idx-pill tag-violet">04 — CALIBRATION</span>
          <h3>SPEED CALIBRATION</h3>
          <p>The raw CNN prediction contains systematic speed-dependent bias. A leakage-free Isotonic Regression calibrator maps raw predictions to corrected speed estimates.</p>
        </div>

        <div className="content-card">
          <span className="step-idx-pill">05</span>
          <h3>SENSOR FUSION</h3>
          <p>The calibrated speed is supplied to a 5-State Extended Kalman Filter. The state is X = [x, y, v, θ, b<sup>gyro</sup>]<sup>T</sup> where x, y — local position, v — vehicle speed, θ — heading, b<sup>gyro</sup> — gyroscope bias.</p>
        </div>

        <div className="content-card">
          <span className="step-idx-pill">06</span>
          <h3>ROAD CONSTRAINT</h3>
          <p>OpenStreetMap road geometry provides a road-heading constraint. This helps keep the estimated heading consistent with the road network.</p>
        </div>
      </div>

      <div className="content-card highlight-green-box" style={{ marginTop: '12px' }}>
        <span className="step-idx-pill tag-green">07</span>
        <h3 style={{ color: '#059669' }}>POSITION OUTPUT</h3>
        <p style={{ color: '#0B1220' }}>The EKF continuously propagates the vehicle state throughout the GNSS outage.</p>
      </div>
    </div>
  );
}

// 05. Architecture Section
export function ArchitectureSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">05 · SYSTEM ARCHITECTURE</span>
        <h2 className="section-title">FROM SENSOR TO POSITION</h2>
      </div>

      <div className="content-card">
        <table className="tech-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Component</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Sensor</strong></td>
              <td>IMU</td>
              <td>10 Hz acceleration and yaw-rate measurements</td>
            </tr>
            <tr>
              <td><strong>Features</strong></td>
              <td>Temporal motion window</td>
              <td>Captures recent vehicle dynamics</td>
            </tr>
            <tr>
              <td style={{ color: '#7C3AED', fontWeight: '800' }}>AI</td>
              <td style={{ color: '#7C3AED' }}>1D-CNN</td>
              <td>Estimates vehicle speed</td>
            </tr>
            <tr>
              <td style={{ color: '#7C3AED', fontWeight: '800' }}>Calibration</td>
              <td style={{ color: '#7C3AED' }}>Isotonic Regression</td>
              <td>Corrects speed-dependent CNN bias</td>
            </tr>
            <tr>
              <td><strong>Fusion</strong></td>
              <td>5-State EKF</td>
              <td>Estimates continuous vehicle state</td>
            </tr>
            <tr>
              <td><strong>Map</strong></td>
              <td>OpenStreetMap</td>
              <td>Provides road-heading constraint</td>
            </tr>
            <tr className="row-winner">
              <td style={{ color: '#059669', fontWeight: '800' }}>Output</td>
              <td style={{ color: '#059669', fontWeight: '800' }}>Position Track</td>
              <td style={{ color: '#059669' }}>Produces GNSS-free positioning estimate</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="content-card">
        <h3>DESIGN PRINCIPLE</h3>
        <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '20px', color: '#5B6575', fontSize: '14.5px', lineHeight: '1.8' }}>
          <li><strong>AI</strong> estimates what the sensors cannot directly measure.</li>
          <li>The <strong>EKF</strong> decides how all measurements fit together.</li>
          <li>The <strong>road network</strong> provides an additional geometric constraint.</li>
        </ul>
      </div>
    </div>
  );
}

// 06. ML Experiments Section
export function MLExperimentsSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">06 · ML EXPERIMENTS</span>
        <h2 className="section-title">FROM 18.72% → 4.54% → 2.40%</h2>
        <p className="section-desc">
          The final result was not obtained by simply training a larger model. The major improvement came from identifying and correcting the dominant source of error: speed estimation bias.
        </p>
      </div>

      <div className="exp-card">
        <div className="exp-info">
          <h4>EXPERIMENT 0 — RAW CNN</h4>
          <p>The original CNN systematically underestimated higher-speed motion. That caused the EKF to estimate less travelled distance than the vehicle actually covered.</p>
        </div>
        <div className="exp-badge-group">
          <span className="drift-val val-red">18.72%</span>
          <span className="badge-status badge-fail">FAIL (ABOVE SIH LIMIT)</span>
        </div>
      </div>

      <div className="exp-card">
        <div className="exp-info">
          <h4>EXPERIMENT 1 — LINEAR CALIBRATION</h4>
          <p>A leakage-free linear calibration was applied to the CNN output. The systematic speed bias was substantially reduced.</p>
        </div>
        <div className="exp-badge-group">
          <span className="drift-val val-blue">4.54%</span>
          <span className="badge-status badge-pass">PASS</span>
        </div>
      </div>

      <div className="exp-card">
        <div className="exp-info">
          <h4>EXPERIMENT 2 — GYRO BIAS DIAGNOSIS</h4>
          <p>This experiment investigated whether heading error was the dominant remaining source of drift. The analysis showed that heading and cross-track error contributed to final position error, but speed estimation remained the dominant issue in the baseline.</p>
        </div>
        <div className="exp-badge-group">
          <span className="badge-status badge-neutral">DIAGNOSTIC STEP</span>
        </div>
      </div>

      <div className="exp-card winner">
        <div className="exp-info">
          <h4 style={{ color: '#059669' }}>EXPERIMENT 3 — ISOTONIC CALIBRATION</h4>
          <p style={{ color: '#059669' }}>The relationship between CNN predictions and actual speed was not purely linear. Isotonic Regression provided a monotonic nonlinear mapping that better corrected speed-dependent bias on the outage trajectory.</p>
        </div>
        <div className="exp-badge-group">
          <span className="drift-val val-green">2.40%</span>
          <span className="badge-status badge-winner">BEST DEMONSTRATED RESULT</span>
        </div>
      </div>

      <div className="exp-card">
        <div className="exp-info">
          <h4>EXPERIMENT 4A — SOFT REGIME GATE</h4>
          <p>A soft gate was introduced to blend linear and isotonic calibration according to predicted speed. It improved chronological speed-estimation robustness relative to linear calibration, but its outage positioning result was worse than the isotonic model.</p>
        </div>
        <div className="exp-badge-group">
          <span className="drift-val val-amber">5.60%</span>
          <span className="badge-status badge-pass">PASS (NOT THE WINNER)</span>
        </div>
      </div>
    </div>
  );
}

// 07. Results Section
export function ResultsSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">07 · RESULTS</span>
        <h2 className="section-title">THE NUMBERS</h2>
      </div>

      <div className="content-card">
        <table className="tech-table">
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Validation MAE</th>
              <th>Test MAE</th>
              <th>Outage MAE</th>
              <th>Final Error</th>
              <th>Drift %</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Raw CNN</td>
              <td>13.03 km/h</td>
              <td>30.59 km/h</td>
              <td>15.89 km/h</td>
              <td>217.67 m</td>
              <td style={{ color: '#DC2626', fontWeight: '800' }}>18.72%</td>
            </tr>
            <tr>
              <td>Linear Calibration</td>
              <td>20.71 km/h</td>
              <td>40.67 km/h</td>
              <td>8.52 km/h</td>
              <td>52.80 m</td>
              <td style={{ color: '#2563EB', fontWeight: '800' }}>4.54%</td>
            </tr>
            <tr>
              <td>Soft Gate</td>
              <td>19.61 km/h</td>
              <td>36.95 km/h</td>
              <td>7.67 km/h</td>
              <td>65.09 m</td>
              <td style={{ color: '#D97706', fontWeight: '800' }}>5.60%</td>
            </tr>
            <tr className="row-winner">
              <td style={{ color: '#059669', fontWeight: '800' }}>Isotonic Calibration</td>
              <td>19.23 km/h</td>
              <td>32.74 km/h</td>
              <td>7.17 km/h</td>
              <td style={{ color: '#059669', fontWeight: '800' }}>27.95 m</td>
              <td style={{ color: '#059669', fontWeight: '900', fontSize: '16px' }}>2.40%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="grid-2-cols">
        <div className="content-card highlight-green-box">
          <h3>BEST DEMONSTRATED RESULT</h3>
          <div className="val-large val-green" style={{ fontSize: '32px', margin: '10px 0' }}>
            27.95 m final position error
          </div>
          <p style={{ color: '#5B6575' }}>
            over 1162.5 m of travel during 60 seconds without GNSS
          </p>
        </div>

        <div className="content-card">
          <h3>DRIFT CALCULATION</h3>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '16px', fontWeight: '700', color: '#0B1220', marginTop: '10px' }}>
            27.95 / 1162.5 × 100 = 2.40%
          </div>
          <div style={{ marginTop: '14px', fontSize: '13px', color: '#5B6575' }}>
            <div>SIH limit: <strong>10.00%</strong></div>
            <div>GHOST: <strong style={{ color: '#059669' }}>2.40%</strong></div>
            <div style={{ marginTop: '6px', fontWeight: '700', color: '#059669' }}>
              MARGIN: 7.60 percentage points below the SIH limit
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// 08. Error Analysis Section
export function ErrorAnalysisSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">08 · ERROR ANALYSIS</span>
        <h2 className="section-title">WHY DID THE DRIFT DROP?</h2>
        <p className="section-desc">
          The original failure was primarily a speed-estimation problem.
        </p>
      </div>

      <div className="content-card">
        <h3>RAW CNN BOTTLENECK</h3>
        <p>During the evaluated blackout:</p>
        <div className="grid-3-cols" style={{ margin: '16px 0' }}>
          <div className="metric-card accent-blue">
            <span className="val-large" style={{ fontSize: '22px' }}>69.75 km/h</span>
            <span className="lbl-card">Ground-truth mean speed</span>
          </div>
          <div className="metric-card accent-red">
            <span className="val-large val-red" style={{ fontSize: '22px' }}>54.85 km/h</span>
            <span className="lbl-card">Raw CNN mean speed</span>
          </div>
          <div className="metric-card accent-red">
            <span className="val-large val-red" style={{ fontSize: '22px' }}>-14.90 km/h</span>
            <span className="lbl-card">Average signed error</span>
          </div>
        </div>
        <p>
          The CNN compressed higher-speed predictions toward lower values. That created a travelled-distance deficit, which accumulated primarily as longitudinal position error.
        </p>
        <div style={{ marginTop: '14px', padding: '12px 16px', background: '#FEE2E2', borderRadius: '8px', fontSize: '13px', color: '#DC2626', fontWeight: '700' }}>
          ORIGINAL ERROR: Along-track: −215.95 m | Cross-track: −27.32 m | Final error: 217.67 m | Drift: 18.72%
        </div>
      </div>

      <div className="content-card highlight-green-box">
        <h3>AFTER ISOTONIC CALIBRATION</h3>
        <p style={{ color: '#0B1220' }}>
          The calibrated speed estimate substantially reduced the longitudinal error.
        </p>
        <div className="grid-3-cols" style={{ margin: '16px 0' }}>
          <div className="metric-card accent-green">
            <span className="val-large val-green" style={{ fontSize: '22px' }}>-1.16 m</span>
            <span className="lbl-card">Along-track error</span>
          </div>
          <div className="metric-card accent-yellow">
            <span className="val-large val-amber" style={{ fontSize: '22px' }}>-27.93 m</span>
            <span className="lbl-card">Cross-track error</span>
          </div>
          <div className="metric-card accent-green">
            <span className="val-large val-green" style={{ fontSize: '22px' }}>27.95 m</span>
            <span className="lbl-card">Total vector error</span>
          </div>
        </div>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: '#059669', fontWeight: '700' }}>
          The vector relationship is: √((-1.16)² + (-27.93)²) = 27.95 m
        </div>
      </div>

      <div className="content-card">
        <h3>KEY INSIGHT</h3>
        <p>
          The largest improvement came from correcting the longitudinal speed error. The remaining demonstrated error was primarily cross-track rather than longitudinal.
        </p>
      </div>
    </div>
  );
}

// 09. Methodology Section
export function MethodologySection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">09 · DATA & METHODOLOGY</span>
        <h2 className="section-title">DATA INTEGRITY MATTERS.</h2>
      </div>

      <div className="grid-4-cols">
        <div className="metric-card accent-blue">
          <span className="val-large val-blue">50,000</span>
          <span className="lbl-card">Original records</span>
        </div>
        <div className="metric-card accent-blue">
          <span className="val-large val-blue">50</span>
          <span className="lbl-card">Recorded tracks</span>
        </div>
        <div className="metric-card accent-green">
          <span className="val-large val-green">49,950</span>
          <span className="lbl-card">ML-ready records</span>
        </div>
        <div className="metric-card accent-violet">
          <span className="val-large val-violet">15</span>
          <span className="lbl-card">ML features used</span>
        </div>
      </div>

      <div className="content-card">
        <h3>WHY 49,950?</h3>
        <p>
          Each track loses its first observation because previous-history features cannot be calculated without a preceding measurement.
        </p>
        <p style={{ marginTop: '8px', fontWeight: '700', fontFamily: 'JetBrains Mono, monospace', color: '#0B1220' }}>
          Therefore: 50,000 − 50 = 49,950 ML-ready records
        </p>
      </div>

      <div className="content-card">
        <h3>DATA SPLITS</h3>
        <div style={{ padding: '12px 16px', background: '#EEF2F7', borderRadius: '8px', fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', color: '#0B1220', fontWeight: '700' }}>
          Training → Validation → Test → Outage Evaluation
        </div>
        <p style={{ marginTop: '12px' }}>
          The outage trajectory remains untouched during model fitting and selection.
        </p>
      </div>

      <div className="content-card highlight-green-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3>LEAKAGE CONTROL</h3>
          <span className="badge-status badge-winner">LEAKAGE AUDIT: PASS ✓</span>
        </div>
        <p style={{ marginTop: '8px', color: '#0B1220' }}>
          Calibration models were fitted using clean training data only. The outage interval was excluded from fitting. Validation and test data were not used to fit the final calibration.
        </p>
      </div>
    </div>
  );
}

// 10. Final Model Section
export function FinalModelSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">10 · FINAL MODEL</span>
        <h2 className="section-title">THE FINAL GHOST STACK</h2>
      </div>

      <div className="vertical-flow">
        <div className="flow-step-card accent-violet">
          <div className="step-tag tag-violet">AI</div>
          <h4>1D-CNN</h4>
          <p>Temporal speed estimation — Learns speed-related patterns from 30-step IMU windows.</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card accent-violet">
          <div className="step-tag tag-violet">CALIBRATION</div>
          <h4>Isotonic Regression</h4>
          <p>Nonlinear speed calibration — Corrects systematic prediction bias while preserving monotonicity.</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card">
          <div className="step-tag">FUSION</div>
          <h4>5-State EKF</h4>
          <p>Sensor fusion — Maintains the continuous vehicle state during GNSS loss.</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card">
          <div className="step-tag">MAP</div>
          <h4>OSM Road Heading</h4>
          <p>Geometric constraint — Provides road-network heading information.</p>
        </div>

        <div className="flow-connector">↓</div>

        <div className="flow-step-card highlight-green">
          <div className="step-tag tag-green">RESULT</div>
          <h4 style={{ color: '#059669' }}>Position Continuity</h4>
        </div>
      </div>

      <div className="content-card" style={{ marginTop: '24px', borderLeft: '4px solid #2563EB' }}>
        <h3>The final demonstrated configuration is:</h3>
        <p style={{ fontWeight: '700', fontSize: '16px', color: '#0B1220', marginTop: '6px' }}>
          1D-CNN Speed Estimator + Leakage-Free Isotonic Speed Calibration + 5-State EKF + OSM Road-Heading Constraint
        </p>
      </div>

      <div className="content-card">
        <h3>WHY THIS CONFIGURATION?</h3>
        <p>Because each component addresses a different part of the problem:</p>
        <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '20px', color: '#5B6575', fontSize: '14.5px', lineHeight: '1.8' }}>
          <li><strong style={{ color: '#7C3AED' }}>CNN</strong> → estimates speed</li>
          <li><strong style={{ color: '#7C3AED' }}>Calibration</strong> → corrects speed bias</li>
          <li><strong>EKF</strong> → fuses motion information</li>
          <li><strong>OSM</strong> → constrains heading</li>
          <li><strong style={{ color: '#059669' }}>Together</strong> → maintain positioning continuity</li>
        </ul>
      </div>
    </div>
  );
}

// 11. Limitations Section
export function LimitationsSection() {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">11 · LIMITATIONS & HONEST EVALUATION</span>
        <h2 className="section-title">WHAT THE RESULT DOES — AND DOES NOT — CLAIM</h2>
      </div>

      <div className="content-card">
        <p>
          The 2.40% drift is the best demonstrated result on the evaluated 60-second GNSS outage trajectory. It demonstrates that the proposed architecture can satisfy the SIH drift benchmark on that evaluation.
        </p>
        <p style={{ marginTop: '10px', fontWeight: '600', color: '#DC2626' }}>
          However, the result should not be interpreted as a universal accuracy guarantee.
        </p>
      </div>

      <div className="content-card">
        <h3>IMPORTANT LIMITATIONS</h3>
        <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '20px', color: '#5B6575', fontSize: '14.5px', lineHeight: '1.8' }}>
          <li>Evaluation is based on the demonstrated outage trajectory.</li>
          <li>Calibration performance is sensitive to speed-distribution differences.</li>
          <li>Chronological test performance differs from outage performance.</li>
          <li>The system provides positioning continuity rather than permanent GNSS replacement.</li>
          <li>Broader validation across different roads, speeds, vehicles, and outage conditions is required before making general deployment claims.</li>
        </ul>
      </div>

      <div className="content-card highlight-green-box">
        <h3>THE SCIENTIFIC TAKEAWAY</h3>
        <p style={{ color: '#0B1220', fontWeight: '600' }}>
          The architecture works on the demonstrated scenario, and the experiments reveal why it works.
        </p>
      </div>
    </div>
  );
}

// 12. Tech Stack & Prototype Section
export function TechStackSection({ onSwitchToPrototype }) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <span className="section-tag">12 · TECHNOLOGY & PROTOTYPE</span>
        <h2 className="section-title">BUILT TO BE DEMONSTRATED</h2>
      </div>

      <div className="grid-2-cols">
        <div className="content-card accent-card-violet">
          <span className="step-tag tag-violet">AI / MACHINE LEARNING</span>
          <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '18px', color: '#5B6575', fontSize: '14px', lineHeight: '1.7' }}>
            <li>Python</li>
            <li>PyTorch</li>
            <li>Scikit-Learn</li>
            <li>1D-CNN</li>
            <li>Isotonic Regression</li>
          </ul>
        </div>

        <div className="content-card">
          <span className="step-tag">SENSOR FUSION</span>
          <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '18px', color: '#5B6575', fontSize: '14px', lineHeight: '1.7' }}>
            <li>NumPy</li>
            <li>SciPy</li>
            <li>Extended Kalman Filter</li>
          </ul>
        </div>

        <div className="content-card">
          <span className="step-tag">MAP DATA</span>
          <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '18px', color: '#5B6575', fontSize: '14px', lineHeight: '1.7' }}>
            <li>OpenStreetMap</li>
            <li>Road geometry</li>
            <li>Road-heading constraints</li>
          </ul>
        </div>

        <div className="content-card">
          <span className="step-tag">VISUALIZATION</span>
          <ul className="clean-list" style={{ marginTop: '10px', paddingLeft: '18px', color: '#5B6575', fontSize: '14px', lineHeight: '1.7' }}>
            <li>React 18</li>
            <li>Three.js</li>
            <li>React Three Fiber</li>
            <li>Vite</li>
            <li>Lucide Icons</li>
          </ul>
        </div>
      </div>

      <div className="hero-box" style={{ textAlign: 'center', marginTop: '32px' }}>
        <h3 style={{ fontSize: '24px', fontWeight: '900', color: '#0B1220' }}>EXPERIENCE GHOST</h3>
        <p style={{ margin: '12px auto 24px auto', maxWidth: '650px', color: '#5B6575', fontSize: '15px' }}>
          The technical pipeline is backed by an interactive two-vehicle 3D demonstration. The prototype visualizes the navigation scenario and shows the system continuing to estimate vehicle position during GNSS loss.
        </p>
        <button className="btn-cta" onClick={onSwitchToPrototype}>
          LAUNCH LIVE PROTOTYPE →
        </button>
      </div>

      <div className="content-card" style={{ marginTop: '32px', textAlign: 'center', background: '#EEF2F7', border: '1px solid #DCE2EA' }}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', fontWeight: '800', color: '#2563EB', letterSpacing: '1.5px' }}>
          FINAL PROJECT STATEMENT
        </div>
        <div style={{ fontSize: '22px', fontWeight: '900', color: '#0B1220', margin: '8px 0' }}>
          GHOST: WHEN GNSS DISAPPEARS, POSITIONING DOESN’T HAVE TO.
        </div>
        <p style={{ fontSize: '14px', color: '#5B6575', maxWidth: '750px', margin: '0 auto' }}>
          GHOST combines AI-based speed estimation, leakage-free calibration, inertial sensor fusion, and road-network constraints to maintain vehicle positioning continuity during temporary GNSS outages.
        </p>

        <div className="grid-5-cols" style={{ marginTop: '20px' }}>
          <div className="metric-card accent-green">
            <span className="val-large val-green" style={{ fontSize: '20px' }}>2.40%</span>
            <span className="lbl-card">Positional drift</span>
          </div>
          <div className="metric-card accent-green">
            <span className="val-large val-green" style={{ fontSize: '20px' }}>27.95 m</span>
            <span className="lbl-card">Final error</span>
          </div>
          <div className="metric-card accent-blue">
            <span className="val-large val-blue" style={{ fontSize: '20px' }}>60 s</span>
            <span className="lbl-card">GNSS outage</span>
          </div>
          <div className="metric-card accent-blue">
            <span className="val-large val-blue" style={{ fontSize: '20px' }}>1162.5 m</span>
            <span className="lbl-card">Travelled</span>
          </div>
          <div className="metric-card accent-blue">
            <span className="val-large val-blue" style={{ fontSize: '20px' }}>≤10%</span>
            <span className="lbl-card">SIH benchmark</span>
          </div>
        </div>

        <div style={{ marginTop: '16px', fontSize: '12px', color: '#5B6575', fontWeight: '600' }}>
          SMART INDIA HACKATHON 2026 · SIH26168 · ISRO — AI-ML based Intelligent Dead Reckoning system for seamless navigation
        </div>
      </div>
    </div>
  );
}

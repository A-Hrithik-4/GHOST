import React from 'react';

const SIDEBAR_GROUPS = [
  {
    groupTitle: null, // Level 1: Core Overview
    items: [
      { id: 'overview', num: '01', title: 'PROJECT OVERVIEW', isAi: false },
      { id: 'problem', num: '02', title: 'PROBLEM STATEMENT', isAi: false },
      { id: 'objectives', num: '03', title: 'OBJECTIVES & REQUIREMENTS', isAi: false },
      { id: 'solution', num: '04', title: 'PROPOSED SOLUTION', isAi: false },
      { id: 'architecture', num: '05', title: 'SYSTEM ARCHITECTURE', isAi: false },
      { id: 'workflow', num: '06', title: 'END-TO-END WORKFLOW', isAi: false }
    ]
  },
  {
    groupTitle: 'SENSORS & NAVIGATION',
    items: [
      { id: 'sensor_inputs', num: '07', title: 'SENSOR INPUTS', isAi: false },
      { id: 'data_processing', num: '08', title: 'DATA PROCESSING', isAi: false },
      { id: 'feature_engineering', num: '09', title: 'FEATURE ENGINEERING', isAi: false },
      { id: 'dead_reckoning', num: '10', title: 'DEAD RECKONING', isAi: false },
      { id: 'ekf_navigation', num: '11', title: 'EKF NAVIGATION', isAi: false },
      { id: 'road_constraint', num: '12', title: 'ROAD-HEADING CONSTRAINT', isAi: false }
    ]
  },
  {
    groupTitle: 'MACHINE LEARNING',
    items: [
      { id: 'ml_pipeline', num: '13', title: 'ML PIPELINE', isAi: true },
      { id: 'ml_dataset', num: '14', title: 'DATASET FOR ML', isAi: true },
      { id: 'ml_estimator', num: '15', title: '1D-CNN SPEED ESTIMATOR', isAi: true },
      { id: 'speed_calibration', num: '16', title: 'SPEED CALIBRATION', isAi: true },
      { id: 'leakage_prevention', num: '17', title: 'LEAKAGE PREVENTION', isAi: true }
    ]
  },
  {
    groupTitle: 'EXPERIMENTATION',
    items: [
      { id: 'exp_setup', num: '18', title: 'EXPERIMENT SETUP', isAi: false },
      { id: 'model_comparison', num: '19', title: 'MODEL COMPARISON', isAi: false },
      { id: 'final_model', num: '20', title: 'FINAL MODEL', isAi: true },
      { id: 'results_benchmark', num: '21', title: 'RESULTS & BENCHMARK', isAi: false },
      { id: 'error_analysis', num: '22', title: 'ERROR ANALYSIS', isAi: false }
    ]
  },
  {
    groupTitle: 'IMPLEMENTATION',
    items: [
      { id: 'project_structure', num: '23', title: 'PROJECT STRUCTURE', isAi: false },
      { id: 'algo_impl', num: '24', title: 'ALGORITHM IMPLEMENTATION', isAi: false },
      { id: 'code_explorer', num: '25', title: 'CODE EXPLORER', isAi: false },
      { id: 'tech_stack', num: '26', title: 'TECHNOLOGY STACK', isAi: false }
    ]
  },
  {
    groupTitle: 'EVALUATION',
    items: [
      { id: 'validation_audit', num: '27', title: 'VALIDATION & AUDIT', isAi: false },
      { id: 'limitations', num: '28', title: 'LIMITATIONS', isAi: false },
      { id: 'future_scope', num: '29', title: 'FUTURE SCOPE', isAi: false }
    ]
  },
  {
    groupTitle: 'DEMONSTRATION',
    items: [
      { id: 'prototype', num: '30', title: 'INTERACTIVE PROTOTYPE', isAi: false },
      { id: 'team', num: '31', title: 'TEAM', isAi: false }
    ]
  }
];

export default function ProjectSidebar({ activeSection, onSelectSection }) {
  return (
    <aside className="project-sidebar">
      <div className="sidebar-header-box" style={{ padding: '0 16px 12px 16px', borderBottom: '1.5px solid var(--border-light)' }}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px', fontWeight: '900', color: 'var(--graphite-dark)', letterSpacing: '1px' }}>
          GHOST
        </div>
        <div style={{ fontSize: '9.5px', fontWeight: '800', color: 'var(--amber-primary)', letterSpacing: '0.5px', marginTop: '2px', textTransform: 'uppercase' }}>
          GNSS-Free Hybrid Sensor Tracker
        </div>
        <div style={{ fontSize: '9px', fontWeight: '700', color: 'var(--text-muted)', marginTop: '1px' }}>
          SIH 26168 · ISRO / DOS
        </div>
      </div>

      <div className="sidebar-scroll-content" style={{ padding: '10px 0' }}>
        {SIDEBAR_GROUPS.map((group, gIdx) => (
          <div key={gIdx} className="sidebar-group" style={{ marginBottom: '12px' }}>
            {group.groupTitle && (
              <div className="sidebar-title" style={{ padding: '8px 16px 4px 16px', fontSize: '9.5px', color: 'var(--text-muted)' }}>
                {group.groupTitle}
              </div>
            )}
            {group.items.map((sec) => {
              const isActive = activeSection === sec.id;
              return (
                <div
                  key={sec.id}
                  className={`sidebar-nav-item ${isActive ? 'active' : ''} ${isActive && sec.isAi ? 'ai-active' : ''}`}
                  onClick={() => onSelectSection(sec.id)}
                >
                  <span className="nav-idx">{sec.num}</span>
                  <span>{sec.title}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </aside>
  );
}

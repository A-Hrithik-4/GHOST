import React from 'react';

const SECTIONS = [
  { id: 'overview', num: '01', title: 'MISSION OVERVIEW', isAi: false },
  { id: 'problem', num: '02', title: 'PROBLEM STATEMENT', isAi: false },
  { id: 'why_ghost', num: '03', title: 'WHY GHOST', isAi: false },
  { id: 'architecture', num: '04', title: 'SYSTEM ARCHITECTURE', isAi: false },
  { id: 'math', num: '05', title: 'NAVIGATION MATHEMATICS', isAi: false },
  { id: 'sensor_pipeline', num: '06', title: 'SENSOR PIPELINE', isAi: false },
  { id: 'ml_pipeline', num: '07', title: 'ML PIPELINE', isAi: true },
  { id: 'ml_algorithm', num: '08', title: 'ML ALGORITHM', isAi: true },
  { id: 'speed_calibration', num: '09', title: 'SPEED CALIBRATION', isAi: true },
  { id: 'ekf_navigation', num: '10', title: 'EKF NAVIGATION', isAi: false },
  { id: 'road_constraint', num: '11', title: 'ROAD-HEADING CONSTRAINT', isAi: false },
  { id: 'experiments', num: '12', title: 'EXPERIMENTS', isAi: false },
  { id: 'final_model', num: '13', title: 'FINAL MODEL', isAi: true },
  { id: 'results', num: '14', title: 'RESULTS & BENCHMARK', isAi: false },
  { id: 'error_analysis', num: '15', title: 'ERROR ANALYSIS', isAi: false },
  { id: 'dataset', num: '16', title: 'DATASET & METHODOLOGY', isAi: false },
  { id: 'code', num: '17', title: 'IMPLEMENTATION / CODE', isAi: false },
  { id: 'tech_stack', num: '18', title: 'TECHNOLOGY STACK', isAi: false },
  { id: 'limitations', num: '19', title: 'LIMITATIONS', isAi: false },
  { id: 'future_scope', num: '20', title: 'FUTURE SCOPE', isAi: false },
  { id: 'prototype', num: '21', title: 'PROTOTYPE', isAi: false },
  { id: 'team', num: '22', title: 'TEAM', isAi: false }
];

export default function ProjectSidebar({ activeSection, onSelectSection }) {
  return (
    <aside className="project-sidebar">
      <div className="sidebar-title">GHOST TECHNICAL SPECIFICATION</div>
      {SECTIONS.map((sec) => {
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
    </aside>
  );
}

import React from 'react';

/**
 * ProjectSidebar.jsx
 * 
 * Collapsible left navigation sidebar with exactly 12 sections.
 */

export const SECTIONS = [
  { id: 'overview', title: 'Overview' },
  { id: 'problem', title: 'Problem' },
  { id: 'approach', title: 'Approach' },
  { id: 'how_it_works', title: 'How It Works' },
  { id: 'architecture', title: 'Architecture' },
  { id: 'experiments', title: 'Experiments' },
  { id: 'results', title: 'Results' },
  { id: 'error_analysis', title: 'Error Analysis' },
  { id: 'methodology', title: 'Methodology' },
  { id: 'final_model', title: 'Final Model' },
  { id: 'limitations', title: 'Limitations' },
  { id: 'tech_stack', title: 'Tech Stack & Prototype' }
];

export default function ProjectSidebar({ activeSection, onSelectSection }) {
  return (
    <aside className="project-sidebar">
      <div className="sidebar-title">GHOST PROJECT</div>
      
      <nav>
        {SECTIONS.map((sec, idx) => (
          <div
            key={sec.id}
            className={`sidebar-nav-item ${activeSection === sec.id ? 'active' : ''}`}
            onClick={() => onSelectSection(sec.id)}
          >
            <span className="nav-idx">{(idx + 1).toString().padStart(2, '0')}</span>
            <span>{sec.title}</span>
          </div>
        ))}
      </nav>
    </aside>
  );
}

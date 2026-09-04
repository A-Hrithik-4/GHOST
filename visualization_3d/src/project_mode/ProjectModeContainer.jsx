import React, { useState, useRef } from 'react';
import ProjectNavbar from './ProjectNavbar';
import ProjectSidebar from './ProjectSidebar';
import {
  OverviewSection,
  ProblemSection,
  ApproachSection,
  HowItWorksSection,
  ArchitectureSection,
  MLExperimentsSection,
  ResultsSection,
  ErrorAnalysisSection,
  MethodologySection,
  FinalModelSection,
  LimitationsSection,
  TechStackSection
} from './ProjectSections';
import '../styles/project.css';

/**
 * ProjectModeContainer.jsx
 * 
 * Main container for GHOST Project Mode / Technical Dashboard.
 */

export default function ProjectModeContainer({ activeMode, onSwitchMode }) {
  const [activeSection, setActiveSection] = useState('overview');
  const contentRef = useRef(null);

  const handleSelectSection = (secId) => {
    setActiveSection(secId);
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  };

  // Switch back to prototype handler
  const handleSwitchToPrototype = () => {
    onSwitchMode('prototype');
  };

  const renderSectionContent = () => {
    switch (activeSection) {
      case 'overview':
        return <OverviewSection onSwitchToPrototype={handleSwitchToPrototype} />;
      case 'problem':
        return <ProblemSection />;
      case 'approach':
        return <ApproachSection />;
      case 'how_it_works':
        return <HowItWorksSection />;
      case 'architecture':
        return <ArchitectureSection />;
      case 'experiments':
        return <MLExperimentsSection />;
      case 'results':
        return <ResultsSection />;
      case 'error_analysis':
        return <ErrorAnalysisSection />;
      case 'methodology':
        return <MethodologySection />;
      case 'final_model':
        return <FinalModelSection />;
      case 'limitations':
        return <LimitationsSection />;
      case 'tech_stack':
      case 'prototype':
        return <TechStackSection onSwitchToPrototype={handleSwitchToPrototype} />;
      default:
        return <OverviewSection onSwitchToPrototype={handleSwitchToPrototype} />;
    }
  };

  return (
    <div className="project-container">
      {/* Top Navbar with Mode Switcher */}
      <ProjectNavbar activeMode={activeMode} onSwitchMode={onSwitchMode} />

      {/* Main Body: Fixed Left Sidebar + Scrollable Technical Content */}
      <div className="project-body">
        <ProjectSidebar activeSection={activeSection} onSelectSection={handleSelectSection} />

        <main className="project-content" ref={contentRef}>
          {renderSectionContent()}
        </main>
      </div>
    </div>
  );
}

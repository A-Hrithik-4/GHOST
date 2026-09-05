import React, { useState, useRef } from 'react';
import ProjectNavbar from './ProjectNavbar';
import ProjectSidebar from './ProjectSidebar';
import {
  OverviewSection,
  ProblemObjectiveSection,
  SolutionSection,
  SystemArchitectureSection,
  DataFeatureEngineeringSection,
  MlModelSection,
  NavigationEkfSection,
  ExperimentsResultsSection,
  ImplementationCodeSection,
  LimitationsFutureScopeSection
} from './ProjectSections';
import '../styles/project.css';

export default function ProjectModeContainer({ activeMode, onSwitchMode }) {
  const [activeSection, setActiveSection] = useState('overview');
  const contentRef = useRef(null);

  const handleSelectSection = (secId) => {
    setActiveSection(secId);
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  };

  const renderSectionContent = () => {
    switch (activeSection) {
      case 'overview':
        return <OverviewSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'problem_objective':
        return <ProblemObjectiveSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'solution':
        return <SolutionSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'system_architecture':
        return <SystemArchitectureSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'data_feature_engineering':
        return <DataFeatureEngineeringSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'ml_model':
        return <MlModelSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'navigation_ekf':
        return <NavigationEkfSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'experiments_results':
        return <ExperimentsResultsSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'implementation_code':
        return <ImplementationCodeSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      case 'limitations_future_scope':
        return <LimitationsFutureScopeSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
      default:
        return <OverviewSection onSwitchToPrototype={() => onSwitchMode('prototype')} />;
    }
  };

  return (
    <div className="project-container">
      <ProjectNavbar activeMode={activeMode} onSwitchMode={onSwitchMode} />
      <div className="project-body">
        <ProjectSidebar 
          activeSection={activeSection} 
          onSelectSection={handleSelectSection} 
          onSwitchMode={onSwitchMode}
        />
        <main className="project-content" ref={contentRef}>
          {renderSectionContent()}
        </main>
      </div>
    </div>
  );
}

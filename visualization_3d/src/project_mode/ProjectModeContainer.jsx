import React, { useState, useRef } from 'react';
import ProjectNavbar from './ProjectNavbar';
import ProjectSidebar from './ProjectSidebar';
import {
  MissionOverviewSection,
  ProblemStatementSection,
  WhyGhostSection,
  SystemArchitectureSection,
  NavigationMathSection,
  SensorPipelineSection,
  MlPipelineSection,
  MlAlgorithmSection,
  SpeedCalibrationSection,
  EkfNavigationSection,
  RoadConstraintSection,
  ExperimentsSection,
  FinalModelSection,
  ResultsBenchmarkSection,
  ErrorAnalysisSection,
  DatasetMethodologySection,
  ImplementationCodeSection,
  TechnologyStackSection,
  LimitationsSection,
  FutureScopeSection,
  PrototypePageSection,
  TeamSection
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

  const handleSwitchToPrototype = () => {
    onSwitchMode('prototype');
  };

  const renderSectionContent = () => {
    switch (activeSection) {
      case 'overview':
        return <MissionOverviewSection onSwitchToPrototype={handleSwitchToPrototype} />;
      case 'problem':
        return <ProblemStatementSection />;
      case 'why_ghost':
        return <WhyGhostSection />;
      case 'architecture':
        return <SystemArchitectureSection />;
      case 'math':
        return <NavigationMathSection />;
      case 'sensor_pipeline':
        return <SensorPipelineSection />;
      case 'ml_pipeline':
        return <MlPipelineSection />;
      case 'ml_algorithm':
        return <MlAlgorithmSection />;
      case 'speed_calibration':
        return <SpeedCalibrationSection />;
      case 'ekf_navigation':
        return <EkfNavigationSection />;
      case 'road_constraint':
        return <RoadConstraintSection />;
      case 'experiments':
        return <ExperimentsSection />;
      case 'final_model':
        return <FinalModelSection />;
      case 'results':
        return <ResultsBenchmarkSection />;
      case 'error_analysis':
        return <ErrorAnalysisSection />;
      case 'dataset':
        return <DatasetMethodologySection />;
      case 'code':
        return <ImplementationCodeSection />;
      case 'tech_stack':
        return <TechnologyStackSection />;
      case 'limitations':
        return <LimitationsSection />;
      case 'future_scope':
        return <FutureScopeSection />;
      case 'prototype':
        return <PrototypePageSection onSwitchToPrototype={handleSwitchToPrototype} />;
      case 'team':
        return <TeamSection />;
      default:
        return <MissionOverviewSection onSwitchToPrototype={handleSwitchToPrototype} />;
    }
  };

  return (
    <div className="project-container">
      <ProjectNavbar activeMode={activeMode} onSwitchMode={onSwitchMode} />
      <div className="project-body">
        <ProjectSidebar activeSection={activeSection} onSelectSection={handleSelectSection} />
        <main className="project-content" ref={contentRef}>
          {renderSectionContent()}
        </main>
      </div>
    </div>
  );
}

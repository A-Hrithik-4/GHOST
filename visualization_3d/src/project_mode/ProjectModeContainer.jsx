import React, { useState, useRef } from 'react';
import ProjectNavbar from './ProjectNavbar';
import ProjectSidebar from './ProjectSidebar';
import {
  ProjectOverviewSection,
  ProblemStatementSection,
  ObjectivesRequirementsSection,
  ProposedSolutionSection,
  SystemArchitectureSection,
  EndToEndWorkflowSection,
  SensorInputsSection,
  DataProcessingSection,
  FeatureEngineeringSection,
  DeadReckoningSection,
  EkfNavigationSection,
  RoadHeadingConstraintSection,
  MlPipelineSection,
  DatasetForMlSection,
  CnnSpeedEstimatorSection,
  SpeedCalibrationSection,
  LeakagePreventionSection,
  ExperimentSetupSection,
  ModelComparisonSection,
  FinalModelSection,
  ResultsBenchmarkSection,
  ErrorAnalysisSection,
  ProjectStructureSection,
  AlgorithmImplementationSection,
  CodeExplorerSection,
  TechnologyStackSection,
  ValidationAuditSection,
  LimitationsSection,
  FutureScopeSection,
  InteractivePrototypeSection,
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
        return <ProjectOverviewSection />;
      case 'problem':
        return <ProblemStatementSection />;
      case 'objectives':
        return <ObjectivesRequirementsSection />;
      case 'solution':
        return <ProposedSolutionSection />;
      case 'architecture':
        return <SystemArchitectureSection />;
      case 'workflow':
        return <EndToEndWorkflowSection />;
      case 'sensor_inputs':
        return <SensorInputsSection />;
      case 'data_processing':
        return <DataProcessingSection />;
      case 'feature_engineering':
        return <FeatureEngineeringSection />;
      case 'dead_reckoning':
        return <DeadReckoningSection />;
      case 'ekf_navigation':
        return <EkfNavigationSection />;
      case 'road_constraint':
        return <RoadHeadingConstraintSection />;
      case 'ml_pipeline':
        return <MlPipelineSection />;
      case 'ml_dataset':
        return <DatasetForMlSection />;
      case 'ml_estimator':
        return <CnnSpeedEstimatorSection />;
      case 'speed_calibration':
        return <SpeedCalibrationSection />;
      case 'leakage_prevention':
        return <LeakagePreventionSection />;
      case 'exp_setup':
        return <ExperimentSetupSection />;
      case 'model_comparison':
        return <ModelComparisonSection />;
      case 'final_model':
        return <FinalModelSection />;
      case 'results_benchmark':
        return <ResultsBenchmarkSection />;
      case 'error_analysis':
        return <ErrorAnalysisSection />;
      case 'project_structure':
        return <ProjectStructureSection />;
      case 'algo_impl':
        return <AlgorithmImplementationSection />;
      case 'code_explorer':
        return <CodeExplorerSection />;
      case 'tech_stack':
        return <TechnologyStackSection />;
      case 'validation_audit':
        return <ValidationAuditSection />;
      case 'limitations':
        return <LimitationsSection />;
      case 'future_scope':
        return <FutureScopeSection />;
      case 'prototype':
        return <InteractivePrototypeSection onSwitchToPrototype={handleSwitchToPrototype} />;
      case 'team':
        return <TeamSection />;
      default:
        return <ProjectOverviewSection />;
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

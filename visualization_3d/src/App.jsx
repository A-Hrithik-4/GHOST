import React, { useState, useEffect, useRef } from 'react';
import SingleVehicleScene from './components/SingleVehicleScene';
import StatusOverlay from './components/StatusOverlay';
import SimulationControls from './components/SimulationControls';
import ProjectModeContainer from './project_mode/ProjectModeContainer';
import trajectoryData from './data/trajectoryData.json';
import { getInterpolatedSample } from './utils/trajectoryUtils';
import './styles/app.css';

export default function App() {
  const [activeMode, setActiveMode] = useState('prototype'); // 'prototype' or 'project'
  const [currentTime, setCurrentTime] = useState(0.0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const maxTime = 120.0;

  const requestRef = useRef();
  const lastTimeRef = useRef();

  const animate = (time) => {
    if (lastTimeRef.current !== undefined && isPlaying && activeMode === 'prototype') {
      const delta = (time - lastTimeRef.current) / 1000.0;
      setCurrentTime((prevTime) => {
        let nextTime = prevTime + delta * playbackSpeed;
        if (nextTime > maxTime) {
          nextTime = 0.0;
        }
        return nextTime;
      });
    }
    lastTimeRef.current = time;
    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying, playbackSpeed, activeMode]);

  const currentSample = getInterpolatedSample(trajectoryData.samples, currentTime);
  const isOutage = currentSample ? currentSample.state === 'OUTAGE' : false;

  const handleTogglePlay = () => setIsPlaying(!isPlaying);
  const handleReset = () => {
    setCurrentTime(0.0);
    setIsPlaying(true);
  };
  const handleSeek = (time) => setCurrentTime(time);
  const handleSpeedChange = (spd) => setPlaybackSpeed(spd);
  const handleSwitchMode = (mode) => setActiveMode(mode);

  const p7_err = currentSample && currentSample.p7 ? currentSample.p7.err : (currentSample ? currentSample.p6.err : 0);
  const p7_spd = currentSample && currentSample.p7 ? currentSample.p7.spd : (currentSample ? currentSample.p6.spd : 0);

  if (activeMode === 'project') {
    return <ProjectModeContainer activeMode={activeMode} onSwitchMode={handleSwitchMode} />;
  }

  return (
    <div className="app-container">
      {/* Top Header Bar & Outage Alert Banner */}
      <StatusOverlay 
        sample={currentSample} 
        currentTime={currentTime} 
        activeMode={activeMode} 
        onSwitchMode={handleSwitchMode} 
      />

      {/* Main Stage: 2 Framed Viewport Cards (Side 1: Raw Baseline vs Side 2: GHOST AI Final Track) */}
      <main className="main-stage">
        {/* Viewport Card 1: CAR 1 — WITHOUT GHOST AI (Raw Baseline from ISRO Dataset) */}
        <div className="viewport-card">
          <div className="card-top-bar bar-red">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="card-badge badge-red">CAR 1</span>
              <span className="card-title">WITHOUT GHOST AI (ISRO RAW DATASET)</span>
            </div>
            <span className="card-status-pill status-gnss-lost">
              {isOutage ? 'GNSS LOST — RAW DR' : '● GNSS AVAILABLE'}
            </span>
          </div>

          <div className="canvas-frame">
            <SingleVehicleScene 
              sample={currentSample} 
              allSamples={trajectoryData.samples} 
              gtRoadPoints={trajectoryData.gt_road_points || []} 
              isBaseline={true} 
            />
          </div>

          {currentSample && (
            <div className="card-hud-bottom">
              <div className="hud-metric">
                <span className="hud-lbl">POSITION ERROR</span>
                <span className="hud-val text-red">{isOutage ? `${currentSample.p4.err.toFixed(1)} m` : '0.0 m'}</span>
              </div>
              <div className="hud-metric">
                <span className="hud-lbl">OUTAGE DRIFT</span>
                <span className="hud-val text-red">69.73 % (FAIL)</span>
              </div>
              <div className="hud-metric">
                <span className="hud-lbl">TRACKER STATUS</span>
                <span className="hud-val text-red">{isOutage ? 'UNCALIBRATED DR' : 'GPS LOCKED'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Viewport Card 2: CAR 2 — GHOST AI ENABLED (Green Theme matching Green Trajectory Path) */}
        <div className="viewport-card">
          <div className="card-top-bar bar-green">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="card-badge badge-green">CAR 2</span>
              <span className="card-title">GHOST AI ENABLED (FINAL PHASE 7)</span>
            </div>
            <span className="card-status-pill status-ghost-active">
              {isOutage ? 'GHOST AI ACTIVE (ISOTONIC EKF)' : '● GNSS AVAILABLE'}
            </span>
          </div>

          <div className="canvas-frame">
            <SingleVehicleScene 
              sample={currentSample} 
              allSamples={trajectoryData.samples} 
              gtRoadPoints={trajectoryData.gt_road_points || []} 
              isBaseline={false} 
            />
          </div>

          {currentSample && (
            <div className="card-hud-bottom">
              <div className="hud-metric">
                <span className="hud-lbl">POSITION ERROR</span>
                <span className="hud-val text-green">{isOutage ? `${p7_err.toFixed(1)} m` : '0.0 m'}</span>
              </div>
              <div className="hud-metric">
                <span className="hud-lbl">CALIBRATED SPEED</span>
                <span className="hud-val text-green">{p7_spd.toFixed(1)} km/h</span>
              </div>
              <div className="hud-metric">
                <span className="hud-lbl">OUTAGE DRIFT</span>
                <span className="hud-val text-green">2.40 % (PASS ≤10%)</span>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Bottom Timeline Controls Bar */}
      <SimulationControls
        currentTime={currentTime}
        maxTime={maxTime}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
        onTogglePlay={handleTogglePlay}
        onReset={handleReset}
        onSeek={handleSeek}
        onSpeedChange={handleSpeedChange}
      />
    </div>
  );
}

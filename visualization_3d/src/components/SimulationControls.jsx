import React from 'react';
import { Play, Pause, RotateCcw, AlertTriangle, CheckCircle2 } from 'lucide-react';

/**
 * SimulationControls.jsx
 * 
 * Full-width long timeline slider bar with exact 30s and 90s outage markers
 */

export default function SimulationControls({
  currentTime,
  maxTime = 120,
  isPlaying,
  playbackSpeed,
  onTogglePlay,
  onReset,
  onSeek,
  onSpeedChange
}) {
  const isOutage = currentTime >= 30.0 && currentTime < 90.0;

  return (
    <div className="sim-controls-bar">
      {/* Left Control Buttons */}
      <div className="controls-left">
        <button 
          className={`btn-control ${isPlaying ? 'btn-pause' : 'btn-play'}`}
          onClick={onTogglePlay}
        >
          {isPlaying ? <Pause size={13} /> : <Play size={13} />}
          <span>{isPlaying ? 'PAUSE' : 'PLAY'}</span>
        </button>

        <button className="btn-control btn-reset" onClick={onReset}>
          <RotateCcw size={13} />
          <span>RESET</span>
        </button>
      </div>

      {/* Center Long Timeline Section */}
      <div className="controls-center">
        {/* Info Header */}
        <div className="timeline-info-row">
          <div className="time-reading">
            <span className="time-curr">{currentTime.toFixed(1)} s</span>
            <span className="time-max"> / {maxTime.toFixed(0)} s</span>
          </div>
          {isOutage ? (
            <span className="outage-active-pill" style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <AlertTriangle size={13} />
              <span>60s GNSS OUTAGE ACTIVE (t = 30.0s — 90.0s)</span>
            </span>
          ) : (
            <span className="gnss-ok-pill" style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <CheckCircle2 size={13} />
              <span>GNSS SIGNAL LOCKED</span>
            </span>
          )}
        </div>

        {/* Long Slider Track Container */}
        <div className="long-slider-container">
          {/* Outage Shaded Band (25% to 75% corresponding to 30s to 90s) */}
          <div className="outage-shaded-band" />

          {/* Interactive Range Input */}
          <input
            type="range"
            min="0"
            max={maxTime}
            step="0.1"
            value={currentTime}
            onChange={(e) => onSeek(parseFloat(e.target.value))}
            className="long-timeline-slider"
          />

          {/* Marker Tick Labels Below Slider */}
          <div className="timeline-ticks-row">
            <span className="tick-start">0s</span>
            <span className="tick-marker tick-outage-start" style={{ left: '25%' }}>
              30s OUTAGE START
            </span>
            <span className="tick-marker tick-outage-end" style={{ left: '75%' }}>
              90s GNSS RESTORED
            </span>
            <span className="tick-end">120s</span>
          </div>
        </div>
      </div>

      {/* Right Playback Speed Buttons */}
      <div className="controls-right">
        <span className="speed-label">SPEED</span>
        <div className="speed-buttons">
          {[1, 2, 5].map((spd) => (
            <button
              key={spd}
              className={`btn-speed ${playbackSpeed === spd ? 'active' : ''}`}
              onClick={() => onSpeedChange(spd)}
            >
              {spd}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

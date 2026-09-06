import React from 'react';
import { Layers, FileText } from 'lucide-react';

export default function SlidingModeToggle({ activeMode, onSwitchMode }) {
  const isProto = activeMode === 'prototype';

  return (
    <div className={`sliding-toggle-container ${isProto ? 'is-proto' : 'is-project'}`}>
      <div className="sliding-pill-indicator" />
      <button
        type="button"
        className={`sliding-btn ${isProto ? 'active' : ''}`}
        onClick={() => onSwitchMode('prototype')}
      >
        <Layers size={13} />
        <span>Prototype</span>
      </button>
      <button
        type="button"
        className={`sliding-btn ${!isProto ? 'active' : ''}`}
        onClick={() => onSwitchMode('project')}
      >
        <FileText size={13} />
        <span>Project</span>
      </button>
    </div>
  );
}

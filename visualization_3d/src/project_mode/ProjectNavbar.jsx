import React from 'react';
import SlidingModeToggle from '../components/SlidingModeToggle';

export default function ProjectNavbar({ activeMode, onSwitchMode }) {
  return (
    <header className="global-header">
      <div className="header-brand-box">
        <span className="header-logo" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '18px', fontWeight: '900', color: 'var(--amber-primary)', letterSpacing: '2px' }}>
          GHOST
        </span>
        <div className="header-divider" />
        
        {/* Top-Left Sliding Mode Toggle Button */}
        <SlidingModeToggle activeMode={activeMode} onSwitchMode={onSwitchMode} />

        <div className="header-divider" />
        <div className="header-sub-info">
          <h1 style={{ fontSize: '12px', fontWeight: '800', letterSpacing: '1px', color: 'var(--graphite-dark)', textTransform: 'uppercase', margin: 0 }}>
            PROJECT MODE
          </h1>
        </div>
      </div>

      <div className="header-right-group" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <div className="sih-tag" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', fontWeight: '800', color: 'var(--graphite-secondary)', letterSpacing: '1px' }}>
          SIH 26168
        </div>
      </div>
    </header>
  );
}

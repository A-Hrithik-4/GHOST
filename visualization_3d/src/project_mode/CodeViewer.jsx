import React, { useState } from 'react';
import { Copy, Check, ChevronDown, ChevronUp, Code } from 'lucide-react';

/**
 * CodeViewer Component
 * Renders a dark contrast code panel with line numbers, copy button, syntax formatting, and expand/collapse support.
 */
export default function CodeViewer({ filename, language = 'python', code = '', initialExpanded = true }) {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(initialExpanded);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = code.trim().split('\n');

  // Simple token highlighter for Python/JS syntax
  const highlightLine = (line) => {
    if (!line) return '';

    // Handle comments
    if (line.trim().startsWith('#') || line.trim().startsWith('//')) {
      return <span className="token-comment">{line}</span>;
    }

    // Keyword replace list
    const keywords = ['import', 'from', 'def', 'class', 'return', 'for', 'if', 'else', 'elif', 'in', 'while', 'assert', 'with', 'as', 'try', 'except', 'raise', 'True', 'False', 'None', 'const', 'let', 'function', 'export'];
    
    // Split into tokens safely
    const words = line.split(/(\s+|[(),.:=\[\]{}])/);
    return words.map((word, idx) => {
      if (keywords.includes(word)) {
        return <span key={idx} className="token-keyword">{word}</span>;
      }
      if (/^\d+\.?\d*$/.test(word)) {
        return <span key={idx} className="token-number">{word}</span>;
      }
      if (/^(['"]).*\1$/.test(word)) {
        return <span key={idx} className="token-string">{word}</span>;
      }
      return word;
    });
  };

  return (
    <div className="code-viewer-container">
      <div className="code-viewer-header">
        <div className="code-filename">
          <Code size={14} color="#E0A526" />
          <span>{filename}</span>
          <span style={{ fontSize: '10px', color: '#6B7268', paddingLeft: '6px', textTransform: 'uppercase' }}>({language})</span>
        </div>
        <div className="code-actions">
          <button className="btn-code-action" onClick={handleCopy} title="Copy code to clipboard">
            {copied ? <Check size={12} color="#16803C" /> : <Copy size={12} />}
            <span>{copied ? 'COPIED' : 'COPY'}</span>
          </button>
          <button className="btn-code-action" onClick={() => setIsExpanded(!isExpanded)}>
            {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            <span>{isExpanded ? 'COLLAPSE' : 'EXPAND'}</span>
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="code-body">
          {lines.map((lineText, idx) => (
            <div key={idx} className="code-line">
              <span className="line-num">{idx + 1}</span>
              <span className="line-text">{highlightLine(lineText)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

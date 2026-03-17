import { useState, useEffect } from 'react';
import { ShieldCheck, Database, FileText, Microscope } from 'lucide-react';
import './PathologyLoader.css';

const PathologyLoader = () => {
  const [stage, setStage] = useState(0);
  const stages = [
    { label: "Connecting to Secure Lab Server", icon: <Database className="stage-icon" /> },
    { label: "Sequencing Histopathology Data", icon: <Microscope className="stage-icon" /> },
    { label: "Verifying Physician Signatures", icon: <ShieldCheck className="stage-icon" /> },
    { label: "Finalizing Report Layout", icon: <FileText className="stage-icon" /> }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStage((prev) => (prev < stages.length - 1 ? prev + 1 : prev));
    }, 1800);
    return () => clearInterval(interval);
  }, [stages.length]);

  return (
    <div className="pathology-loader-overlay">
      {/* Soft Background "Cell" Animation */}
      <div className="pathology-loader-background">
        <div className="pathology-blob blob-1"></div>
        <div className="pathology-blob blob-2"></div>
      </div>

      <div className="pathology-loader-card">
        {/* Header */}
        <div className="pathology-loader-header">
          <div className="pathology-loader-icon-container">
            <Microscope className="pathology-main-icon" />
          </div>
          <h2 className="pathology-loader-title">Preparing Your Report</h2>
          <p className="pathology-loader-subtitle">Ensuring clinical accuracy & security</p>
        </div>

        {/* Stages List */}
        <div className="pathology-stages-list">
          {stages.map((s, index) => (
            <div
              key={index}
              className={`pathology-stage ${index <= stage ? 'active' : 'inactive'}`}
            >
              <div className={`pathology-stage-icon ${index === stage ? 'current' : ''}`}>
                {s.icon}
              </div>
              <span className={`pathology-stage-label ${index === stage ? 'current' : ''}`}>
                {s.label}
              </span>
              {index < stage && (
                <div className="pathology-stage-check">
                  <svg className="check-icon" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Progress Bar */}
        <div className="pathology-progress-container">
          <div
            className="pathology-progress-bar"
            style={{ width: `${((stage + 1) / stages.length) * 100}%` }}
          />
        </div>

        <p className="pathology-loader-footer">
          Encrypted End-to-End • HIPAA Compliant
        </p>
      </div>
    </div>
  );
};

export default PathologyLoader;

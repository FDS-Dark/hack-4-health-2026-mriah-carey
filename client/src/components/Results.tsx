import { useEffect, useState, useCallback, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { saveDocument } from '../utils/documentStorage';
import Glossary from './Glossary';
import CitedText from './CitedText';
import './Results.css';

// Helper function to normalize source titles to valid HTML IDs
const normalizeToId = (title: string): string => {
  return `source-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}`;
};

interface SourceWithNumber {
  title: string;
  url: string;
  summary: string;
  citationNumber: number;
}

function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const documentId = location.state?.documentId;
  const [highlightedSourceId, setHighlightedSourceId] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);
  
  // Get data from API response or navigate back if missing
  const fileName = location.state?.fileName;
  const originalDiagnosis = location.state?.originalText;
  const simplifiedDiagnosis = location.state?.simplifiedText;
  const glossaryTerms = location.state?.glossary || [];
  const simplifiedJson = location.state?.simplifiedJson || [];
  const citations = location.state?.citations || simplifiedJson.reduce((acc: { [key: string]: string[] }, chunk: any) => {
    if (chunk.citations) {
      Object.entries(chunk.citations).forEach(([sourceId, segments]) => {
        if (!acc[sourceId]) acc[sourceId] = [];
        acc[sourceId].push(...(segments as string[]));
      });
    }
    return acc;
  }, {} as { [sourceId: string]: string[] });

  // Create citation number map (same logic as CitedText component)
  const citationNumbers = useMemo(() => {
    const numbers: { [key: string]: number } = {};
    let counter = 1;
    Object.keys(citations).forEach(key => {
      numbers[key] = counter++;
    });
    return numbers;
  }, [citations]);

  // Get all sources from chunks and filter to only show cited ones
  const citedSources = useMemo((): SourceWithNumber[] => {
    const allSources: { [title: string]: { url: string; summary: string } } = {};
    
    // Collect all sources from chunks
    simplifiedJson.forEach((chunk: any) => {
      try {
        const sources = JSON.parse(chunk.sources || '[]');
        sources.forEach((source: any) => {
          const title = source.title || 'Untitled';
          if (!allSources[title]) {
            allSources[title] = {
              url: source.url || source.link || '',
              summary: source.summary || 'No summary available'
            };
          }
        });
      } catch (e) {
        console.error('Error parsing sources:', e);
      }
    });

    // Filter to only include sources that are cited
    const citedSourcesList: SourceWithNumber[] = [];
    Object.keys(citations).forEach(sourceTitle => {
      const citationNum = citationNumbers[sourceTitle];
      if (citationNum && allSources[sourceTitle]) {
        citedSourcesList.push({
          title: sourceTitle,
          url: allSources[sourceTitle].url,
          summary: allSources[sourceTitle].summary,
          citationNumber: citationNum
        });
      }
    });

    // Sort by citation number
    citedSourcesList.sort((a, b) => a.citationNumber - b.citationNumber);
    
    return citedSourcesList;
  }, [simplifiedJson, citations, citationNumbers]);

  // Redirect if no data (shouldn't happen, but handle gracefully)
  useEffect(() => {
    if (!fileName && !documentId) {
      navigate('/');
    }
  }, [fileName, documentId, navigate]);

  // Handle citation click - scroll to the corresponding source
  const handleCitationClick = useCallback((sourceTitle: string) => {
    const sourceId = normalizeToId(sourceTitle);
    const sourceElement = document.getElementById(sourceId);
    
    if (sourceElement) {
      // Scroll to the source with smooth behavior
      sourceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      // Highlight the source temporarily
      setHighlightedSourceId(sourceId);
      
      // Remove highlight after 2 seconds
      setTimeout(() => {
        setHighlightedSourceId(null);
      }, 2000);
    }
  }, []);

  // Save document to localStorage when component mounts (only if it's a new document)
  useEffect(() => {
    if (!documentId && fileName && originalDiagnosis && simplifiedDiagnosis) {
      saveDocument({
        fileName,
        summary: simplifiedDiagnosis, // Use simplified text as summary
        originalDiagnosis,
        simplifiedDiagnosis,
      });
    }
  }, [documentId, fileName, originalDiagnosis, simplifiedDiagnosis]);

  // Show loading or error state if data is missing
  if (!fileName || !originalDiagnosis || !simplifiedDiagnosis) {
    if (documentId) {
      // This is a saved document being viewed, data should be in storage
      return null; // PastDocuments component handles this
    }
    return (
      <div className="results-container">
        <div className="results-content">
          <h1 className="page-title">Medical Report Analysis</h1>
          <p>Loading document data...</p>
        </div>
      </div>
    );
  }

  // Navigate to Doctor View with current document data
  const handleDoctorView = () => {
    navigate('/doctor-view', {
      state: {
        fileName,
        originalText: originalDiagnosis,
        simplifiedText: simplifiedDiagnosis,
      }
    });
  };

  return (
    <div className="results-container">
      <div className="results-content">
        <div className="results-header">
          <h1 className="page-title">Medical Report Analysis</h1>
          <button className="doctor-view-btn" onClick={handleDoctorView}>
            Doctor View
          </button>
        </div>
        
        {/* Simplified Diagnosis and Glossary side by side */}
        <div className="main-content-section">
          <div className="simplified-box">
            <label className="section-label">Simplified Medical Diagnosis</label>
            <div className="diagnosis-text">
              <CitedText
                text={simplifiedDiagnosis}
                chunks={simplifiedJson}
                citations={citations}
                onCitationClick={handleCitationClick}
                glossaryTerms={glossaryTerms}
              />
            </div>
          </div>

          <div className="glossary-sidebar">
            <Glossary terms={glossaryTerms} />
          </div>
        </div>

        {/* Sources Section - Only show cited sources */}
        {citedSources.length > 0 && (
          <div className="sources-section">
            <label className="section-label">Supporting Medical Sources</label>
            <div className="sources-list">
              {citedSources.map((source) => {
                const sourceId = normalizeToId(source.title);
                const isHighlighted = highlightedSourceId === sourceId;
                return (
                  <div 
                    key={sourceId}
                    id={sourceId}
                    className={`source-item ${isHighlighted ? 'source-item-highlighted' : ''}`}
                  >
                    <div className="source-header">
                      <span className="source-citation-badge">[{source.citationNumber}]</span>
                      <div className="source-title">{source.title}</div>
                    </div>
                    <a 
                      href={source.url} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="source-link"
                    >
                      {source.url || 'No link available'}
                    </a>
                    <div className="source-summary">
                      {source.summary}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Collapsible Original Diagnosis */}
        <div className="original-diagnosis-section">
          <button 
            className="toggle-original-btn"
            onClick={() => setShowOriginal(!showOriginal)}
          >
            {showOriginal ? '▼ Hide' : '▶ Show'} Original Medical Diagnosis
          </button>
          {showOriginal && (
            <div className="original-diagnosis-content">
              <div className="diagnosis-text original-text">
                {originalDiagnosis}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Results;

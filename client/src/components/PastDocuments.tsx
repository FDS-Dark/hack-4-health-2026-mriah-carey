import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getDocuments, formatDate, formatDateShort, deleteDocument, type DocumentData } from '../utils/documentStorage';
import './PastDocuments.css';

function PastDocuments() {
  const [documents, setDocuments] = useState<DocumentData[]>([]);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const docs = getDocuments();
    // Sort by date, most recent first
    const sortedDocs = [...docs].sort((a, b) => 
      new Date(b.date).getTime() - new Date(a.date).getTime()
    );
    setDocuments(sortedDocs);
    if (sortedDocs.length > 0 && !activeTab) {
      setActiveTab(sortedDocs[0].id);
    }
  }, [activeTab]);

  const handleTabClick = (id: string) => {
    setActiveTab(id);
  };

  const handleCloseTab = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    deleteDocument(id);
    const updated = getDocuments();
    setDocuments(updated);
    if (activeTab === id) {
      setActiveTab(updated.length > 0 ? updated[0].id : null);
    }
  };

  const activeDocument = documents.find(doc => doc.id === activeTab);

  if (documents.length === 0) {
    return (
      <div className="past-documents-container">
        <div className="empty-state">
          <h2>No Past Documents</h2>
          <p>You haven't uploaded any documents yet.</p>
          <button onClick={() => navigate('/')} className="upload-button">
            Upload Your First Document
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="past-documents-container">
      <h1 className="page-title">Past Documents</h1>
      
      <div className="documents-list">
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={`document-link ${activeTab === doc.id ? 'active' : ''}`}
            onClick={() => handleTabClick(doc.id)}
          >
            <div className="document-link-content">
              <span className="document-name">{doc.fileName}</span>
              <span className="document-date">Date of Visit: {formatDate(doc.date)}</span>
            </div>
            <button
              className="document-delete"
              onClick={(e) => handleCloseTab(e, doc.id)}
              aria-label="Delete document"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {activeDocument && (
        <div className="document-content">
          <div className="summary-section">
            <label className="section-label">Summary</label>
            <textarea
              className="summary-box"
              value={activeDocument.summary}
              readOnly
            />
          </div>

          <div className="diagnosis-section">
            <div className="diagnosis-box original">
              <label className="section-label">Original Medical Diagnosis</label>
              <textarea
                className="diagnosis-textarea"
                value={activeDocument.originalDiagnosis}
                readOnly
              />
            </div>

            <div className="diagnosis-box simplified">
              <label className="section-label">Simplified Medical Diagnosis</label>
              <textarea
                className="diagnosis-textarea"
                value={activeDocument.simplifiedDiagnosis}
                readOnly
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PastDocuments;

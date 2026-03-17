import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ThumbsUp, ThumbsDown } from 'lucide-react';
import './DoctorView.css';

function DoctorView() {
  const location = useLocation();
  const navigate = useNavigate();
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedbackType, setFeedbackType] = useState<'up' | 'down' | null>(null);

  // Get data from navigation state
  const fileName = location.state?.fileName;
  const originalDiagnosis = location.state?.originalText;
  const simplifiedDiagnosis = location.state?.simplifiedText;

  const handleFeedback = (type: 'up' | 'down') => {
    setFeedbackType(type);
    setFeedbackSubmitted(true);
    
    // In a real app, you would send this feedback to the server
    console.log(`Doctor feedback: ${type === 'up' ? 'Accurate' : 'Inaccurate'}`);
  };

  // If no data, show message to process a document first
  if (!simplifiedDiagnosis) {
    return (
      <div className="doctor-view-container">
        <div className="doctor-view-content">
          <h1 className="page-title">Doctor View</h1>
          <div className="no-data-message">
            <p>No document data available.</p>
            <p>Please process a medical report first to review the summarization.</p>
            <button 
              className="upload-button"
              onClick={() => navigate('/')}
            >
              Upload a Document
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="doctor-view-container">
      <div className="doctor-view-content">
        <h1 className="page-title">Doctor View</h1>
        
        {fileName && (
          <p className="document-name">Reviewing: <strong>{fileName}</strong></p>
        )}

        <div className="summary-sections">
          {/* Original Diagnosis */}
          <div className="summary-box">
            <h2 className="section-title">Original Medical Report</h2>
            <div className="summary-text original">
              {originalDiagnosis}
            </div>
          </div>

          {/* Simplified Diagnosis */}
          <div className="summary-box">
            <h2 className="section-title">AI-Generated Summary</h2>
            <div className="summary-text simplified">
              {simplifiedDiagnosis}
            </div>
          </div>
        </div>

        {/* Feedback Section */}
        <div className="feedback-section">
          <h2 className="feedback-title">Rate the accuracy of this summarization</h2>
          
          {!feedbackSubmitted ? (
            <div className="feedback-buttons">
              <button 
                className="feedback-btn thumbs-up"
                onClick={() => handleFeedback('up')}
                aria-label="Thumbs up - Accurate"
              >
                <ThumbsUp size={32} />
                <span>Accurate</span>
              </button>
              <button 
                className="feedback-btn thumbs-down"
                onClick={() => handleFeedback('down')}
                aria-label="Thumbs down - Inaccurate"
              >
                <ThumbsDown size={32} />
                <span>Inaccurate</span>
              </button>
            </div>
          ) : (
            <div className="feedback-thank-you">
              <div className={`feedback-icon ${feedbackType}`}>
                {feedbackType === 'up' ? <ThumbsUp size={40} /> : <ThumbsDown size={40} />}
              </div>
              <p>Thank you for your feedback!</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DoctorView;

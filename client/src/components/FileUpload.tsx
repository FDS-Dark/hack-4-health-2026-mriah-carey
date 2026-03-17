import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getUserData } from '../utils/userStorage';
import PathologyLoader from './PathologyLoader';
import './FileUpload.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function FileUpload() {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    const validExtensions = ['.txt', '.pdf'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

    // Check file extension (more reliable than MIME type)
    if (!validExtensions.includes(fileExtension)) {
      alert('Please upload a .txt or .pdf file');
      return;
    }

    // Check file size (max 10MB for PDFs, 1MB for text)
    const maxSize = fileExtension === '.pdf' ? 10 * 1024 * 1024 : 1024 * 1024;
    if (file.size > maxSize) {
      alert(`File too large. Maximum size is ${maxSize / 1024 / 1024}MB`);
      return;
    }

    setUploading(true);

    try {
      // For PDFs, we send the raw file - the API will handle OCR
      // For text files, we can read and validate the content first
      let formData = new FormData();
      
      if (fileExtension === '.pdf') {
        // Send PDF directly - API uses Gemini OCR
        formData.append('file', file);
      } else {
        // Text file - read and validate content
        const originalText = await file.text();
        
        if (!originalText || originalText.trim().length === 0) {
          throw new Error('File is empty or could not be read');
        }
        
        // Create a new File object from the text for FormData
        const fileBlob = new Blob([originalText], { type: 'text/plain' });
        const fileForUpload = new File([fileBlob], file.name, { type: 'text/plain' });
        formData.append('file', fileForUpload);
      }

      console.log('Sending file to API:', `${API_BASE_URL}/process-report`);
      
      const response = await fetch(`${API_BASE_URL}/process-report`, {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status, response.statusText);

      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          const errorText = await response.text();
          console.error('Error response text:', errorText);
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('API Response received:', data);
      
      // Map the ProcessingResponse to what the frontend expects
      // data contains: structured_json, simplified_json, simplified_text, summary_text, original_text, metadata
      const simplifiedText = data.simplified_text || '';
      const summaryText = data.summary_text || '';
      // For PDFs, original text comes from API (Gemini OCR extraction)
      // For text files, we could use local read but API provides it too
      const originalText = data.original_text || '';
      
      console.log('Navigating to results with:', {
        fileName: file.name,
        originalTextLength: originalText.length,
        simplifiedTextLength: simplifiedText.length,
        summaryTextLength: summaryText.length,
        isPdf: file.name.toLowerCase().endsWith('.pdf'),
      });
      
      navigate('/results', {
        state: {
          fileName: file.name,
          originalText: originalText, // Original text from API (OCR for PDFs)
          simplifiedText: simplifiedText, // Compiled simplified text
          summary: summaryText, // Summary text
          glossary: data.glossary || [], // Glossary terms
          simplifiedJson: data.simplified_json || [], // Full simplified data with sources
          needsReview: data.needs_review || false, // Validation flag
        },
      });
    } catch (error) {
      console.error('Error uploading file:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Error processing file: ${errorMessage}\n\nPlease check:\n1. The API server is running at ${API_BASE_URL}\n2. The file is a valid .txt or .pdf file\n3. Check the browser console for more details`);
    } finally {
      setUploading(false);
    }
  };

  const userData = getUserData();
  const userName = userData.name || 'User';

  return (
    <>
      {uploading && <PathologyLoader />}
      <div className="upload-container">
        <div className="welcome-message">
          <h1>Welcome {userName}!</h1>
        </div>
        <h2 className="page-title">Upload New File</h2>
        <div
          className={`upload-box ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="upload-content">
            <svg
              className="upload-icon"
              width="64"
              height="64"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <h2>Upload Medical Document</h2>
            <p>Drag and drop your file here, or click to browse</p>
            <p className="file-types">Supported formats: PDF, TXT</p>
            <input
              type="file"
              id="file-upload"
              accept=".txt,.pdf,application/pdf,text/plain"
              onChange={handleChange}
              className="file-input"
              disabled={uploading}
            />
            <label htmlFor="file-upload" className={`upload-button ${uploading ? 'uploading' : ''}`}>
              {uploading ? 'Processing...' : 'Choose File'}
            </label>
          </div>
        </div>
      </div>
    </>
  );
}

export default FileUpload;

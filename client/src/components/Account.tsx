import { useState, useEffect } from 'react';
import { getDocuments } from '../utils/documentStorage';
import { getUserData, saveUserData, type UserData } from '../utils/userStorage';
import './Account.css';

function Account() {
  const [documentCount, setDocumentCount] = useState(0);
  const [userData, setUserData] = useState<UserData>(getUserData());
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    const docs = getDocuments();
    setDocumentCount(docs.length);
    setUserData(getUserData());
  }, []);

  const handleInputChange = (field: keyof UserData, value: string) => {
    setUserData({ ...userData, [field]: value });
  };

  const handleSave = () => {
    saveUserData(userData);
    setIsEditing(false);
    alert('Profile information saved successfully!');
  };

  return (
    <div className="account-container">
      <h1 className="page-title">Account</h1>
      
      <div className="patient-info">
        <div className="info-item">
          <span className="info-label">Patient Name:</span>
          <span className="info-value">{userData.name || 'Not set'}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Date of Birth:</span>
          <span className="info-value">
            {userData.dateOfBirth 
              ? new Date(userData.dateOfBirth).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })
              : 'Not set'}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Address:</span>
          <span className="info-value">{userData.address || 'Not set'}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Preferred Medical Facility:</span>
          <span className="info-value">{userData.preferredMedicalFacility || 'Not set'}</span>
        </div>
        {userData.email && (
          <div className="info-item">
            <span className="info-label">Email:</span>
            <span className="info-value">{userData.email}</span>
          </div>
        )}
      </div>
      
      <div className="account-content">
        <div className="account-section">
          <h2>Profile Information</h2>
          <div className="form-group">
            <label>Name</label>
            <input 
              type="text" 
              placeholder="Enter your name"
              value={userData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Email</label>
            <input 
              type="email" 
              placeholder="Enter your email"
              value={userData.email || ''}
              onChange={(e) => handleInputChange('email', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Date of Birth</label>
            <input 
              type="date"
              value={userData.dateOfBirth}
              onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Address</label>
            <input 
              type="text" 
              placeholder="Enter your address"
              value={userData.address}
              onChange={(e) => handleInputChange('address', e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Preferred Medical Facility</label>
            <input 
              type="text" 
              placeholder="Enter preferred medical facility"
              value={userData.preferredMedicalFacility}
              onChange={(e) => handleInputChange('preferredMedicalFacility', e.target.value)}
            />
          </div>
          <button className="save-button" onClick={handleSave}>Save Changes</button>
        </div>

        <div className="account-section">
          <h2>Account Statistics</h2>
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">{documentCount}</div>
              <div className="stat-label">Documents Uploaded</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{documentCount}</div>
              <div className="stat-label">Saved Documents</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Account;

import './Settings.css';

function Settings() {
  return (
    <div className="settings-container">
      <h1 className="page-title">Settings</h1>
      
      <div className="settings-content">
        <div className="settings-section">
          <h2>Preferences</h2>
          <div className="setting-item">
            <label>
              <input type="checkbox" defaultChecked />
              Email notifications for new documents
            </label>
          </div>
          <div className="setting-item">
            <label>
              <input type="checkbox" />
              Auto-save documents
            </label>
          </div>
        </div>

        <div className="settings-section">
          <h2>Privacy</h2>
          <div className="setting-item">
            <label>
              <input type="checkbox" defaultChecked />
              Share anonymized data for research
            </label>
          </div>
        </div>

        <div className="settings-section">
          <h2>Data Management</h2>
          <div className="setting-item">
            <button className="danger-button" onClick={() => {
              if (confirm('Are you sure you want to delete all saved documents? This action cannot be undone.')) {
                localStorage.removeItem('benmed_documents');
                alert('All documents have been deleted.');
              }
            }}>
              Delete All Documents
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;

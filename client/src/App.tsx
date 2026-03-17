import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import Results from './components/Results';
import PastDocuments from './components/PastDocuments';
import Settings from './components/Settings';
import Account from './components/Account';
import DoctorView from './components/DoctorView';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Header />
        <Routes>
          <Route path="/" element={<FileUpload />} />
          <Route path="/results" element={<Results />} />
          <Route path="/past-documents" element={<PastDocuments />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/account" element={<Account />} />
          <Route path="/doctor-view" element={<DoctorView />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

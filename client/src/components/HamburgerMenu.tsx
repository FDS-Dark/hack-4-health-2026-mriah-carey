import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './HamburgerMenu.css';

function HamburgerMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleMenuClick = (path: string) => {
    setIsOpen(false);
    navigate(path);
  };

  const handleLogout = () => {
    setIsOpen(false);
    // In a real app, you'd clear authentication tokens here
    navigate('/');
  };

  return (
    <div className="hamburger-menu" ref={menuRef}>
      <button
        className={`hamburger-button ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Menu"
      >
        <span></span>
        <span></span>
        <span></span>
      </button>
      {isOpen && (
        <div className="menu-dropdown">
          <button
            className="menu-item"
            onClick={() => handleMenuClick('/account')}
          >
            Account
          </button>
          <button
            className="menu-item"
            onClick={() => handleMenuClick('/')}
          >
            Upload New File
          </button>
          <button
            className="menu-item"
            onClick={() => handleMenuClick('/past-documents')}
          >
            Past Documents
          </button>
          <button
            className="menu-item"
            onClick={() => handleMenuClick('/doctor-view')}
          >
            Doctor View
          </button>
          <button
            className="menu-item"
            onClick={() => handleMenuClick('/settings')}
          >
            Settings
          </button>
          <button
            className="menu-item logout"
            onClick={handleLogout}
          >
            Log Out
          </button>
        </div>
      )}
    </div>
  );
}

export default HamburgerMenu;

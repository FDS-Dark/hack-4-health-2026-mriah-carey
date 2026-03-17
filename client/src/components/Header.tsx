import benmedLogo from '../assets/benmed-logo.png';
import HamburgerMenu from './HamburgerMenu';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <img src={benmedLogo} alt="BenMed Logo" className="header-logo" />
      <HamburgerMenu />
    </header>
  );
}

export default Header;

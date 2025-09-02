import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import './Navbar.css';

const Navbar = ({ systemStatus }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Home', href: '/', icon: 'fas fa-home' },
    { name: 'Analysis', href: '/analysis', icon: 'fas fa-search' },
    { name: 'Live Detection', href: '/live', icon: 'fas fa-video' },
    { name: 'Dashboard', href: '/dashboard', icon: 'fas fa-chart-bar' },
    { name: 'Demo', href: '/demo', icon: 'fas fa-play-circle' },
    { name: 'Workflow', href: '/workflow', icon: 'fas fa-project-diagram' }
  ];

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <motion.nav 
      className="navbar"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="nav-container">
        {/* Logo and Brand */}
        <Link to="/" className="nav-brand">
          <div className="brand-icon">
            <i className="fas fa-shield-alt"></i>
          </div>
          <div className="brand-text">
            <span className="brand-title">DeepFake Shield</span>
            <span className="brand-subtitle">AI Detection System</span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="nav-menu">
          {navigation.map((item) => (
            <Link
              key={item.name}
              to={item.href}
              className={`nav-link ${location.pathname === item.href ? 'active' : ''}`}
            >
              <i className={item.icon}></i>
              <span>{item.name}</span>
            </Link>
          ))}
        </div>

        {/* System Status */}
        <div className="nav-status">
          <div className={`status-indicator ${systemStatus}`}>
            <div className="status-light"></div>
            <span className="status-label">
              {systemStatus === 'ready' && 'Online'}
              {systemStatus === 'checking' && 'Loading'}
              {systemStatus === 'error' && 'Offline'}
            </span>
          </div>
        </div>

        {/* Mobile Menu Button */}
        <button 
          className={`mobile-menu-btn ${isMenuOpen ? 'open' : ''}`}
          onClick={toggleMenu}
          aria-label="Toggle navigation menu"
        >
          <span></span>
          <span></span>
          <span></span>
        </button>
      </div>

      {/* Mobile Navigation */}
      <motion.div 
        className={`mobile-menu ${isMenuOpen ? 'open' : ''}`}
        initial={false}
        animate={{ height: isMenuOpen ? 'auto' : 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mobile-menu-content">
          {navigation.map((item) => (
            <Link
              key={item.name}
              to={item.href}
              className={`mobile-nav-link ${location.pathname === item.href ? 'active' : ''}`}
              onClick={() => setIsMenuOpen(false)}
            >
              <i className={item.icon}></i>
              <span>{item.name}</span>
            </Link>
          ))}
          
          <div className="mobile-status">
            <div className={`status-indicator ${systemStatus}`}>
              <div className="status-light"></div>
              <span className="status-label">
                System Status: {systemStatus === 'ready' ? 'Online' : systemStatus === 'checking' ? 'Loading' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.nav>
  );
};

export default Navbar;

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import AnalysisPage from './pages/AnalysisPage';
import DashboardPage from './pages/DashboardPage';
import LiveDetectionPage from './pages/LiveDetectionPage';
import ResultsPage from './pages/ResultsPage';
import DemoVisualization from './components/DemoVisualization';
import WorkflowVisualization from './components/WorkflowVisualization';
import { motion } from 'framer-motion';
import './App.css';

function App() {
  const [loading, setLoading] = useState(true);
  const [systemStatus, setSystemStatus] = useState('checking');

  useEffect(() => {
    // Check system status on app load
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('/health');
      const data = await response.json();
      
      if (data.status === 'healthy') {
        setSystemStatus('ready');
      } else {
        setSystemStatus('error');
      }
    } catch (error) {
      console.error('System status check failed:', error);
      setSystemStatus('error');
    } finally {
      // Minimum loading time for better UX
      setTimeout(() => setLoading(false), 1500);
    }
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <motion.div 
          className="loading-container"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="loading-spinner"></div>
          <h2>Digital Deepfake Detection System</h2>
          <p>Initializing AI models and security protocols...</p>
          <div className="loading-progress">
            <div className="loading-bar"></div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <Navbar systemStatus={systemStatus} />
        
        <motion.main 
          className="main-content"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Routes>
            <Route path="/" element={<HomePage systemStatus={systemStatus} />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/live" element={<LiveDetectionPage />} />
            <Route path="/results/:taskId" element={<ResultsPage />} />
            <Route path="/demo" element={<DemoVisualization />} />
            <Route path="/workflow" element={<WorkflowVisualization />} />
          </Routes>
        </motion.main>

        {/* System Status Indicator */}
        <div className={`system-status-indicator ${systemStatus}`}>
          <div className="status-dot"></div>
          <span className="status-text">
            {systemStatus === 'ready' && 'System Ready'}
            {systemStatus === 'checking' && 'Checking Status'}
            {systemStatus === 'error' && 'System Error'}
          </span>
        </div>
      </div>
    </Router>
  );
}

export default App;

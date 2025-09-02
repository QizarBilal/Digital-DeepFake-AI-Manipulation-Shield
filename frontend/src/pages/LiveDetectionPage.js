import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import './LiveDetectionPage.css';

const LiveDetectionPage = () => {
  const [isActive, setIsActive] = useState(false);
  const [detectionMode, setDetectionMode] = useState('video'); // 'video', 'audio', 'both'
  const [currentScore, setCurrentScore] = useState(85.7);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [systemStatus, setSystemStatus] = useState('ready');
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, []);

  const startDetection = async () => {
    try {
      setSystemStatus('initializing');
      
      // Get user media
      const constraints = {
        video: detectionMode === 'video' || detectionMode === 'both',
        audio: detectionMode === 'audio' || detectionMode === 'both'
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current && constraints.video) {
        videoRef.current.srcObject = stream;
      }
      
      // Connect to WebSocket
      connectWebSocket();
      
      setIsActive(true);
      setSystemStatus('active');
      
      // Start processing
      if (constraints.video) {
        startVideoProcessing();
      }
      
    } catch (error) {
      console.error('Failed to start detection:', error);
      setSystemStatus('error');
      addAlert('Failed to access camera/microphone. Please check permissions.', 'error');
    }
  };

  const stopDetection = () => {
    setIsActive(false);
    setSystemStatus('ready');
    
    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // Stop animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    
    // Clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live-detection`;
    
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      setSystemStatus('active');
    };
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleDetectionResult(data);
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setSystemStatus('error');
      addAlert('Connection error. Please try again.', 'error');
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
      if (isActive) {
        setSystemStatus('error');
        addAlert('Connection lost. Please restart detection.', 'warning');
      }
    };
  };

  const startVideoProcessing = () => {
    const processFrame = () => {
      if (!isActive || !videoRef.current || !canvasRef.current) return;
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Set canvas dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame
      ctx.drawImage(video, 0, 0);
      
      // Get frame data and send to server
      canvas.toBlob((blob) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onload = () => {
            wsRef.current.send(JSON.stringify({
              type: 'video_frame',
              data: reader.result.split(',')[1], // Remove data:image/png;base64, prefix
              timestamp: Date.now()
            }));
          };
          reader.readAsDataURL(blob);
        }
      }, 'image/jpeg', 0.8);
      
      animationRef.current = requestAnimationFrame(processFrame);
    };
    
    // Wait for video to be ready
    if (videoRef.current.readyState >= 2) {
      processFrame();
    } else {
      videoRef.current.addEventListener('loadeddata', processFrame);
    }
  };

  const handleDetectionResult = (data) => {
    const { authenticity_score, risk_level, detected_manipulations, timestamp } = data;
    
    setCurrentScore(authenticity_score);
    
    // Add to history
    const historyEntry = {
      id: Date.now(),
      score: authenticity_score,
      risk: risk_level,
      timestamp: new Date(timestamp),
      manipulations: detected_manipulations || []
    };
    
    setDetectionHistory(prev => [historyEntry, ...prev.slice(0, 49)]); // Keep last 50 entries
    
    // Check for alerts
    if (authenticity_score < 60) {
      addAlert(
        `Potential deepfake detected! Authenticity score: ${authenticity_score.toFixed(1)}%`,
        'error'
      );
    } else if (authenticity_score < 80) {
      addAlert(
        `Medium risk content detected. Score: ${authenticity_score.toFixed(1)}%`,
        'warning'
      );
    }
  };

  const addAlert = (message, type) => {
    const alert = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date()
    };
    
    setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      setAlerts(prev => prev.filter(a => a.id !== alert.id));
    }, 5000);
  };

  const getRiskColor = (score) => {
    if (score >= 80) return '#48bb78';
    if (score >= 60) return '#ed8936';
    if (score >= 40) return '#f56565';
    return '#e53e3e';
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return 'Low Risk';
    if (score >= 60) return 'Medium Risk';
    if (score >= 40) return 'High Risk';
    return 'Critical Risk';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready': return '#718096';
      case 'initializing': return '#ed8936';
      case 'active': return '#48bb78';
      case 'error': return '#f56565';
      default: return '#718096';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'ready': return 'Ready to Start';
      case 'initializing': return 'Initializing...';
      case 'active': return 'Live Detection Active';
      case 'error': return 'Error - Check Connection';
      default: return 'Unknown Status';
    }
  };

  return (
    <div className="live-detection-page">
      <div className="container">
        <motion.div 
          className="page-header"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="page-title">Live Detection</h1>
          <p className="page-description">
            Real-time deepfake detection from your camera and microphone input
          </p>
        </motion.div>

        <div className="detection-layout">
          {/* Main Detection Area */}
          <motion.div 
            className="detection-main"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            {/* Controls */}
            <div className="detection-controls">
              <div className="control-group">
                <label className="control-label">Detection Mode</label>
                <select 
                  value={detectionMode} 
                  onChange={(e) => setDetectionMode(e.target.value)}
                  disabled={isActive}
                  className="control-select"
                >
                  <option value="video">Video Only</option>
                  <option value="audio">Audio Only</option>
                  <option value="both">Video + Audio</option>
                </select>
              </div>
              
              <div className="control-actions">
                <button 
                  onClick={isActive ? stopDetection : startDetection}
                  className={`detection-toggle ${isActive ? 'active' : ''}`}
                  disabled={systemStatus === 'initializing'}
                >
                  {systemStatus === 'initializing' ? (
                    <>
                      <i className="fas fa-spinner fa-spin"></i>
                      Initializing...
                    </>
                  ) : isActive ? (
                    <>
                      <i className="fas fa-stop"></i>
                      Stop Detection
                    </>
                  ) : (
                    <>
                      <i className="fas fa-play"></i>
                      Start Detection
                    </>
                  )}
                </button>
                
                <div className="system-status">
                  <div 
                    className="status-indicator"
                    style={{ color: getStatusColor(systemStatus) }}
                  >
                    <div className="status-dot"></div>
                    <span>{getStatusText(systemStatus)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Video Feed */}
            <div className="video-container">
              <div className="video-wrapper">
                <video 
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="video-feed"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!isActive && (
                  <div className="video-placeholder">
                    <div className="placeholder-content">
                      <i className="fas fa-video placeholder-icon"></i>
                      <h3 className="placeholder-title">Video Feed Inactive</h3>
                      <p className="placeholder-text">
                        Click "Start Detection" to begin live analysis
                      </p>
                    </div>
                  </div>
                )}
                
                {isActive && (
                  <div className="video-overlay">
                    <div className="detection-info">
                      <div className="info-item">
                        <span className="info-label">Mode:</span>
                        <span className="info-value">{detectionMode.toUpperCase()}</span>
                      </div>
                      <div className="info-item">
                        <span className="info-label">Status:</span>
                        <span className="info-value recording">RECORDING</span>
                      </div>
                    </div>
                    
                    <div className="scan-lines"></div>
                  </div>
                )}
              </div>
            </div>

            {/* Real-time Score */}
            <div className="score-display">
              <div className="score-card">
                <div className="score-header">
                  <h3 className="score-title">Authenticity Score</h3>
                  <div 
                    className="risk-badge"
                    style={{ backgroundColor: getRiskColor(currentScore) }}
                  >
                    {getRiskLevel(currentScore)}
                  </div>
                </div>
                
                <div className="score-visualization">
                  <div className="score-meter">
                    <div 
                      className="score-fill"
                      style={{ 
                        width: `${currentScore}%`,
                        backgroundColor: getRiskColor(currentScore)
                      }}
                    ></div>
                  </div>
                  
                  <div className="score-value">
                    {currentScore.toFixed(1)}%
                  </div>
                </div>
                
                <div className="score-description">
                  {currentScore >= 80 && "Content appears authentic with high confidence"}
                  {currentScore >= 60 && currentScore < 80 && "Moderate confidence in authenticity"}
                  {currentScore >= 40 && currentScore < 60 && "Potential manipulation detected"}
                  {currentScore < 40 && "High probability of synthetic content"}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Sidebar */}
          <motion.div 
            className="detection-sidebar"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* Alerts */}
            <div className="alerts-section">
              <h4 className="section-title">
                <i className="fas fa-bell mr-2"></i>
                Live Alerts
              </h4>
              
              <div className="alerts-list">
                {alerts.length === 0 ? (
                  <div className="no-alerts">
                    <i className="fas fa-check-circle"></i>
                    <span>No alerts</span>
                  </div>
                ) : (
                  alerts.map(alert => (
                    <div key={alert.id} className={`alert-item ${alert.type}`}>
                      <div className="alert-icon">
                        <i className={
                          alert.type === 'error' ? 'fas fa-exclamation-triangle' :
                          alert.type === 'warning' ? 'fas fa-exclamation-circle' :
                          'fas fa-info-circle'
                        }></i>
                      </div>
                      <div className="alert-content">
                        <div className="alert-message">{alert.message}</div>
                        <div className="alert-time">
                          {alert.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Detection History */}
            <div className="history-section">
              <h4 className="section-title">
                <i className="fas fa-history mr-2"></i>
                Detection History
              </h4>
              
              <div className="history-list">
                {detectionHistory.length === 0 ? (
                  <div className="no-history">
                    <i className="fas fa-chart-line"></i>
                    <span>No detection data yet</span>
                  </div>
                ) : (
                  detectionHistory.slice(0, 10).map(entry => (
                    <div key={entry.id} className="history-item">
                      <div className="history-score">
                        <span 
                          className="score-dot"
                          style={{ backgroundColor: getRiskColor(entry.score) }}
                        ></span>
                        <span className="score-text">{entry.score.toFixed(1)}%</span>
                      </div>
                      
                      <div className="history-details">
                        <div className="history-risk">{getRiskLevel(entry.score)}</div>
                        <div className="history-time">
                          {entry.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* System Metrics */}
            <div className="metrics-section">
              <h4 className="section-title">
                <i className="fas fa-tachometer-alt mr-2"></i>
                System Metrics
              </h4>
              
              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-label">Processing Rate</div>
                  <div className="metric-value">
                    {isActive ? '15 FPS' : '0 FPS'}
                  </div>
                </div>
                
                <div className="metric-item">
                  <div className="metric-label">Latency</div>
                  <div className="metric-value">
                    {isActive ? '120ms' : 'N/A'}
                  </div>
                </div>
                
                <div className="metric-item">
                  <div className="metric-label">Frames Analyzed</div>
                  <div className="metric-value">
                    {isActive ? detectionHistory.length : 0}
                  </div>
                </div>
                
                <div className="metric-item">
                  <div className="metric-label">Average Score</div>
                  <div className="metric-value">
                    {detectionHistory.length > 0 
                      ? (detectionHistory.reduce((sum, entry) => sum + entry.score, 0) / detectionHistory.length).toFixed(1) + '%'
                      : 'N/A'
                    }
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default LiveDetectionPage;

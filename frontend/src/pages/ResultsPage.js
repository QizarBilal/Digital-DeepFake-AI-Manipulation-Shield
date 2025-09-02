import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './ResultsPage.css';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Get results from location state or fetch from API
    if (location.state?.results) {
      setResults(location.state.results);
      setIsLoading(false);
    } else {
      // Fetch recent results or redirect to analysis page
      fetchRecentResults();
    }
  }, [location.state]);

  const fetchRecentResults = async () => {
    try {
      const response = await fetch('/api/results/recent');
      if (response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          setResults(data[0]); // Show most recent result
        } else {
          navigate('/analysis'); // No results found, redirect to analysis
        }
      } else {
        navigate('/analysis');
      }
    } catch (error) {
      console.error('Failed to fetch results:', error);
      navigate('/analysis');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadReport = async (format = 'pdf') => {
    if (!results) return;

    try {
      const response = await fetch('/api/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          results: [results],
          format: format
        })
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deepfake-analysis-${results.file_id}-${Date.now()}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download report:', error);
    }
  };

  const shareResults = async () => {
    if (!results || !navigator.share) {
      // Fallback to copy to clipboard
      const shareText = `Deepfake Analysis Result: ${results.authenticity_score.toFixed(1)}% authenticity score for ${results.file_name}`;
      navigator.clipboard.writeText(shareText);
      return;
    }

    try {
      await navigator.share({
        title: 'Deepfake Analysis Results',
        text: `Analysis completed with ${results.authenticity_score.toFixed(1)}% authenticity score`,
        url: window.location.href
      });
    } catch (error) {
      console.error('Failed to share:', error);
    }
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return { level: 'Low Risk', color: '#48bb78', icon: 'fas fa-check-circle' };
    if (score >= 60) return { level: 'Medium Risk', color: '#ed8936', icon: 'fas fa-exclamation-circle' };
    if (score >= 40) return { level: 'High Risk', color: '#f56565', icon: 'fas fa-exclamation-triangle' };
    return { level: 'Critical Risk', color: '#e53e3e', icon: 'fas fa-times-circle' };
  };

  const getFileIcon = (type) => {
    switch (type) {
      case 'video': return 'fas fa-video';
      case 'audio': return 'fas fa-microphone';
      case 'image': return 'fas fa-image';
      default: return 'fas fa-file';
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (isLoading) {
    return (
      <div className="results-page">
        <div className="container">
          <div className="loading-container">
            <div className="loading-spinner">
              <i className="fas fa-spinner fa-spin"></i>
            </div>
            <p>Loading analysis results...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-page">
        <div className="container">
          <div className="no-results">
            <i className="fas fa-search"></i>
            <h2>No Results Found</h2>
            <p>Please analyze a file first to view results.</p>
            <button 
              onClick={() => navigate('/analysis')}
              className="btn btn-primary"
            >
              <i className="fas fa-upload mr-2"></i>
              Start Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }

  const risk = getRiskLevel(results.authenticity_score || 0);

  return (
    <div className="results-page">
      <div className="container">
        <motion.div 
          className="page-header"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="header-content">
            <h1 className="page-title">Analysis Results</h1>
            <p className="page-description">
              Comprehensive deepfake detection analysis for your uploaded content
            </p>
          </div>
          
          <div className="header-actions">
            <button 
              onClick={() => downloadReport('pdf')}
              className="btn btn-primary"
            >
              <i className="fas fa-download mr-2"></i>
              Download PDF
            </button>
            <button 
              onClick={() => downloadReport('csv')}
              className="btn btn-secondary"
            >
              <i className="fas fa-file-csv mr-2"></i>
              Export CSV
            </button>
            <button 
              onClick={shareResults}
              className="btn btn-secondary"
            >
              <i className="fas fa-share-alt mr-2"></i>
              Share
            </button>
          </div>
        </motion.div>

        <div className="results-layout">
          {/* Main Results */}
          <motion.div 
            className="results-main"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            {/* File Information */}
            <div className="file-info-card">
              <div className="file-header">
                <div className="file-icon-wrapper">
                  <i className={getFileIcon(results.file_type)}></i>
                </div>
                <div className="file-details">
                  <h2 className="file-name">{results.file_name}</h2>
                  <div className="file-meta">
                    <span className="meta-item">
                      <i className="fas fa-tag"></i>
                      {results.file_type?.toUpperCase()}
                    </span>
                    {results.file_size && (
                      <span className="meta-item">
                        <i className="fas fa-hdd"></i>
                        {formatFileSize(results.file_size)}
                      </span>
                    )}
                    <span className="meta-item">
                      <i className="fas fa-clock"></i>
                      {formatTimestamp(results.created_at)}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Authenticity Score */}
            <div className="score-card">
              <div className="score-header">
                <h3 className="score-title">Authenticity Assessment</h3>
                <div className="processing-time">
                  Processed in {results.processing_time || '2.3'}s
                </div>
              </div>
              
              <div className="score-display">
                <div className="score-circle">
                  <svg className="score-svg" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="#e2e8f0"
                      strokeWidth="8"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke={risk.color}
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray={`${(results.authenticity_score || 0) * 2.827} 282.7`}
                      transform="rotate(-90 50 50)"
                      className="score-progress"
                    />
                  </svg>
                  <div className="score-content">
                    <div className="score-value">
                      {(results.authenticity_score || 0).toFixed(1)}%
                    </div>
                    <div className="score-label">Authentic</div>
                  </div>
                </div>
                
                <div className="score-details">
                  <div className="risk-assessment">
                    <div className="risk-indicator">
                      <i className={risk.icon} style={{ color: risk.color }}></i>
                      <span className="risk-text" style={{ color: risk.color }}>
                        {risk.level}
                      </span>
                    </div>
                    
                    <div className="risk-description">
                      {results.authenticity_score >= 80 && "This content appears to be authentic with high confidence. No significant signs of manipulation detected."}
                      {results.authenticity_score >= 60 && results.authenticity_score < 80 && "Moderate confidence in authenticity. Some anomalies detected but within acceptable ranges."}
                      {results.authenticity_score >= 40 && results.authenticity_score < 60 && "Potential manipulation detected. Further analysis recommended."}
                      {results.authenticity_score < 40 && "High probability of synthetic or manipulated content. Exercise caution."}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Details */}
            <div className="analysis-card">
              <h3 className="card-title">
                <i className="fas fa-microscope mr-2"></i>
                Analysis Details
              </h3>
              
              {results.analysis_details && (
                <div className="details-grid">
                  {Object.entries(results.analysis_details).map(([key, value]) => (
                    <div key={key} className="detail-item">
                      <div className="detail-label">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </div>
                      <div className="detail-value">
                        {typeof value === 'number' ? 
                          (value < 1 ? (value * 100).toFixed(1) + '%' : value.toFixed(2)) : 
                          value
                        }
                      </div>
                      <div className="detail-bar">
                        <div 
                          className="detail-fill"
                          style={{ 
                            width: `${typeof value === 'number' ? 
                              (value < 1 ? value * 100 : Math.min(value, 100)) : 50}%`,
                            backgroundColor: typeof value === 'number' && value < 1 ? 
                              (value > 0.8 ? '#48bb78' : value > 0.6 ? '#ed8936' : '#f56565') : '#63b3ed'
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Detected Issues */}
            {results.detected_manipulations && results.detected_manipulations.length > 0 && (
              <div className="issues-card">
                <h3 className="card-title">
                  <i className="fas fa-exclamation-triangle mr-2"></i>
                  Detected Issues
                </h3>
                
                <div className="issues-list">
                  {results.detected_manipulations.map((issue, index) => (
                    <div key={index} className="issue-item">
                      <div className="issue-icon">
                        <i className="fas fa-exclamation-circle"></i>
                      </div>
                      <div className="issue-content">
                        <div className="issue-title">{issue}</div>
                        <div className="issue-description">
                          This anomaly was detected during the analysis process and may indicate synthetic content.
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>

          {/* Sidebar */}
          <motion.div 
            className="results-sidebar"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* Quick Actions */}
            <div className="quick-actions-card">
              <h4 className="card-title">Quick Actions</h4>
              
              <div className="actions-list">
                <button 
                  onClick={() => navigate('/analysis')}
                  className="action-btn"
                >
                  <i className="fas fa-plus-circle"></i>
                  <span>Analyze Another File</span>
                  <i className="fas fa-chevron-right"></i>
                </button>
                
                <button 
                  onClick={() => navigate('/dashboard')}
                  className="action-btn"
                >
                  <i className="fas fa-chart-bar"></i>
                  <span>View Dashboard</span>
                  <i className="fas fa-chevron-right"></i>
                </button>
                
                <button 
                  onClick={() => navigate('/live')}
                  className="action-btn"
                >
                  <i className="fas fa-video"></i>
                  <span>Live Detection</span>
                  <i className="fas fa-chevron-right"></i>
                </button>
              </div>
            </div>

            {/* Analysis Summary */}
            <div className="summary-card">
              <h4 className="card-title">Analysis Summary</h4>
              
              <div className="summary-stats">
                <div className="stat-item">
                  <div className="stat-icon">
                    <i className="fas fa-shield-alt"></i>
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">
                      {results.authenticity_score >= 80 ? 'Passed' : 'Failed'}
                    </div>
                    <div className="stat-label">Security Check</div>
                  </div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-icon">
                    <i className="fas fa-clock"></i>
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{results.processing_time || '2.3'}s</div>
                    <div className="stat-label">Processing Time</div>
                  </div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-icon">
                    <i className="fas fa-microchip"></i>
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">
                      {results.model_version || 'v2.1.0'}
                    </div>
                    <div className="stat-label">Model Version</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="recommendations-card">
              <h4 className="card-title">Recommendations</h4>
              
              <div className="recommendations-list">
                {results.authenticity_score >= 80 ? (
                  <>
                    <div className="recommendation-item positive">
                      <i className="fas fa-check-circle"></i>
                      <span>Content appears authentic and safe to use</span>
                    </div>
                    <div className="recommendation-item">
                      <i className="fas fa-info-circle"></i>
                      <span>Regular monitoring recommended for sensitive content</span>
                    </div>
                  </>
                ) : results.authenticity_score >= 60 ? (
                  <>
                    <div className="recommendation-item warning">
                      <i className="fas fa-exclamation-triangle"></i>
                      <span>Exercise caution when using this content</span>
                    </div>
                    <div className="recommendation-item">
                      <i className="fas fa-search"></i>
                      <span>Consider additional verification methods</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="recommendation-item negative">
                      <i className="fas fa-times-circle"></i>
                      <span>Do not use this content without verification</span>
                    </div>
                    <div className="recommendation-item">
                      <i className="fas fa-flag"></i>
                      <span>Report suspicious content to authorities</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;

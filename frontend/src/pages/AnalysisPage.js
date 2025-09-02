import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import './AnalysisPage.css';

const AnalysisPage = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [analysisType, setAnalysisType] = useState('auto');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState([]);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files);
      handleFiles(files);
    }
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      const files = Array.from(e.target.files);
      handleFiles(files);
    }
  };

  const handleFiles = (files) => {
    const validFiles = files.filter(file => {
      const validTypes = [
        'video/mp4', 'video/avi', 'video/mov', 'video/webm',
        'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/ogg',
        'image/jpeg', 'image/jpg', 'image/png', 'image/webp'
      ];
      return validTypes.includes(file.type) && file.size <= 100 * 1024 * 1024; // 100MB limit
    });

    setSelectedFiles(prev => [...prev, ...validFiles.map(file => ({
      file,
      id: Date.now() + Math.random(),
      type: getFileType(file.type),
      status: 'pending',
      progress: 0
    }))]);
  };

  const getFileType = (mimeType) => {
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType.startsWith('image/')) return 'image';
    return 'unknown';
  };

  const removeFile = (id) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== id));
  };

  const analyzeFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsAnalyzing(true);
    
    try {
      const analysisPromises = selectedFiles.map(async (fileObj) => {
        const formData = new FormData();
        formData.append('file', fileObj.file);

        // Update progress
        setSelectedFiles(prev => 
          prev.map(f => f.id === fileObj.id ? { ...f, status: 'analyzing', progress: 10 } : f)
        );

        const endpoint = `/api/upload/${fileObj.type}`;
        
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
          }

          const result = await response.json();
          
          // Update file status
          setSelectedFiles(prev => 
            prev.map(f => f.id === fileObj.id ? { ...f, status: 'completed', progress: 100 } : f)
          );

          return {
            fileId: fileObj.id,
            fileName: fileObj.file.name,
            fileType: fileObj.type,
            ...result
          };
        } catch (error) {
          console.error(`Failed to analyze ${fileObj.file.name}:`, error);
          
          setSelectedFiles(prev => 
            prev.map(f => f.id === fileObj.id ? { ...f, status: 'error', progress: 0 } : f)
          );
          
          return {
            fileId: fileObj.id,
            fileName: fileObj.file.name,
            fileType: fileObj.type,
            error: error.message,
            authenticity_score: 0,
            risk_level: 'error'
          };
        }
      });

      const analysisResults = await Promise.all(analysisPromises);
      setResults(analysisResults);
      
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadReport = async () => {
    try {
      const response = await fetch('/api/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          results: results,
          format: 'pdf'
        })
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deepfake-analysis-report-${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download report:', error);
    }
  };

  const getFileIcon = (type) => {
    switch (type) {
      case 'video': return 'fas fa-video';
      case 'audio': return 'fas fa-microphone';
      case 'image': return 'fas fa-image';
      default: return 'fas fa-file';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending': return '#718096';
      case 'analyzing': return '#ed8936';
      case 'completed': return '#48bb78';
      case 'error': return '#f56565';
      default: return '#718096';
    }
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return { level: 'Low Risk', color: '#48bb78' };
    if (score >= 60) return { level: 'Medium Risk', color: '#ed8936' };
    if (score >= 40) return { level: 'High Risk', color: '#f56565' };
    return { level: 'Critical Risk', color: '#e53e3e' };
  };

  return (
    <div className="analysis-page">
      <div className="container">
        <motion.div 
          className="page-header"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="page-title">Media Analysis</h1>
          <p className="page-description">
            Upload your video, audio, or image files for comprehensive deepfake detection analysis
          </p>
        </motion.div>

        <div className="analysis-container">
          {/* Upload Section */}
          <motion.div 
            className="upload-section"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <div className="analysis-controls">
              <div className="control-group">
                <label className="control-label">Analysis Type</label>
                <select 
                  value={analysisType} 
                  onChange={(e) => setAnalysisType(e.target.value)}
                  className="control-select"
                >
                  <option value="auto">Auto-detect</option>
                  <option value="comprehensive">Comprehensive Analysis</option>
                  <option value="fast">Fast Screening</option>
                  <option value="biometric">Biometric Focus</option>
                </select>
              </div>
              
              <div className="supported-formats">
                <span className="format-label">Supported formats:</span>
                <div className="format-tags">
                  <span className="format-tag">MP4</span>
                  <span className="format-tag">AVI</span>
                  <span className="format-tag">MP3</span>
                  <span className="format-tag">WAV</span>
                  <span className="format-tag">JPG</span>
                  <span className="format-tag">PNG</span>
                </div>
              </div>
            </div>

            <div 
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="upload-content">
                <i className="upload-icon fas fa-cloud-upload-alt"></i>
                <h3 className="upload-title">Drop files here or click to browse</h3>
                <p className="upload-subtitle">
                  Support for video, audio, and image files up to 100MB each
                </p>
                
                <input
                  type="file"
                  multiple
                  accept="video/*,audio/*,image/*"
                  onChange={handleFileInput}
                  className="file-input"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="btn btn-primary">
                  <i className="fas fa-folder-open mr-2"></i>
                  Select Files
                </label>
              </div>
            </div>
          </motion.div>

          {/* File List */}
          {selectedFiles.length > 0 && (
            <motion.div 
              className="file-list-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="section-header">
                <h3 className="section-title">Selected Files ({selectedFiles.length})</h3>
                <div className="section-actions">
                  <button 
                    onClick={analyzeFiles}
                    disabled={isAnalyzing}
                    className="btn btn-primary"
                  >
                    {isAnalyzing ? (
                      <>
                        <i className="fas fa-spinner fa-spin mr-2"></i>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-search mr-2"></i>
                        Analyze Files
                      </>
                    )}
                  </button>
                  
                  <button 
                    onClick={() => setSelectedFiles([])}
                    className="btn btn-secondary"
                    disabled={isAnalyzing}
                  >
                    <i className="fas fa-trash mr-2"></i>
                    Clear All
                  </button>
                </div>
              </div>

              <div className="file-list">
                {selectedFiles.map((fileObj) => (
                  <div key={fileObj.id} className="file-item">
                    <div className="file-info">
                      <div className="file-icon">
                        <i className={getFileIcon(fileObj.type)}></i>
                      </div>
                      
                      <div className="file-details">
                        <div className="file-name">{fileObj.file.name}</div>
                        <div className="file-meta">
                          {fileObj.type.toUpperCase()} • {(fileObj.file.size / 1024 / 1024).toFixed(2)} MB
                        </div>
                      </div>
                    </div>

                    <div className="file-status">
                      <div className="status-indicator">
                        <span 
                          className="status-text"
                          style={{ color: getStatusColor(fileObj.status) }}
                        >
                          {fileObj.status.toUpperCase()}
                        </span>
                        
                        {fileObj.status === 'analyzing' && (
                          <div className="progress-bar">
                            <div 
                              className="progress-fill"
                              style={{ width: `${fileObj.progress}%` }}
                            ></div>
                          </div>
                        )}
                      </div>

                      <button 
                        onClick={() => removeFile(fileObj.id)}
                        className="remove-btn"
                        disabled={isAnalyzing}
                      >
                        <i className="fas fa-times"></i>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Results Section */}
          {results.length > 0 && (
            <motion.div 
              className="results-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <div className="section-header">
                <h3 className="section-title">Analysis Results</h3>
                <button 
                  onClick={downloadReport}
                  className="btn btn-secondary"
                >
                  <i className="fas fa-download mr-2"></i>
                  Download Report
                </button>
              </div>

              <div className="results-grid">
                {results.map((result) => {
                  const risk = getRiskLevel(result.authenticity_score || 0);
                  
                  return (
                    <div key={result.fileId} className="result-card">
                      <div className="result-header">
                        <div className="result-file">
                          <i className={getFileIcon(result.fileType)}></i>
                          <span className="result-filename">{result.fileName}</span>
                        </div>
                        
                        <div className="result-score">
                          <div className="score-value">
                            {result.error ? 'Error' : `${result.authenticity_score?.toFixed(1)}%`}
                          </div>
                          <div className="score-label">Authenticity</div>
                        </div>
                      </div>

                      <div className="result-content">
                        {result.error ? (
                          <div className="error-message">
                            <i className="fas fa-exclamation-triangle"></i>
                            {result.error}
                          </div>
                        ) : (
                          <>
                            <div className="risk-indicator">
                              <div 
                                className="risk-badge"
                                style={{ backgroundColor: risk.color }}
                              >
                                {risk.level}
                              </div>
                            </div>

                            <div className="analysis-details">
                              {result.analysis_details && (
                                <div className="details-grid">
                                  {Object.entries(result.analysis_details).map(([key, value]) => (
                                    <div key={key} className="detail-item">
                                      <span className="detail-label">
                                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                                      </span>
                                      <span className="detail-value">
                                        {typeof value === 'number' ? value.toFixed(2) : value}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>

                            {result.detected_manipulations && result.detected_manipulations.length > 0 && (
                              <div className="manipulations">
                                <h4 className="manipulations-title">Detected Issues:</h4>
                                <ul className="manipulations-list">
                                  {result.detected_manipulations.map((manipulation, index) => (
                                    <li key={index} className="manipulation-item">
                                      <i className="fas fa-exclamation-circle"></i>
                                      {manipulation}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;

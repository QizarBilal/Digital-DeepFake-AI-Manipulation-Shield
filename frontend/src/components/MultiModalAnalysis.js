import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './MultiModalAnalysis.css';

const MultiModalAnalysis = () => {
  const [uploadedFiles, setUploadedFiles] = useState({
    image: null,
    video: null,
    audio: null
  });
  
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedDemo, setSelectedDemo] = useState(null);
  const [demoSamples, setDemoSamples] = useState(null);

  useEffect(() => {
    fetchDemoSamples();
  }, []);

  const fetchDemoSamples = async () => {
    try {
      const response = await fetch('http://localhost:8000/demo/samples');
      if (response.ok) {
        const data = await response.json();
        setDemoSamples(data);
      }
    } catch (error) {
      console.error('Failed to fetch demo samples:', error);
    }
  };

  const handleFileUpload = (modality, file) => {
    setUploadedFiles(prev => ({
      ...prev,
      [modality]: file
    }));
  };

  const handleAnalysis = async () => {
    if (!Object.values(uploadedFiles).some(file => file !== null)) {
      alert('Please upload at least one file for analysis');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    const formData = new FormData();
    
    if (uploadedFiles.image) formData.append('image', uploadedFiles.image);
    if (uploadedFiles.video) formData.append('video', uploadedFiles.video);
    if (uploadedFiles.audio) formData.append('audio', uploadedFiles.audio);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await fetch('http://localhost:8000/detect/multimodal', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      clearInterval(progressInterval);
      setAnalysisProgress(100);
      
      setTimeout(() => {
        setAnalysisResult(result);
        setIsAnalyzing(false);
        setAnalysisProgress(0);
      }, 500);

    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      alert('Analysis failed. Please try again.');
    }
  };

  const runDemoTestCase = async (testCaseId) => {
    setIsAnalyzing(true);
    setSelectedDemo(testCaseId);
    
    try {
      const response = await fetch(`http://localhost:8000/demo/test-case/${testCaseId}`, {
        method: 'POST'
      });
      
      const result = await response.json();
      
      setTimeout(() => {
        setAnalysisResult(result);
        setIsAnalyzing(false);
      }, 2000);
      
    } catch (error) {
      console.error('Demo test failed:', error);
      setIsAnalyzing(false);
      alert('Demo test failed. Please try again.');
    }
  };

  const clearAnalysis = () => {
    setAnalysisResult(null);
    setUploadedFiles({ image: null, video: null, audio: null });
    setSelectedDemo(null);
  };

  const FileUploadArea = ({ modality, icon, acceptedTypes, file }) => (
    <motion.div 
      className="upload-area"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="upload-header">
        <span className="upload-icon">{icon}</span>
        <h3>{modality.charAt(0).toUpperCase() + modality.slice(1)} Upload</h3>
      </div>
      
      {!file ? (
        <label className="upload-label">
          <input
            type="file"
            accept={acceptedTypes}
            onChange={(e) => handleFileUpload(modality, e.target.files[0])}
            hidden
          />
          <div className="upload-placeholder">
            <p>Click to upload {modality}</p>
            <span className="upload-formats">Supported: {acceptedTypes}</span>
          </div>
        </label>
      ) : (
        <div className="uploaded-file">
          <span className="file-name">{file.name}</span>
          <button 
            className="remove-file"
            onClick={() => handleFileUpload(modality, null)}
          >
            ×
          </button>
        </div>
      )}
    </motion.div>
  );

  const AnalysisProgress = () => (
    <motion.div 
      className="analysis-progress"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="progress-header">
        <h3>🔍 Analyzing Multi-Modal Content...</h3>
        <span className="progress-percentage">{analysisProgress}%</span>
      </div>
      
      <div className="progress-bar">
        <motion.div 
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${analysisProgress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
      
      <div className="progress-steps">
        <div className={`step ${analysisProgress >= 25 ? 'completed' : ''}`}>
          📷 Image Analysis
        </div>
        <div className={`step ${analysisProgress >= 50 ? 'completed' : ''}`}>
          🎥 Video Analysis  
        </div>
        <div className={`step ${analysisProgress >= 75 ? 'completed' : ''}`}>
          🎵 Audio Analysis
        </div>
        <div className={`step ${analysisProgress >= 100 ? 'completed' : ''}`}>
          🔄 Fusion & Results
        </div>
      </div>
    </motion.div>
  );

  const ResultsDisplay = ({ result }) => {
    const confidenceColor = result.confidence > 80 ? '#00FFF7' : 
                           result.confidence > 60 ? '#FFD700' : '#FF6B6B';
    
    const riskColor = result.risk_level === 'High' ? '#FF6B6B' :
                     result.risk_level === 'Medium' ? '#FFD700' : '#00FFF7';

    return (
      <motion.div 
        className="results-display"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="results-header">
          <h2>🎯 Analysis Results</h2>
          <button className="clear-button" onClick={clearAnalysis}>
            Clear Results
          </button>
        </div>

        <div className="main-result">
          <div className={`prediction ${result.prediction.toLowerCase().replace(/[^a-z]/g, '')}`}>
            <span className="prediction-icon">
              {result.prediction === 'Real' ? '✅' : '⚠️'}
            </span>
            <h3>{result.prediction}</h3>
          </div>
          
          <div className="confidence-circle">
            <div 
              className="confidence-fill"
              style={{ 
                background: `conic-gradient(${confidenceColor} ${result.confidence * 3.6}deg, #2a2a2a 0deg)`
              }}
            >
              <div className="confidence-inner">
                <span className="confidence-value">{result.confidence.toFixed(1)}%</span>
                <span className="confidence-label">Confidence</span>
              </div>
            </div>
          </div>
        </div>

        <div className="analysis-details">
          <div className="detail-card">
            <h4>🎯 Risk Level</h4>
            <span className="risk-badge" style={{ color: riskColor }}>
              {result.risk_level}
            </span>
          </div>

          {result.fusion_type === 'multi_modal' && (
            <div className="detail-card">
              <h4>🔄 Fusion Strategy</h4>
              <span>{result.fusion_strategy?.replace(/_/g, ' ').toUpperCase()}</span>
            </div>
          )}

          {result.agreement_score && (
            <div className="detail-card">
              <h4>🤝 Modality Agreement</h4>
              <span>{(result.agreement_score * 100).toFixed(1)}%</span>
            </div>
          )}

          {result.available_modalities && (
            <div className="detail-card">
              <h4>📊 Modalities Analyzed</h4>
              <div className="modality-tags">
                {result.available_modalities.map(modality => (
                  <span key={modality} className="modality-tag">
                    {modality}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {result.modality_results && (
          <div className="modality-breakdown">
            <h4>🔍 Individual Modality Results</h4>
            <div className="modality-grid">
              {Object.entries(result.modality_results).map(([modality, modalResult]) => (
                <div key={modality} className="modality-result">
                  <div className="modality-header">
                    <span className="modality-icon">
                      {modality === 'image' ? '📷' : 
                       modality === 'video' ? '🎥' : '🎵'}
                    </span>
                    <span className="modality-name">{modality}</span>
                  </div>
                  
                  <div className="modality-prediction">
                    <span className={`prediction-badge ${modalResult.prediction?.toLowerCase().replace(/[^a-z]/g, '')}`}>
                      {modalResult.prediction}
                    </span>
                    <span className="confidence-small">
                      {modalResult.confidence?.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.demo_mode && (
          <div className="demo-info">
            <h4>🎪 Demo Test Case: {result.test_name}</h4>
            <p>{result.description}</p>
            <p><strong>Focus:</strong> {result.test_focus}</p>
          </div>
        )}
      </motion.div>
    );
  };

  const DemoSection = () => (
    <div className="demo-section">
      <h3>🎪 Try Demo Test Cases</h3>
      <p>Experience the detection system with pre-configured samples</p>
      
      <div className="demo-grid">
        {[
          { id: 'TC001', name: 'Authentic Content', desc: 'Real person across all modalities' },
          { id: 'TC002', name: 'Full AI Content', desc: 'AI-generated across all modalities' },
          { id: 'TC003', name: 'Mixed Content', desc: 'Real image with synthetic audio' },
          { id: 'TC004', name: 'Sophisticated Deepfake', desc: 'High-quality face swap' }
        ].map(demo => (
          <motion.button
            key={demo.id}
            className="demo-card"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => runDemoTestCase(demo.id)}
            disabled={isAnalyzing}
          >
            <h4>{demo.name}</h4>
            <p>{demo.desc}</p>
          </motion.button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="multimodal-analysis">
      <motion.div 
        className="analysis-container"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="analysis-header">
          <h1>🎯 Multi-Modal Deepfake Detection</h1>
          <p>Upload image, video, and audio files for comprehensive analysis</p>
        </div>

        {!analysisResult && !isAnalyzing && (
          <>
            <div className="upload-grid">
              <FileUploadArea 
                modality="image"
                icon="📷"
                acceptedTypes="image/*"
                file={uploadedFiles.image}
              />
              <FileUploadArea 
                modality="video"
                icon="🎥"
                acceptedTypes="video/*"
                file={uploadedFiles.video}
              />
              <FileUploadArea 
                modality="audio"
                icon="🎵"
                acceptedTypes="audio/*"
                file={uploadedFiles.audio}
              />
            </div>

            <motion.button
              className="analyze-button"
              onClick={handleAnalysis}
              disabled={!Object.values(uploadedFiles).some(file => file !== null)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              🔍 Start Multi-Modal Analysis
            </motion.button>

            <DemoSection />
          </>
        )}

        <AnimatePresence mode="wait">
          {isAnalyzing && (
            <AnalysisProgress key="progress" />
          )}
          
          {analysisResult && !isAnalyzing && (
            <ResultsDisplay key="results" result={analysisResult} />
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default MultiModalAnalysis;

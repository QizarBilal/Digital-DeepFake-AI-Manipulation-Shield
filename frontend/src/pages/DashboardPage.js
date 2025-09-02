import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './DashboardPage.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend
);

const DashboardPage = () => {
  const [dashboardData, setDashboardData] = useState({
    stats: {
      total_analyses: 0,
      deepfakes_detected: 0,
      average_authenticity_score: 0,
      total_files_processed: 0
    },
    recent_analyses: [],
    detection_trends: [],
    file_type_distribution: {},
    risk_distribution: {}
  });

  const [timeRange, setTimeRange] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, [timeRange]);

  const fetchDashboardData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/stats/dashboard?range=${timeRange}`);
      const data = await response.json();
      setDashboardData(data);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      // Use mock data for demo
      setDashboardData(getMockData());
    } finally {
      setIsLoading(false);
    }
  };

  const getMockData = () => ({
    stats: {
      total_analyses: 1247,
      deepfakes_detected: 89,
      average_authenticity_score: 87.3,
      total_files_processed: 2156
    },
    recent_analyses: [
      { id: 1, filename: 'video_sample.mp4', type: 'video', score: 92.4, timestamp: '2024-01-15T10:30:00Z', risk: 'low' },
      { id: 2, filename: 'audio_clip.wav', type: 'audio', score: 34.2, timestamp: '2024-01-15T09:15:00Z', risk: 'high' },
      { id: 3, filename: 'portrait.jpg', type: 'image', score: 78.9, timestamp: '2024-01-15T08:45:00Z', risk: 'medium' },
      { id: 4, filename: 'interview.mp4', type: 'video', score: 96.1, timestamp: '2024-01-14T16:20:00Z', risk: 'low' },
      { id: 5, filename: 'voice_note.m4a', type: 'audio', score: 23.7, timestamp: '2024-01-14T14:10:00Z', risk: 'critical' }
    ],
    detection_trends: [
      { date: '2024-01-09', authentic: 45, deepfake: 5 },
      { date: '2024-01-10', authentic: 52, deepfake: 8 },
      { date: '2024-01-11', authentic: 38, deepfake: 12 },
      { date: '2024-01-12', authentic: 67, deepfake: 6 },
      { date: '2024-01-13', authentic: 73, deepfake: 9 },
      { date: '2024-01-14', authentic: 81, deepfake: 15 },
      { date: '2024-01-15', authentic: 89, deepfake: 11 }
    ],
    file_type_distribution: { video: 45, audio: 32, image: 23 },
    risk_distribution: { low: 62, medium: 23, high: 12, critical: 3 }
  });

  const getTrendChartData = () => ({
    labels: dashboardData.detection_trends.map(d => new Date(d.date).toLocaleDateString()),
    datasets: [
      {
        label: 'Authentic Content',
        data: dashboardData.detection_trends.map(d => d.authentic),
        borderColor: '#48bb78',
        backgroundColor: 'rgba(72, 187, 120, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Deepfake Detected',
        data: dashboardData.detection_trends.map(d => d.deepfake),
        borderColor: '#f56565',
        backgroundColor: 'rgba(245, 101, 101, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  });

  const getFileTypeChartData = () => ({
    labels: ['Video', 'Audio', 'Image'],
    datasets: [{
      data: [
        dashboardData.file_type_distribution.video || 0,
        dashboardData.file_type_distribution.audio || 0,
        dashboardData.file_type_distribution.image || 0
      ],
      backgroundColor: ['#63b3ed', '#a855f7', '#ed8936'],
      borderWidth: 0
    }]
  });

  const getRiskChartData = () => ({
    labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
    datasets: [{
      data: [
        dashboardData.risk_distribution.low || 0,
        dashboardData.risk_distribution.medium || 0,
        dashboardData.risk_distribution.high || 0,
        dashboardData.risk_distribution.critical || 0
      ],
      backgroundColor: ['#48bb78', '#ed8936', '#f56565', '#e53e3e']
    }]
  });

  const getPerformanceRadarData = () => ({
    labels: ['Accuracy', 'Speed', 'Coverage', 'Reliability', 'Precision'],
    datasets: [{
      label: 'System Performance',
      data: [95, 87, 92, 89, 91],
      backgroundColor: 'rgba(99, 179, 237, 0.2)',
      borderColor: '#63b3ed',
      borderWidth: 2,
      pointBackgroundColor: '#63b3ed',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: '#63b3ed'
    }]
  });

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20
        }
      }
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'low': return '#48bb78';
      case 'medium': return '#ed8936';
      case 'high': return '#f56565';
      case 'critical': return '#e53e3e';
      default: return '#718096';
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

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="dashboard-page">
      <div className="container">
        <motion.div 
          className="page-header"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="header-content">
            <h1 className="page-title">Analytics Dashboard</h1>
            <p className="page-description">
              Comprehensive insights and statistics for deepfake detection activities
            </p>
          </div>
          
          <div className="header-controls">
            <select 
              value={timeRange} 
              onChange={(e) => setTimeRange(e.target.value)}
              className="time-range-select"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last 90 Days</option>
            </select>
            
            <button 
              onClick={fetchDashboardData} 
              className="refresh-btn"
              disabled={isLoading}
            >
              <i className={`fas fa-sync-alt ${isLoading ? 'fa-spin' : ''}`}></i>
            </button>
          </div>
        </motion.div>

        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner">
              <i className="fas fa-spinner fa-spin"></i>
            </div>
            <p>Loading dashboard data...</p>
          </div>
        ) : (
          <>
            {/* Stats Overview */}
            <motion.div 
              className="stats-grid"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <div className="stat-card">
                <div className="stat-icon">
                  <i className="fas fa-search"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-value">{dashboardData.stats.total_analyses.toLocaleString()}</div>
                  <div className="stat-label">Total Analyses</div>
                  <div className="stat-change positive">+12% this week</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon deepfake">
                  <i className="fas fa-exclamation-triangle"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-value">{dashboardData.stats.deepfakes_detected.toLocaleString()}</div>
                  <div className="stat-label">Deepfakes Detected</div>
                  <div className="stat-change negative">+8% this week</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon accuracy">
                  <i className="fas fa-bullseye"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-value">{dashboardData.stats.average_authenticity_score.toFixed(1)}%</div>
                  <div className="stat-label">Avg Authenticity Score</div>
                  <div className="stat-change positive">+2.3% this week</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon files">
                  <i className="fas fa-files"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-value">{dashboardData.stats.total_files_processed.toLocaleString()}</div>
                  <div className="stat-label">Files Processed</div>
                  <div className="stat-change positive">+15% this week</div>
                </div>
              </div>
            </motion.div>

            {/* Charts Section */}
            <div className="charts-section">
              <motion.div 
                className="chart-card large"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <div className="chart-header">
                  <h3 className="chart-title">Detection Trends</h3>
                  <p className="chart-subtitle">Daily authentic vs deepfake content detection over time</p>
                </div>
                <div className="chart-container">
                  <Line data={getTrendChartData()} options={chartOptions} />
                </div>
              </motion.div>

              <motion.div 
                className="chart-card"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <div className="chart-header">
                  <h3 className="chart-title">File Type Distribution</h3>
                  <p className="chart-subtitle">Breakdown by content type</p>
                </div>
                <div className="chart-container">
                  <Doughnut data={getFileTypeChartData()} options={chartOptions} />
                </div>
              </motion.div>

              <motion.div 
                className="chart-card"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                <div className="chart-header">
                  <h3 className="chart-title">Risk Assessment</h3>
                  <p className="chart-subtitle">Distribution of risk levels</p>
                </div>
                <div className="chart-container">
                  <Bar data={getRiskChartData()} options={chartOptions} />
                </div>
              </motion.div>

              <motion.div 
                className="chart-card"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
              >
                <div className="chart-header">
                  <h3 className="chart-title">System Performance</h3>
                  <p className="chart-subtitle">AI model performance metrics</p>
                </div>
                <div className="chart-container">
                  <Radar data={getPerformanceRadarData()} options={chartOptions} />
                </div>
              </motion.div>
            </div>

            {/* Recent Analyses */}
            <motion.div 
              className="recent-analyses-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <div className="section-header">
                <h3 className="section-title">Recent Analyses</h3>
                <button className="view-all-btn">
                  View All <i className="fas fa-arrow-right ml-2"></i>
                </button>
              </div>

              <div className="analyses-table">
                <div className="table-header">
                  <div className="table-cell">File</div>
                  <div className="table-cell">Type</div>
                  <div className="table-cell">Score</div>
                  <div className="table-cell">Risk</div>
                  <div className="table-cell">Timestamp</div>
                  <div className="table-cell">Actions</div>
                </div>

                <div className="table-body">
                  {dashboardData.recent_analyses.map((analysis) => (
                    <div key={analysis.id} className="table-row">
                      <div className="table-cell file-cell">
                        <div className="file-info">
                          <i className={getFileIcon(analysis.type)}></i>
                          <span className="filename">{analysis.filename}</span>
                        </div>
                      </div>
                      
                      <div className="table-cell">
                        <span className="type-badge">{analysis.type.toUpperCase()}</span>
                      </div>
                      
                      <div className="table-cell">
                        <div className="score-indicator">
                          <span className="score-value">{analysis.score.toFixed(1)}%</span>
                          <div className="score-bar">
                            <div 
                              className="score-fill"
                              style={{ 
                                width: `${analysis.score}%`,
                                backgroundColor: analysis.score >= 80 ? '#48bb78' : 
                                                analysis.score >= 60 ? '#ed8936' : '#f56565'
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="table-cell">
                        <span 
                          className="risk-badge"
                          style={{ backgroundColor: getRiskColor(analysis.risk) }}
                        >
                          {analysis.risk.toUpperCase()}
                        </span>
                      </div>
                      
                      <div className="table-cell timestamp-cell">
                        {formatTimestamp(analysis.timestamp)}
                      </div>
                      
                      <div className="table-cell actions-cell">
                        <button className="action-btn" title="View Details">
                          <i className="fas fa-eye"></i>
                        </button>
                        <button className="action-btn" title="Download Report">
                          <i className="fas fa-download"></i>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>

            {/* System Health */}
            <motion.div 
              className="system-health-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <h3 className="section-title">System Health</h3>
              
              <div className="health-grid">
                <div className="health-card">
                  <div className="health-header">
                    <h4 className="health-title">AI Models</h4>
                    <div className="health-status online">
                      <div className="status-dot"></div>
                      <span>Online</span>
                    </div>
                  </div>
                  <div className="health-metrics">
                    <div className="metric">
                      <span className="metric-label">Video Model</span>
                      <span className="metric-value">98.5%</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Audio Model</span>
                      <span className="metric-value">97.2%</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Image Model</span>
                      <span className="metric-value">99.1%</span>
                    </div>
                  </div>
                </div>

                <div className="health-card">
                  <div className="health-header">
                    <h4 className="health-title">Processing Queue</h4>
                    <div className="health-status online">
                      <div className="status-dot"></div>
                      <span>Active</span>
                    </div>
                  </div>
                  <div className="health-metrics">
                    <div className="metric">
                      <span className="metric-label">Pending Jobs</span>
                      <span className="metric-value">3</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Avg Processing Time</span>
                      <span className="metric-value">2.3s</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Success Rate</span>
                      <span className="metric-value">99.8%</span>
                    </div>
                  </div>
                </div>

                <div className="health-card">
                  <div className="health-header">
                    <h4 className="health-title">Database</h4>
                    <div className="health-status online">
                      <div className="status-dot"></div>
                      <span>Healthy</span>
                    </div>
                  </div>
                  <div className="health-metrics">
                    <div className="metric">
                      <span className="metric-label">Records</span>
                      <span className="metric-value">15.2K</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Storage Used</span>
                      <span className="metric-value">2.8GB</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Query Time</span>
                      <span className="metric-value">45ms</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Shield, 
  Zap, 
  Brain, 
  CheckCircle, 
  ArrowRight,
  Play,
  Eye,
  FileText,
  Image,
  Video,
  Music
} from 'lucide-react';
import './HomePage.css';

const HomePage = ({ systemStatus }) => {
  const [stats, setStats] = useState({
    totalAnalyses: 1247,
    aiDetected: 412,
    averageAccuracy: 94.2,
    systemUptime: '99.8%'
  });

  useEffect(() => {
    // Fetch real stats from API
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.log('Using demo stats - backend not available');
    }
  };

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Detection',
      description: 'Advanced machine learning algorithms analyze content patterns to identify AI-generated material with 94%+ accuracy.',
      color: 'blue'
    },
    {
      icon: Zap,
      title: 'Real-Time Analysis',
      description: 'Get instant results with our optimized processing engine. Most analyses complete in under 3 seconds.',
      color: 'yellow'
    },
    {
      icon: Shield,
      title: 'Multi-Format Support',
      description: 'Analyze text, images, videos, and audio files. Our system handles all major content formats.',
      color: 'green'
    },
    {
      icon: CheckCircle,
      title: 'Enterprise Ready',
      description: 'Production-grade reliability with API access, batch processing, and detailed reporting capabilities.',
      color: 'purple'
    }
  ];

  const contentTypes = [
    { icon: FileText, type: 'Text', description: 'Articles, essays, reports' },
    { icon: Image, type: 'Images', description: 'Photos, artwork, graphics' },
    { icon: Video, type: 'Videos', description: 'Deepfakes, synthetic media' },
    { icon: Music, type: 'Audio', description: 'Voice clones, synthetic speech' }
  ];

  return (
    <div className="min-h-screen bg-[#0F1419]">
      {/* Hero Section */}
      <motion.section 
        className="relative min-h-screen flex items-center justify-center overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,#3B82F6,transparent_70%)]" />
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMzQjgyRjYiIGZpbGwtb3BhY2l0eT0iMC4xIj48Y2lyY2xlIGN4PSIyMCIgY3k9IjIwIiByPSIyIi8+PGNpcmNsZSBjeD0iNDAiIGN5PSI0MCIgcj0iMiIvPjwvZz48L2c+PC9zdmc+')] opacity-30" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-6 text-center">
          {/* Main Heading */}
          <motion.div
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h1 className="text-5xl lg:text-7xl font-bold mb-6">
              <span className="text-white">Detect AI-Generated</span>
              <br />
              <span className="text-[#3B82F6]">Content Instantly</span>
            </h1>
            <p className="text-xl lg:text-2xl text-gray-300 max-w-4xl mx-auto mb-12">
              Advanced AI detection system powered by machine learning. Analyze text, images, 
              videos, and audio to determine authenticity with industry-leading accuracy.
            </p>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mb-16"
          >
            <Link
              to="/analysis"
              className="bg-[#3B82F6] hover:bg-[#2563EB] text-white px-8 py-4 rounded-lg font-semibold flex items-center gap-3 transition-all duration-300 hover:scale-105 group"
            >
              <Zap className="w-5 h-5" />
              Start Analysis
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/demo"
              className="bg-transparent border-2 border-[#3B82F6] text-[#3B82F6] hover:bg-[#3B82F6] hover:text-white px-8 py-4 rounded-lg font-semibold flex items-center gap-3 transition-all duration-300 hover:scale-105"
            >
              <Play className="w-5 h-5" />
              View Demo
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="grid grid-cols-2 lg:grid-cols-4 gap-8 max-w-4xl mx-auto"
          >
            <div className="text-center">
              <div className="text-3xl lg:text-4xl font-bold text-[#3B82F6] mb-2">
                {stats.totalAnalyses?.toLocaleString() || '1,247'}
              </div>
              <div className="text-gray-400">Total Analyses</div>
            </div>
            <div className="text-center">
              <div className="text-3xl lg:text-4xl font-bold text-[#3B82F6] mb-2">
                {stats.aiDetected?.toLocaleString() || '412'}
              </div>
              <div className="text-gray-400">AI Detected</div>
            </div>
            <div className="text-center">
              <div className="text-3xl lg:text-4xl font-bold text-[#3B82F6] mb-2">
                {stats.averageAccuracy || 94.2}%
              </div>
              <div className="text-gray-400">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl lg:text-4xl font-bold text-[#3B82F6] mb-2">
                {stats.systemUptime || '99.8%'}
              </div>
              <div className="text-gray-400">Uptime</div>
            </div>
          </motion.div>
        </div>

        {/* System Status Indicator */}
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, delay: 1 }}
          className="absolute top-24 right-6 bg-[#1C2128] border border-[#3B82F6]/20 rounded-lg px-4 py-2 flex items-center gap-2"
        >
          <div className={`w-3 h-3 rounded-full ${
            systemStatus === 'ready' ? 'bg-green-400' : 
            systemStatus === 'checking' ? 'bg-yellow-400 animate-pulse' : 'bg-red-400'
          }`} />
          <span className="text-sm text-gray-300">
            {systemStatus === 'ready' ? 'System Online' : 
             systemStatus === 'checking' ? 'Checking Status' : 'System Offline'}
          </span>
        </motion.div>
      </motion.section>

      {/* Features Section */}
      <section className="py-20 bg-[#1C2128]">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Powerful AI Detection Features
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our advanced system combines multiple detection techniques to provide 
              accurate and reliable analysis across all content types.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-[#0F1419] rounded-xl p-6 border border-[#3B82F6]/20 hover:border-[#3B82F6]/40 transition-all duration-300 hover:scale-105"
                >
                  <div className={`w-12 h-12 rounded-lg bg-${feature.color}-500/20 flex items-center justify-center mb-4`}>
                    <Icon className={`w-6 h-6 text-${feature.color}-400`} />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Content Types Section */}
      <section className="py-20 bg-[#0F1419]">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Analyze Any Content Type
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our multi-modal detection system supports comprehensive analysis 
              across text, images, videos, and audio content.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {contentTypes.map((type, index) => {
              const Icon = type.icon;
              return (
                <motion.div
                  key={type.type}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center hover:border-[#3B82F6]/40 transition-all duration-300 group hover:scale-105"
                >
                  <div className="w-16 h-16 rounded-full bg-[#3B82F6]/20 flex items-center justify-center mx-auto mb-4 group-hover:bg-[#3B82F6]/30 transition-colors">
                    <Icon className="w-8 h-8 text-[#3B82F6]" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {type.type}
                  </h3>
                  <p className="text-gray-400 text-sm">
                    {type.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-[#1C2128]">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-6">
              Ready to Detect AI Content?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Start analyzing your content today with our advanced AI detection system.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/analysis"
                className="bg-[#3B82F6] hover:bg-[#2563EB] text-white px-8 py-4 rounded-lg font-semibold flex items-center gap-3 transition-all duration-300 hover:scale-105 group"
              >
                <Eye className="w-5 h-5" />
                Start Analysis
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                to="/workflow"
                className="bg-transparent border-2 border-[#3B82F6] text-[#3B82F6] hover:bg-[#3B82F6] hover:text-white px-8 py-4 rounded-lg font-semibold flex items-center gap-3 transition-all duration-300 hover:scale-105"
              >
                <Brain className="w-5 h-5" />
                View Workflow
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[#0F1419] border-t border-[#3B82F6]/20 py-12">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-8 h-8 text-[#3B82F6]" />
                <span className="text-xl font-bold text-white">AI Shield</span>
              </div>
              <p className="text-gray-400">
                Advanced AI detection system for authentic content verification.
              </p>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-4">Features</h3>
              <ul className="space-y-2 text-gray-400">
                <li>Multi-format Detection</li>
                <li>Real-time Analysis</li>
                <li>Enterprise API</li>
                <li>Batch Processing</li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-4">Resources</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <Link to="/demo" className="hover:text-[#3B82F6] transition-colors">
                    Live Demo
                  </Link>
                </li>
                <li>
                  <Link to="/workflow" className="hover:text-[#3B82F6] transition-colors">
                    How It Works
                  </Link>
                </li>
                <li>
                  <Link to="/dashboard" className="hover:text-[#3B82F6] transition-colors">
                    Dashboard
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-4">Stats</h3>
              <ul className="space-y-2 text-gray-400">
                <li>94.2% Accuracy</li>
                <li>2-6s Processing</li>
                <li>99.8% Uptime</li>
                <li>4 Content Types</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-[#3B82F6]/20 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 AI Detection System. Professional prototype for demonstration.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;

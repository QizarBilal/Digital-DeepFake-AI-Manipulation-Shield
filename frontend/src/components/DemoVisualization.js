import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  Image, 
  Video, 
  Music, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  BarChart3,
  Download,
  Zap,
  Shield,
  Brain,
  Lightbulb
} from 'lucide-react';

const DemoVisualization = () => {
  const [selectedDemo, setSelectedDemo] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  const demoSamples = [
    {
      id: 'authentic_text',
      type: 'text',
      title: 'Authentic Human Text',
      description: 'Personal travel experience with natural language patterns',
      icon: FileText,
      color: 'blue',
      preview: 'I remember when I first visited Paris last summer...'
    },
    {
      id: 'ai_text',
      type: 'text',
      title: 'AI-Generated Text',
      description: 'Technical content with AI language patterns',
      icon: FileText,
      color: 'red',
      preview: 'Artificial intelligence has revolutionized numerous sectors...'
    },
    {
      id: 'authentic_image',
      type: 'image',
      title: 'Authentic Image',
      description: 'Natural photograph with original metadata',
      icon: Image,
      color: 'green',
      preview: 'Natural lighting and composition'
    },
    {
      id: 'ai_image',
      type: 'image',
      title: 'AI-Generated Image',
      description: 'Synthetic image created by generative AI',
      icon: Image,
      color: 'orange',
      preview: 'Perfect details with subtle AI artifacts'
    }
  ];

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-green-400';
    if (confidence >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getPredictionIcon = (prediction) => {
    switch (prediction) {
      case 'authentic':
        return <CheckCircle className="w-6 h-6 text-green-400" />;
      case 'ai_generated':
        return <XCircle className="w-6 h-6 text-red-400" />;
      case 'ai_modified':
        return <AlertTriangle className="w-6 h-6 text-yellow-400" />;
      default:
        return <Brain className="w-6 h-6 text-blue-400" />;
    }
  };

  const getPredictionLabel = (prediction) => {
    switch (prediction) {
      case 'authentic':
        return 'Human Created';
      case 'ai_generated':
        return 'AI Generated';
      case 'ai_modified':
        return 'AI Modified';
      default:
        return 'Unknown';
    }
  };

  const runDemoAnalysis = async (demoId) => {
    setIsAnalyzing(true);
    setAnalysisResult(null);
    
    try {
      const response = await fetch(`http://localhost:8000/api/demo/${demoId}`);
      if (!response.ok) throw new Error('Analysis failed');
      
      const result = await response.json();
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setAnalysisResult(result);
    } catch (error) {
      console.error('Demo analysis error:', error);
      setAnalysisResult({
        error: 'Demo analysis failed. Please ensure the backend is running.'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const AnalysisMetrics = ({ analysis, type }) => {
    if (!analysis) return null;

    return (
      <div className="grid grid-cols-2 gap-4 mt-6">
        {Object.entries(analysis).map(([key, value], index) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-[#1C2128] rounded-lg p-4 border border-[#3B82F6]/20"
          >
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-300 capitalize">
                {key.replace(/_/g, ' ')}
              </span>
              <span className={`text-sm font-semibold ${getConfidenceColor(value)}`}>
                {value.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ duration: 1, delay: index * 0.1 }}
                className={`h-2 rounded-full ${
                  value >= 70 ? 'bg-green-400' : 
                  value >= 40 ? 'bg-yellow-400' : 'bg-red-400'
                }`}
              />
            </div>
          </motion.div>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#0F1419] pt-20">
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-white mb-4">
            Interactive AI Detection Demo
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Experience our advanced AI detection system with real examples. 
            See how we analyze different content types to determine authenticity.
          </p>
        </motion.div>

        {/* Demo Samples Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {demoSamples.map((sample, index) => {
            const Icon = sample.icon;
            return (
              <motion.div
                key={sample.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`bg-[#1C2128] rounded-xl p-6 border cursor-pointer transition-all duration-300 hover:scale-105 ${
                  selectedDemo === sample.id
                    ? 'border-[#3B82F6] bg-[#3B82F6]/10'
                    : 'border-[#3B82F6]/20 hover:border-[#3B82F6]/40'
                }`}
                onClick={() => setSelectedDemo(sample.id)}
              >
                <div className={`w-12 h-12 rounded-lg bg-${sample.color}-500/20 flex items-center justify-center mb-4`}>
                  <Icon className={`w-6 h-6 text-${sample.color}-400`} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {sample.title}
                </h3>
                <p className="text-sm text-gray-400 mb-3">
                  {sample.description}
                </p>
                <p className="text-xs text-gray-500 italic">
                  "{sample.preview}"
                </p>
              </motion.div>
            );
          })}
        </div>

        {/* Analysis Controls */}
        <div className="text-center mb-12">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => selectedDemo && runDemoAnalysis(selectedDemo)}
            disabled={!selectedDemo || isAnalyzing}
            className="bg-[#3B82F6] hover:bg-[#2563EB] disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-semibold flex items-center gap-2 mx-auto transition-colors"
          >
            {isAnalyzing ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                />
                Analyzing...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Analyze Selected Sample
              </>
            )}
          </motion.button>
          {!selectedDemo && (
            <p className="text-gray-400 text-sm mt-2">
              Select a demo sample above to run analysis
            </p>
          )}
        </div>

        {/* Analysis Results */}
        <AnimatePresence>
          {analysisResult && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-[#1C2128] rounded-xl p-8 border border-[#3B82F6]/20"
            >
              {analysisResult.error ? (
                <div className="text-center">
                  <XCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">Analysis Error</h3>
                  <p className="text-red-400">{analysisResult.error}</p>
                </div>
              ) : (
                <>
                  {/* Result Header */}
                  <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                      {getPredictionIcon(analysisResult.prediction)}
                      <div>
                        <h3 className="text-2xl font-bold text-white">
                          {getPredictionLabel(analysisResult.prediction)}
                        </h3>
                        <p className="text-gray-400">
                          Content Type: {analysisResult.type.toUpperCase()}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-400">Confidence Score</p>
                      <p className={`text-3xl font-bold ${getConfidenceColor(analysisResult.confidence)}`}>
                        {analysisResult.confidence}%
                      </p>
                    </div>
                  </div>

                  {/* Confidence Meter */}
                  <div className="mb-8">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-400">Confidence Level</span>
                      <span className="text-sm text-gray-400">
                        Processing Time: {analysisResult.processing_time}s
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-4">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${analysisResult.confidence}%` }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                        className={`h-4 rounded-full ${
                          analysisResult.confidence >= 90 ? 'bg-green-400' :
                          analysisResult.confidence >= 70 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                      />
                    </div>
                  </div>

                  {/* Detailed Analysis */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      Detailed Analysis Metrics
                    </h4>
                    <AnalysisMetrics 
                      analysis={analysisResult.analysis} 
                      type={analysisResult.type} 
                    />
                  </div>

                  {/* Analysis Insights */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="mt-8 p-6 bg-[#0F1419] rounded-lg border border-[#3B82F6]/20"
                  >
                    <h5 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                      <Lightbulb className="w-5 h-5 text-yellow-400" />
                      Analysis Insights
                    </h5>
                    <div className="space-y-2 text-gray-300">
                      {analysisResult.prediction === 'authentic' && (
                        <p>✅ This content shows strong indicators of human creation with natural patterns and authentic characteristics.</p>
                      )}
                      {analysisResult.prediction === 'ai_generated' && (
                        <p>🤖 This content exhibits clear AI generation patterns with synthetic characteristics and artificial markers.</p>
                      )}
                      {analysisResult.prediction === 'ai_modified' && (
                        <p>⚠️ This content appears to be a hybrid of human and AI elements, showing signs of modification or enhancement.</p>
                      )}
                      <p>🔍 Our multi-modal analysis examined {Object.keys(analysisResult.analysis).length} different aspects to reach this conclusion.</p>
                      <p>⚡ Processing completed in {analysisResult.processing_time} seconds using advanced AI detection algorithms.</p>
                    </div>
                  </motion.div>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Feature Highlights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <Shield className="w-12 h-12 text-blue-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Advanced Detection</h3>
            <p className="text-gray-400">
              Multi-modal analysis across text, images, videos, and audio content
            </p>
          </div>
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <Zap className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Real-time Analysis</h3>
            <p className="text-gray-400">
              Fast processing with detailed confidence scores and insights
            </p>
          </div>
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <Brain className="w-12 h-12 text-purple-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">AI-Powered</h3>
            <p className="text-gray-400">
              State-of-the-art machine learning models for accurate detection
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DemoVisualization;

import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  ArrowRight, 
  Brain, 
  Search, 
  BarChart3,
  CheckCircle,
  Clock,
  Zap
} from 'lucide-react';

const WorkflowVisualization = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const workflowSteps = useMemo(() => [
    {
      id: 1,
      title: 'Upload Content',
      description: 'Upload your text, image, video, or audio file for analysis',
      icon: Upload,
      color: 'blue',
      details: [
        'Supports multiple file formats',
        'Secure file handling',
        'Automatic format detection',
        'File validation and preprocessing'
      ],
      duration: '0.1s'
    },
    {
      id: 2,
      title: 'Content Analysis',
      description: 'AI-powered multi-modal analysis examines your content',
      icon: Search,
      color: 'purple',
      details: [
        'Metadata extraction',
        'Pattern recognition',
        'Feature analysis',
        'Multi-dimensional scoring'
      ],
      duration: '1-4s'
    },
    {
      id: 3,
      title: 'AI Classification',
      description: 'Advanced algorithms determine content authenticity',
      icon: Brain,
      color: 'green',
      details: [
        'Machine learning classification',
        'Confidence score calculation',
        'Authenticity determination',
        'Risk assessment'
      ],
      duration: '0.5s'
    },
    {
      id: 4,
      title: 'Result Visualization',
      description: 'Detailed analysis results with actionable insights',
      icon: BarChart3,
      color: 'orange',
      details: [
        'Visual confidence meters',
        'Detailed breakdown charts',
        'Actionable recommendations',
        'Downloadable reports'
      ],
      duration: '0.1s'
    }
  ], []);

  const playAnimation = () => {
    setIsAnimating(true);
    setActiveStep(0);

    workflowSteps.forEach((_, index) => {
      setTimeout(() => {
        setActiveStep(index);
      }, index * 1000);
    });

    setTimeout(() => {
      setIsAnimating(false);
      setActiveStep(0);
    }, workflowSteps.length * 1000 + 1000);
  };

  useEffect(() => {
    // Auto-play animation on component mount
    const playInitialAnimation = () => {
      setIsAnimating(true);
      setActiveStep(0);

      workflowSteps.forEach((_, index) => {
        setTimeout(() => {
          setActiveStep(index);
        }, index * 1000);
      });

      setTimeout(() => {
        setIsAnimating(false);
        setActiveStep(0);
      }, workflowSteps.length * 1000 + 1000);
    };

    const timer = setTimeout(() => {
      playInitialAnimation();
    }, 1000);

    return () => clearTimeout(timer);
  }, [workflowSteps]);

  const getStepColor = (step, index) => {
    const colors = {
      blue: 'border-blue-400 bg-blue-400/20 text-blue-400',
      purple: 'border-purple-400 bg-purple-400/20 text-purple-400',
      green: 'border-green-400 bg-green-400/20 text-green-400',
      orange: 'border-orange-400 bg-orange-400/20 text-orange-400'
    };

    if (isAnimating) {
      if (index < activeStep) return 'border-green-400 bg-green-400/20 text-green-400';
      if (index === activeStep) return colors[step.color];
      return 'border-gray-600 bg-gray-600/20 text-gray-500';
    }

    return colors[step.color];
  };

  const getConnectorStatus = (index) => {
    if (!isAnimating) return 'bg-gray-600';
    if (index < activeStep) return 'bg-green-400';
    return 'bg-gray-600';
  };

  return (
    <div className="min-h-screen bg-[#0F1419] pt-20">
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl font-bold text-white mb-4">
            AI Detection Workflow
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Understand how our advanced AI detection system processes and analyzes content 
            to determine authenticity with industry-leading accuracy.
          </p>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={playAnimation}
            disabled={isAnimating}
            className="bg-[#3B82F6] hover:bg-[#2563EB] disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-semibold flex items-center gap-2 mx-auto transition-colors"
          >
            {isAnimating ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                />
                Processing...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Start Demo Workflow
              </>
            )}
          </motion.button>
        </motion.div>

        {/* Workflow Steps */}
        <div className="relative">
          {/* Progress Line */}
          <div className="absolute top-16 left-0 right-0 h-1 bg-gray-700 rounded-full hidden lg:block">
            <motion.div
              initial={{ width: '0%' }}
              animate={{ 
                width: isAnimating ? `${(activeStep / (workflowSteps.length - 1)) * 100}%` : '0%'
              }}
              transition={{ duration: 0.5 }}
              className="h-full bg-[#3B82F6] rounded-full"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 relative">
            {workflowSteps.map((step, index) => {
              const Icon = step.icon;
              const isActive = isAnimating && index === activeStep;
              const isCompleted = isAnimating && index < activeStep;

              return (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.2 }}
                  className="relative"
                >
                  {/* Step Card */}
                  <motion.div
                    className={`bg-[#1C2128] rounded-xl p-6 border transition-all duration-500 ${
                      isActive ? 'border-[#3B82F6] bg-[#3B82F6]/10 scale-105' :
                      isCompleted ? 'border-green-400 bg-green-400/10' :
                      'border-[#3B82F6]/20'
                    }`}
                    animate={{
                      scale: isActive ? 1.05 : 1,
                      boxShadow: isActive ? '0 0 30px rgba(59, 130, 246, 0.3)' : '0 0 0px rgba(59, 130, 246, 0)'
                    }}
                  >
                    {/* Step Number & Icon */}
                    <div className="flex items-center justify-between mb-4">
                      <div className={`w-12 h-12 rounded-lg border-2 flex items-center justify-center transition-all duration-500 ${
                        getStepColor(step, index)
                      }`}>
                        {isCompleted ? (
                          <CheckCircle className="w-6 h-6" />
                        ) : (
                          <Icon className="w-6 h-6" />
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Clock className="w-4 h-4" />
                        {step.duration}
                      </div>
                    </div>

                    {/* Step Content */}
                    <h3 className="text-xl font-semibold text-white mb-3">
                      {step.title}
                    </h3>
                    <p className="text-gray-400 mb-4">
                      {step.description}
                    </p>

                    {/* Step Details */}
                    <div className="space-y-2">
                      {step.details.map((detail, detailIndex) => (
                        <motion.div
                          key={detailIndex}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ 
                            opacity: isActive ? 1 : 0.7, 
                            x: isActive ? 0 : -10 
                          }}
                          transition={{ delay: isActive ? detailIndex * 0.1 : 0 }}
                          className="flex items-center gap-2 text-sm text-gray-300"
                        >
                          <div className={`w-2 h-2 rounded-full ${
                            isActive ? 'bg-[#3B82F6]' : 
                            isCompleted ? 'bg-green-400' : 'bg-gray-600'
                          }`} />
                          {detail}
                        </motion.div>
                      ))}
                    </div>

                    {/* Active Indicator */}
                    {isActive && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-2 -right-2 w-6 h-6 bg-[#3B82F6] rounded-full flex items-center justify-center"
                      >
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                          className="w-3 h-3 border-2 border-white border-t-transparent rounded-full"
                        />
                      </motion.div>
                    )}
                  </motion.div>

                  {/* Connector Arrow */}
                  {index < workflowSteps.length - 1 && (
                    <div className="hidden lg:block absolute top-8 -right-4 z-10">
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: index * 0.2 + 0.1 }}
                        className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors duration-500 ${
                          getConnectorStatus(index) === 'bg-green-400' 
                            ? 'bg-green-400 text-white' 
                            : 'bg-gray-600 text-gray-400'
                        }`}
                      >
                        <ArrowRight className="w-4 h-4" />
                      </motion.div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Workflow Statistics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-16 grid grid-cols-1 md:grid-cols-4 gap-6"
        >
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <div className="text-3xl font-bold text-blue-400 mb-2">2-6s</div>
            <div className="text-gray-400">Total Process Time</div>
          </div>
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">94.2%</div>
            <div className="text-gray-400">Detection Accuracy</div>
          </div>
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <div className="text-3xl font-bold text-purple-400 mb-2">50+</div>
            <div className="text-gray-400">Analysis Parameters</div>
          </div>
          <div className="bg-[#1C2128] rounded-xl p-6 border border-[#3B82F6]/20 text-center">
            <div className="text-3xl font-bold text-orange-400 mb-2">4</div>
            <div className="text-gray-400">Content Types</div>
          </div>
        </motion.div>

        {/* Technical Details */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
          className="mt-16 bg-[#1C2128] rounded-xl p-8 border border-[#3B82F6]/20"
        >
          <h3 className="text-2xl font-bold text-white mb-6">Technical Implementation</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-lg font-semibold text-white mb-4">AI Models & Algorithms</h4>
              <ul className="space-y-2 text-gray-300">
                <li>• Multi-modal neural networks for content analysis</li>
                <li>• Advanced pattern recognition algorithms</li>
                <li>• Deep learning classification models</li>
                <li>• Real-time processing optimization</li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold text-white mb-4">Analysis Capabilities</h4>
              <ul className="space-y-2 text-gray-300">
                <li>• Pixel-level image analysis</li>
                <li>• Linguistic pattern detection</li>
                <li>• Temporal video analysis</li>
                <li>• Audio spectral examination</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default WorkflowVisualization;

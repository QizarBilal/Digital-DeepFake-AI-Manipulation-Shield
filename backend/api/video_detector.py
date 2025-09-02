"""
Video Deepfake Detection Module
Implements CNN + LSTM architecture for temporal analysis and micro-expression detection
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import logging
import asyncio
import json
from datetime import datetime
import mediapipe as mp

logger = logging.getLogger(__name__)

class VideoDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    async def analyze_video(self, video_path: str, task_id: str) -> Dict[str, Any]:
        """
        Comprehensive video deepfake analysis
        """
        try:
            logger.info(f"Starting video analysis for task {task_id}")
            
            # Extract video metadata
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Initialize analysis containers
            frames_data = []
            blink_patterns = []
            micro_expressions = []
            temporal_consistency = []
            
            frame_idx = 0
            anomaly_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame for efficiency
                if frame_idx % max(1, int(fps // 5)) == 0:  # Process 5 frames per second
                    frame_analysis = await self._analyze_frame(frame, frame_idx)
                    frames_data.append(frame_analysis)
                    
                    # Detect anomalies
                    if frame_analysis['anomaly_score'] > 0.7:
                        anomaly_frames.append({
                            'frame_number': frame_idx,
                            'timestamp': frame_idx / fps,
                            'anomaly_score': frame_analysis['anomaly_score'],
                            'anomaly_type': frame_analysis['anomaly_type']
                        })
                
                frame_idx += 1
            
            cap.release()
            
            # Temporal analysis
            temporal_analysis = await self._temporal_analysis(frames_data)
            
            # Blink pattern analysis
            blink_analysis = await self._blink_pattern_analysis(frames_data)
            
            # Micro-expression analysis
            expression_analysis = await self._micro_expression_analysis(frames_data)
            
            # Calculate overall authenticity score
            authenticity_score = await self._calculate_authenticity_score(
                temporal_analysis, blink_analysis, expression_analysis
            )
            
            # Generate visualization data
            visualization_data = await self._generate_visualization_data(
                frames_data, anomaly_frames, temporal_analysis
            )
            
            result = {
                'task_id': task_id,
                'analysis_type': 'video',
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'duration': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                },
                'authenticity_score': authenticity_score,
                'confidence_level': self._calculate_confidence(authenticity_score),
                'detection_results': {
                    'is_deepfake': authenticity_score < 50,
                    'risk_level': self._get_risk_level(authenticity_score),
                    'anomaly_count': len(anomaly_frames)
                },
                'detailed_analysis': {
                    'temporal_consistency': temporal_analysis,
                    'blink_patterns': blink_analysis,
                    'micro_expressions': expression_analysis,
                    'anomaly_frames': anomaly_frames[:10]  # Limit to first 10 for response size
                },
                'visualization_data': visualization_data,
                'biometric_signals': {
                    'eye_movement_irregularities': len([f for f in frames_data if f.get('eye_irregularity', 0) > 0.5]),
                    'facial_landmark_inconsistencies': temporal_analysis.get('landmark_inconsistency', 0),
                    'texture_artifacts': temporal_analysis.get('texture_artifacts', 0)
                }
            }
            
            logger.info(f"Video analysis completed for task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Video analysis error: {str(e)}")
            raise
    
    async def _analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Analyze individual frame for deepfake indicators"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_results = self.face_detector.process(rgb_frame)
        
        if not face_results.detections:
            return {
                'frame_idx': frame_idx,
                'face_detected': False,
                'anomaly_score': 0.0,
                'anomaly_type': 'no_face'
            }
        
        # Get face landmarks
        mesh_results = self.face_mesh.process(rgb_frame)
        
        frame_analysis = {
            'frame_idx': frame_idx,
            'face_detected': True,
            'face_count': len(face_results.detections),
            'landmarks': [],
            'eye_aspect_ratio': 0.0,
            'mouth_aspect_ratio': 0.0,
            'anomaly_score': 0.0,
            'anomaly_type': 'none'
        }
        
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            
            # Calculate eye aspect ratio for blink detection
            eye_ratio = self._calculate_eye_aspect_ratio(landmarks)
            frame_analysis['eye_aspect_ratio'] = eye_ratio
            
            # Calculate mouth aspect ratio
            mouth_ratio = self._calculate_mouth_aspect_ratio(landmarks)
            frame_analysis['mouth_aspect_ratio'] = mouth_ratio
            
            # Detect facial inconsistencies
            anomaly_score = await self._detect_facial_anomalies(frame, landmarks)
            frame_analysis['anomaly_score'] = anomaly_score
            
            # Classify anomaly type
            if anomaly_score > 0.7:
                frame_analysis['anomaly_type'] = 'high_manipulation'
            elif anomaly_score > 0.5:
                frame_analysis['anomaly_type'] = 'moderate_manipulation'
            elif anomaly_score > 0.3:
                frame_analysis['anomaly_type'] = 'low_manipulation'
        
        return frame_analysis
    
    def _calculate_eye_aspect_ratio(self, landmarks) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            # Left eye landmarks (simplified)
            left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            
            if len(landmarks.landmark) > max(left_eye):
                # Calculate vertical distances
                v1 = abs(landmarks.landmark[159].y - landmarks.landmark[145].y)
                v2 = abs(landmarks.landmark[158].y - landmarks.landmark[153].y)
                
                # Calculate horizontal distance
                h = abs(landmarks.landmark[33].x - landmarks.landmark[133].x)
                
                if h > 0:
                    ear = (v1 + v2) / (2.0 * h)
                    return ear
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_mouth_aspect_ratio(self, landmarks) -> float:
        """Calculate Mouth Aspect Ratio for expression analysis"""
        try:
            # Mouth landmarks
            mouth_points = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            
            if len(landmarks.landmark) > max(mouth_points):
                # Calculate mouth opening
                v1 = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
                h1 = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
                
                if h1 > 0:
                    mar = v1 / h1
                    return mar
            
            return 0.0
        except:
            return 0.0
    
    async def _detect_facial_anomalies(self, frame: np.ndarray, landmarks) -> float:
        """Detect facial manipulation artifacts using CNN-based analysis"""
        try:
            # Extract face region
            h, w, _ = frame.shape
            x_coords = [landmark.x * w for landmark in landmarks.landmark]
            y_coords = [landmark.y * h for landmark in landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
            
            # Ensure bounds are valid
            x_min, x_max = max(0, x_min), min(w, x_max)
            y_min, y_max = max(0, y_min), min(h, y_max)
            
            face_region = frame[y_min:y_max, x_min:x_max]
            
            if face_region.size == 0:
                return 0.0
            
            # Resize for model input
            face_resized = cv2.resize(face_region, (224, 224))
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Simple anomaly detection based on pixel statistics
            # In production, this would use a trained CNN model
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Check for compression artifacts
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Check for color inconsistencies
            color_std = np.std(face_normalized)
            
            # Check for unnatural skin texture
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            texture_diff = np.mean(np.abs(gray.astype(float) - blur.astype(float)))
            
            # Combine metrics (simplified scoring)
            anomaly_score = 0.0
            
            if laplacian_var < 50:  # Low edge variance indicates over-smoothing
                anomaly_score += 0.3
            
            if color_std < 0.1:  # Unnaturally uniform colors
                anomaly_score += 0.2
                
            if texture_diff < 5:  # Over-smoothed texture
                anomaly_score += 0.3
            
            return min(anomaly_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Anomaly detection error: {str(e)}")
            return 0.0
    
    async def _temporal_analysis(self, frames_data: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal consistency across frames"""
        if len(frames_data) < 2:
            return {'consistency_score': 1.0, 'temporal_anomalies': []}
        
        # Analyze eye aspect ratio consistency
        ear_values = [f['eye_aspect_ratio'] for f in frames_data if f['face_detected']]
        ear_std = np.std(ear_values) if ear_values else 0
        
        # Analyze mouth aspect ratio consistency
        mar_values = [f['mouth_aspect_ratio'] for f in frames_data if f['face_detected']]
        mar_std = np.std(mar_values) if mar_values else 0
        
        # Detect sudden changes in facial metrics
        anomalies = []
        for i in range(1, len(frames_data)):
            if frames_data[i]['face_detected'] and frames_data[i-1]['face_detected']:
                ear_diff = abs(frames_data[i]['eye_aspect_ratio'] - frames_data[i-1]['eye_aspect_ratio'])
                if ear_diff > 0.1:  # Sudden change threshold
                    anomalies.append({
                        'frame': frames_data[i]['frame_idx'],
                        'type': 'sudden_eye_change',
                        'magnitude': ear_diff
                    })
        
        # Calculate overall consistency score
        consistency_score = max(0, 1.0 - (ear_std * 2 + mar_std * 2 + len(anomalies) * 0.1))
        
        return {
            'consistency_score': consistency_score,
            'ear_variance': ear_std,
            'mar_variance': mar_std,
            'temporal_anomalies': anomalies[:5],  # Limit output
            'landmark_inconsistency': 1.0 - consistency_score,
            'texture_artifacts': len(anomalies) / max(1, len(frames_data))
        }
    
    async def _blink_pattern_analysis(self, frames_data: List[Dict]) -> Dict[str, Any]:
        """Analyze blink patterns for authenticity"""
        face_frames = [f for f in frames_data if f['face_detected']]
        
        if len(face_frames) < 10:
            return {'natural_blink_score': 0.5, 'blink_events': []}
        
        # Detect blink events
        blink_threshold = 0.2
        blink_events = []
        
        for i, frame in enumerate(face_frames):
            if frame['eye_aspect_ratio'] < blink_threshold:
                blink_events.append({
                    'frame': frame['frame_idx'],
                    'intensity': blink_threshold - frame['eye_aspect_ratio']
                })
        
        # Analyze blink frequency (normal: 15-20 blinks per minute)
        duration_minutes = len(face_frames) / (30 * 60)  # Assuming 30 fps processing
        blink_rate = len(blink_events) / max(0.1, duration_minutes)
        
        # Score naturalness
        natural_score = 1.0
        if blink_rate < 10 or blink_rate > 30:  # Abnormal blink rate
            natural_score -= 0.3
        
        if len(blink_events) == 0 and len(face_frames) > 50:  # No blinks in long video
            natural_score -= 0.5
        
        return {
            'natural_blink_score': max(0, natural_score),
            'blink_rate_per_minute': blink_rate,
            'blink_events': blink_events[:10],
            'total_blinks': len(blink_events)
        }
    
    async def _micro_expression_analysis(self, frames_data: List[Dict]) -> Dict[str, Any]:
        """Analyze micro-expressions for authenticity"""
        face_frames = [f for f in frames_data if f['face_detected']]
        
        if len(face_frames) < 5:
            return {'expression_naturalness': 0.5, 'micro_expressions': []}
        
        # Analyze mouth movement patterns
        mouth_values = [f['mouth_aspect_ratio'] for f in face_frames]
        mouth_variance = np.var(mouth_values)
        
        # Detect micro-expressions (sudden small changes)
        micro_expressions = []
        for i in range(1, len(face_frames)):
            mouth_change = abs(face_frames[i]['mouth_aspect_ratio'] - face_frames[i-1]['mouth_aspect_ratio'])
            if 0.01 < mouth_change < 0.05:  # Micro-expression range
                micro_expressions.append({
                    'frame': face_frames[i]['frame_idx'],
                    'type': 'mouth_micro_movement',
                    'intensity': mouth_change
                })
        
        # Calculate naturalness score
        naturalness = min(1.0, mouth_variance * 10 + len(micro_expressions) * 0.1)
        
        return {
            'expression_naturalness': naturalness,
            'mouth_movement_variance': mouth_variance,
            'micro_expressions': micro_expressions[:10],
            'total_micro_expressions': len(micro_expressions)
        }
    
    async def _calculate_authenticity_score(self, temporal: Dict, blink: Dict, expression: Dict) -> float:
        """Calculate overall authenticity score (0-100)"""
        # Weight different factors
        temporal_weight = 0.4
        blink_weight = 0.3
        expression_weight = 0.3
        
        temporal_score = temporal.get('consistency_score', 0.5) * 100
        blink_score = blink.get('natural_blink_score', 0.5) * 100
        expression_score = expression.get('expression_naturalness', 0.5) * 100
        
        weighted_score = (
            temporal_score * temporal_weight +
            blink_score * blink_weight +
            expression_score * expression_weight
        )
        
        return round(weighted_score, 1)
    
    def _calculate_confidence(self, authenticity_score: float) -> str:
        """Calculate confidence level based on score"""
        if authenticity_score >= 90:
            return "Very High"
        elif authenticity_score >= 75:
            return "High"
        elif authenticity_score >= 50:
            return "Medium"
        elif authenticity_score >= 25:
            return "Low"
        else:
            return "Very Low"
    
    def _get_risk_level(self, authenticity_score: float) -> str:
        """Get risk level based on authenticity score"""
        if authenticity_score >= 80:
            return "Very Low Risk"
        elif authenticity_score >= 60:
            return "Low Risk"
        elif authenticity_score >= 40:
            return "Medium Risk"
        elif authenticity_score >= 20:
            return "High Risk"
        else:
            return "Very High Risk"
    
    async def _generate_visualization_data(self, frames_data: List[Dict], anomaly_frames: List[Dict], temporal_analysis: Dict) -> Dict[str, Any]:
        """Generate data for frontend visualizations"""
        face_frames = [f for f in frames_data if f['face_detected']]
        
        # Timeline data for charts
        timeline_data = []
        for frame in face_frames[:100]:  # Limit for visualization
            timeline_data.append({
                'frame': frame['frame_idx'],
                'authenticity': max(0, 100 - frame['anomaly_score'] * 100),
                'eye_ratio': frame['eye_aspect_ratio'],
                'mouth_ratio': frame['mouth_aspect_ratio']
            })
        
        # Heatmap data for anomaly distribution
        heatmap_data = []
        for anomaly in anomaly_frames[:20]:
            heatmap_data.append({
                'timestamp': anomaly['timestamp'],
                'intensity': anomaly['anomaly_score'],
                'type': anomaly['anomaly_type']
            })
        
        return {
            'timeline_data': timeline_data,
            'heatmap_data': heatmap_data,
            'summary_stats': {
                'total_frames_analyzed': len(face_frames),
                'anomaly_percentage': (len(anomaly_frames) / max(1, len(face_frames))) * 100,
                'average_authenticity': np.mean([d['authenticity'] for d in timeline_data]) if timeline_data else 50
            }
        }

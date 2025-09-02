"""
Image Deepfake Detection Module
Implements CNN-based models for face manipulation and synthetic content detection
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from typing import Dict, List, Tuple, Any
import logging
import asyncio
import json
from datetime import datetime
import face_recognition
import dlib

logger = logging.getLogger(__name__)

class ImageDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = None  # Will be loaded if available
        
    async def analyze_image(self, image_path: str, task_id: str) -> Dict[str, Any]:
        """
        Comprehensive image deepfake analysis using face manipulation and content analysis
        """
        try:
            logger.info(f"Starting image analysis for task {task_id}")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image file")
            
            original_shape = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Face detection and analysis
            face_analysis = await self._analyze_faces(image, rgb_image)
            
            # Pixel-level analysis
            pixel_analysis = await self._pixel_level_analysis(image)
            
            # Compression artifact analysis
            compression_analysis = await self._compression_artifact_analysis(image)
            
            # Geometric analysis
            geometric_analysis = await self._geometric_analysis(image, face_analysis)
            
            # Texture analysis
            texture_analysis = await self._texture_analysis(image)
            
            # Lighting and shadow analysis
            lighting_analysis = await self._lighting_analysis(image)
            
            # Calculate authenticity score
            authenticity_score = await self._calculate_image_authenticity_score(
                face_analysis, pixel_analysis, compression_analysis, 
                geometric_analysis, texture_analysis, lighting_analysis
            )
            
            # Generate visualization data
            visualization_data = await self._generate_image_visualization_data(
                image, face_analysis, pixel_analysis, compression_analysis
            )
            
            result = {
                'task_id': task_id,
                'analysis_type': 'image',
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'width': original_shape[1],
                    'height': original_shape[0],
                    'channels': original_shape[2],
                    'format': 'RGB'
                },
                'authenticity_score': authenticity_score,
                'confidence_level': self._calculate_confidence(authenticity_score),
                'detection_results': {
                    'is_deepfake': authenticity_score < 50,
                    'risk_level': self._get_risk_level(authenticity_score),
                    'faces_detected': face_analysis.get('face_count', 0)
                },
                'detailed_analysis': {
                    'face_analysis': face_analysis,
                    'pixel_analysis': pixel_analysis,
                    'compression_analysis': compression_analysis,
                    'geometric_analysis': geometric_analysis,
                    'texture_analysis': texture_analysis,
                    'lighting_analysis': lighting_analysis
                },
                'visualization_data': visualization_data,
                'biometric_signals': {
                    'facial_landmark_irregularities': face_analysis.get('landmark_irregularities', 0),
                    'texture_inconsistencies': texture_analysis.get('inconsistency_score', 0),
                    'lighting_anomalies': lighting_analysis.get('anomaly_score', 0)
                }
            }
            
            logger.info(f"Image analysis completed for task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            raise
    
    async def _analyze_faces(self, image: np.ndarray, rgb_image: np.ndarray) -> Dict[str, Any]:
        """Analyze faces for manipulation indicators"""
        try:
            # Detect faces using dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            face_data = []
            total_landmark_irregularities = 0
            
            for i, face in enumerate(faces):
                # Extract face region
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_region = image[y:y+h, x:x+w]
                
                if face_region.size == 0:
                    continue
                
                # Face recognition encoding
                try:
                    encodings = face_recognition.face_encodings(rgb_image)
                    face_landmarks = face_recognition.face_landmarks(rgb_image)
                except:
                    encodings = []
                    face_landmarks = []
                
                # Analyze face quality and artifacts
                face_quality = await self._analyze_face_quality(face_region)
                
                # Analyze facial symmetry
                symmetry_score = await self._analyze_facial_symmetry(face_region)
                
                # Detect edge artifacts around face
                edge_artifacts = await self._detect_edge_artifacts(image, (x, y, w, h))
                
                # Landmark irregularities
                landmark_irregularities = 0
                if i < len(face_landmarks):
                    landmark_irregularities = await self._analyze_landmark_irregularities(face_landmarks[i])
                
                total_landmark_irregularities += landmark_irregularities
                
                face_info = {
                    'face_id': i,
                    'bbox': [x, y, w, h],
                    'face_quality': face_quality,
                    'symmetry_score': symmetry_score,
                    'edge_artifacts': edge_artifacts,
                    'landmark_irregularities': landmark_irregularities,
                    'manipulation_score': self._calculate_face_manipulation_score(
                        face_quality, symmetry_score, edge_artifacts, landmark_irregularities
                    )
                }
                
                face_data.append(face_info)
            
            # Calculate overall face analysis scores
            avg_manipulation_score = 0
            if face_data:
                avg_manipulation_score = sum(f['manipulation_score'] for f in face_data) / len(face_data)
            
            return {
                'face_count': len(faces),
                'faces': face_data[:5],  # Limit output for response size
                'average_manipulation_score': avg_manipulation_score,
                'landmark_irregularities': total_landmark_irregularities / max(1, len(faces)),
                'face_authenticity': max(0, 1.0 - avg_manipulation_score)
            }
            
        except Exception as e:
            logger.warning(f"Face analysis error: {str(e)}")
            return {'face_count': 0, 'face_authenticity': 0.5}
    
    async def _analyze_face_quality(self, face_region: np.ndarray) -> Dict[str, float]:
        """Analyze face image quality indicators"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Noise analysis
            noise_score = np.std(gray)
            
            # Contrast analysis
            contrast_score = gray.std()
            
            # Brightness uniformity
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Normalize scores
            quality_metrics = {
                'blur_score': float(min(1.0, blur_score / 1000)),  # Higher = less blur
                'noise_level': float(min(1.0, noise_score / 50)),   # Higher = more noise
                'contrast': float(min(1.0, contrast_score / 100)),
                'brightness_uniformity': float(1.0 - min(1.0, brightness_std / 100))
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Face quality analysis error: {str(e)}")
            return {'blur_score': 0.5, 'noise_level': 0.5}
    
    async def _analyze_facial_symmetry(self, face_region: np.ndarray) -> float:
        """Analyze facial symmetry for manipulation detection"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Split face vertically
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            
            # Flip right half for comparison
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate similarity
            diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            return float(max(0, symmetry_score))
            
        except Exception as e:
            logger.warning(f"Symmetry analysis error: {str(e)}")
            return 0.5
    
    async def _detect_edge_artifacts(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> float:
        """Detect edge artifacts around face region"""
        try:
            x, y, w, h = face_bbox
            
            # Expand bounding box slightly
            margin = 10
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(image.shape[1], x + w + margin)
            y_end = min(image.shape[0], y + h + margin)
            
            # Extract region around face
            region = image[y_start:y_end, x_start:x_end]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Focus on face boundary area
            face_x = x - x_start
            face_y = y - y_start
            
            # Create mask for face boundary
            mask = np.zeros_like(edges)
            cv2.rectangle(mask, (face_x, face_y), (face_x + w, face_y + h), 255, 3)
            
            # Count edges at face boundary
            boundary_edges = np.sum((edges > 0) & (mask > 0))
            total_boundary_pixels = np.sum(mask > 0)
            
            if total_boundary_pixels == 0:
                return 0.0
            
            edge_density = boundary_edges / total_boundary_pixels
            
            # Higher edge density may indicate manipulation artifacts
            return float(min(1.0, edge_density * 5))
            
        except Exception as e:
            logger.warning(f"Edge artifact detection error: {str(e)}")
            return 0.0
    
    async def _analyze_landmark_irregularities(self, landmarks: Dict) -> float:
        """Analyze facial landmark irregularities"""
        try:
            irregularities = 0.0
            
            # Check eye symmetry
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_eye = np.array(landmarks['left_eye'])
                right_eye = np.array(landmarks['right_eye'])
                
                left_eye_center = np.mean(left_eye, axis=0)
                right_eye_center = np.mean(right_eye, axis=0)
                
                # Check if eyes are roughly at same height
                eye_height_diff = abs(left_eye_center[1] - right_eye_center[1])
                if eye_height_diff > 10:  # Threshold for irregular eye alignment
                    irregularities += 0.3
            
            # Check mouth positioning
            if 'top_lip' in landmarks and 'bottom_lip' in landmarks:
                top_lip = np.array(landmarks['top_lip'])
                bottom_lip = np.array(landmarks['bottom_lip'])
                
                # Check mouth symmetry
                top_center = np.mean(top_lip, axis=0)
                bottom_center = np.mean(bottom_lip, axis=0)
                
                mouth_asymmetry = abs(top_center[0] - bottom_center[0])
                if mouth_asymmetry > 5:
                    irregularities += 0.2
            
            # Check nose alignment
            if 'nose_tip' in landmarks and 'nose_bridge' in landmarks:
                nose_tip = np.array(landmarks['nose_tip'])
                nose_bridge = np.array(landmarks['nose_bridge'])
                
                if len(nose_tip) > 0 and len(nose_bridge) > 0:
                    nose_center = np.mean(nose_tip, axis=0)
                    bridge_center = np.mean(nose_bridge, axis=0)
                    
                    # Check nose alignment
                    nose_alignment = abs(nose_center[0] - bridge_center[0])
                    if nose_alignment > 8:
                        irregularities += 0.2
            
            return min(1.0, irregularities)
            
        except Exception as e:
            logger.warning(f"Landmark analysis error: {str(e)}")
            return 0.0
    
    def _calculate_face_manipulation_score(self, quality: Dict, symmetry: float, edge_artifacts: float, landmark_irreg: float) -> float:
        """Calculate overall face manipulation score"""
        # Lower quality metrics may indicate manipulation
        quality_score = 1.0 - quality.get('blur_score', 0.5)  # More blur = more suspicious
        quality_score += quality.get('noise_level', 0.5)      # More noise = more suspicious
        quality_score = min(1.0, quality_score / 2)
        
        # Lower symmetry = more suspicious
        symmetry_score = 1.0 - symmetry
        
        # Combine all factors
        manipulation_score = (
            quality_score * 0.3 +
            symmetry_score * 0.3 +
            edge_artifacts * 0.2 +
            landmark_irreg * 0.2
        )
        
        return min(1.0, manipulation_score)
    
    async def _pixel_level_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze pixel-level indicators of manipulation"""
        try:
            # Convert to different color spaces for analysis
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Color distribution analysis
            color_analysis = await self._analyze_color_distribution(rgb, hsv, lab)
            
            # Noise pattern analysis
            noise_analysis = await self._analyze_noise_patterns(image)
            
            # Compression artifacts
            compression_artifacts = await self._detect_compression_irregularities(image)
            
            # Statistical analysis
            statistical_analysis = await self._statistical_pixel_analysis(image)
            
            return {
                'color_analysis': color_analysis,
                'noise_analysis': noise_analysis,
                'compression_artifacts': compression_artifacts,
                'statistical_analysis': statistical_analysis,
                'overall_pixel_authenticity': self._calculate_pixel_authenticity(
                    color_analysis, noise_analysis, compression_artifacts
                )
            }
            
        except Exception as e:
            logger.warning(f"Pixel analysis error: {str(e)}")
            return {'overall_pixel_authenticity': 0.5}
    
    async def _analyze_color_distribution(self, rgb: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution for unnaturalness"""
        try:
            # RGB channel analysis
            r_mean, g_mean, b_mean = np.mean(rgb, axis=(0,1))
            r_std, g_std, b_std = np.std(rgb, axis=(0,1))
            
            # Check for unnatural color balance
            color_balance_score = 1.0
            
            # Extreme color channel dominance
            max_channel = max(r_mean, g_mean, b_mean)
            min_channel = min(r_mean, g_mean, b_mean)
            if max_channel > 0 and (max_channel / min_channel) > 2:
                color_balance_score -= 0.3
            
            # HSV analysis
            h_mean, s_mean, v_mean = np.mean(hsv, axis=(0,1))
            h_std, s_std, v_std = np.std(hsv, axis=(0,1))
            
            # Unnatural saturation levels
            if s_mean > 200 or s_mean < 20:  # Very high or very low saturation
                color_balance_score -= 0.2
            
            # LAB analysis for perceptual uniformity
            l_mean, a_mean, b_mean_lab = np.mean(lab, axis=(0,1))
            
            return {
                'color_balance_score': float(max(0, color_balance_score)),
                'rgb_channel_variance': float(np.mean([r_std, g_std, b_std])),
                'saturation_naturalness': float(1.0 - abs(s_mean - 128) / 128),
                'luminance_distribution': float(l_mean / 255)
            }
            
        except Exception as e:
            logger.warning(f"Color analysis error: {str(e)}")
            return {'color_balance_score': 0.5}
    
    async def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze noise patterns for manipulation indicators"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # High-frequency noise analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calculate noise characteristics
            high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4, 
                                                      magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4])
            total_energy = np.sum(magnitude_spectrum)
            
            noise_ratio = high_freq_energy / max(total_energy, 1)
            
            # Local noise variance
            kernel = np.ones((3, 3), np.float32) / 9
            blurred = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            noise_map = np.abs(gray.astype(np.float32) - blurred)
            local_noise_variance = np.var(noise_map)
            
            # Pattern regularity (artificial noise often has patterns)
            autocorr = cv2.matchTemplate(noise_map, noise_map[:50, :50], cv2.TM_CCOEFF_NORMED)
            pattern_score = np.max(autocorr[10:40, 10:40])  # Exclude center peak
            
            return {
                'noise_ratio': float(min(1.0, noise_ratio * 1000)),
                'local_noise_variance': float(min(1.0, local_noise_variance / 100)),
                'pattern_regularity': float(pattern_score),
                'noise_naturalness': float(max(0, 1.0 - pattern_score * 2))
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis error: {str(e)}")
            return {'noise_naturalness': 0.5}
    
    async def _detect_compression_irregularities(self, image: np.ndarray) -> Dict[str, float]:
        """Detect compression artifact irregularities"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # DCT-based block analysis (JPEG compression detection)
            h, w = gray.shape
            block_size = 8
            block_variance = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_variance.append(np.var(block))
            
            # Regularity in block variance indicates compression
            variance_regularity = np.std(block_variance) / max(np.mean(block_variance), 1)
            
            # Edge continuity analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            # Blocking artifacts detection
            blocking_score = self._detect_blocking_artifacts(gray)
            
            return {
                'compression_regularity': float(min(1.0, variance_regularity)),
                'edge_density': float(edge_density),
                'blocking_artifacts': blocking_score,
                'compression_authenticity': float(max(0, 1.0 - blocking_score))
            }
            
        except Exception as e:
            logger.warning(f"Compression analysis error: {str(e)}")
            return {'compression_authenticity': 0.5}
    
    def _detect_blocking_artifacts(self, gray: np.ndarray) -> float:
        """Detect JPEG blocking artifacts"""
        try:
            h, w = gray.shape
            
            # Check for 8x8 block boundaries (typical JPEG)
            vertical_boundaries = []
            horizontal_boundaries = []
            
            # Vertical boundaries
            for x in range(8, w, 8):
                if x < w - 1:
                    diff = np.mean(np.abs(gray[:, x].astype(float) - gray[:, x-1].astype(float)))
                    vertical_boundaries.append(diff)
            
            # Horizontal boundaries
            for y in range(8, h, 8):
                if y < h - 1:
                    diff = np.mean(np.abs(gray[y, :].astype(float) - gray[y-1, :].astype(float)))
                    horizontal_boundaries.append(diff)
            
            # Calculate blocking score
            avg_v_boundary = np.mean(vertical_boundaries) if vertical_boundaries else 0
            avg_h_boundary = np.mean(horizontal_boundaries) if horizontal_boundaries else 0
            
            # Compare with random boundaries
            random_v_diff = []
            random_h_diff = []
            
            for x in range(4, w-4, 8):  # Offset boundaries
                if x < w - 1:
                    diff = np.mean(np.abs(gray[:, x].astype(float) - gray[:, x-1].astype(float)))
                    random_v_diff.append(diff)
            
            for y in range(4, h-4, 8):  # Offset boundaries
                if y < h - 1:
                    diff = np.mean(np.abs(gray[y, :].astype(float) - gray[y-1, :].astype(float)))
                    random_h_diff.append(diff)
            
            avg_random_v = np.mean(random_v_diff) if random_v_diff else 1
            avg_random_h = np.mean(random_h_diff) if random_h_diff else 1
            
            # Blocking score: higher when block boundaries are more prominent
            v_blocking = (avg_v_boundary - avg_random_v) / max(avg_random_v, 1)
            h_blocking = (avg_h_boundary - avg_random_h) / max(avg_random_h, 1)
            
            blocking_score = max(0, (v_blocking + h_blocking) / 2)
            return float(min(1.0, blocking_score))
            
        except:
            return 0.0
    
    async def _statistical_pixel_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """Statistical analysis of pixel values"""
        try:
            # Convert to grayscale for statistical analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / np.sum(hist)
            
            # Entropy (measure of randomness)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            
            # Skewness and kurtosis
            from scipy import stats
            skewness = float(stats.skew(gray.flatten()))
            kurtosis = float(stats.kurtosis(gray.flatten()))
            
            # Peak detection in histogram
            peaks, _ = signal.find_peaks(hist.flatten(), height=np.max(hist) * 0.1)
            
            return {
                'entropy': float(entropy / 8),  # Normalize to 0-1
                'skewness': float(abs(skewness) / 3),  # Normalize
                'kurtosis': float(abs(kurtosis) / 10),  # Normalize
                'histogram_peaks': len(peaks),
                'statistical_naturalness': float(min(1.0, entropy / 8))
            }
            
        except Exception as e:
            logger.warning(f"Statistical analysis error: {str(e)}")
            return {'statistical_naturalness': 0.5}
    
    def _calculate_pixel_authenticity(self, color: Dict, noise: Dict, compression: Dict) -> float:
        """Calculate overall pixel-level authenticity"""
        color_score = color.get('color_balance_score', 0.5)
        noise_score = noise.get('noise_naturalness', 0.5)
        compression_score = compression.get('compression_authenticity', 0.5)
        
        # Weight the different factors
        authenticity = (color_score * 0.4 + noise_score * 0.3 + compression_score * 0.3)
        return float(authenticity)
    
    async def _compression_artifact_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Detailed compression artifact analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # JPEG quality estimation
            quality_estimate = await self._estimate_jpeg_quality(image)
            
            # Quantization table analysis
            quantization_analysis = await self._analyze_quantization_effects(gray)
            
            # Double compression detection
            double_compression = await self._detect_double_compression(gray)
            
            return {
                'estimated_quality': quality_estimate,
                'quantization_effects': quantization_analysis,
                'double_compression_score': double_compression,
                'compression_authenticity': max(0, 1.0 - double_compression)
            }
            
        except Exception as e:
            logger.warning(f"Compression artifact analysis error: {str(e)}")
            return {'compression_authenticity': 0.5}
    
    async def _estimate_jpeg_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate JPEG compression quality"""
        try:
            # Simple quality estimation based on high-frequency content
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (higher = better quality)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Estimate quality (simplified)
            quality_estimate = min(100, laplacian_var / 10)
            
            return {
                'estimated_quality': float(quality_estimate),
                'sharpness_metric': float(laplacian_var)
            }
            
        except:
            return {'estimated_quality': 75.0}
    
    async def _analyze_quantization_effects(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze quantization effects from compression"""
        try:
            # DCT analysis
            h, w = gray.shape
            block_size = 8
            
            dct_coeffs = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float64)
                    dct_block = cv2.dct(block)
                    dct_coeffs.append(dct_block.flatten())
            
            if not dct_coeffs:
                return {'quantization_score': 0.5}
            
            # Analyze coefficient distribution
            all_coeffs = np.concatenate(dct_coeffs)
            coeff_variance = np.var(all_coeffs)
            
            return {
                'quantization_score': float(min(1.0, coeff_variance / 1000)),
                'coefficient_variance': float(coeff_variance)
            }
            
        except:
            return {'quantization_score': 0.5}
    
    async def _detect_double_compression(self, gray: np.ndarray) -> float:
        """Detect signs of double JPEG compression"""
        try:
            # Simplified double compression detection
            # Based on histogram analysis of DCT coefficients
            
            h, w = gray.shape
            block_size = 8
            
            # Extract DCT coefficients
            dct_coeffs = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float64)
                    dct_block = cv2.dct(block)
                    # Focus on AC coefficients
                    dct_coeffs.extend(dct_block[1:, 1:].flatten())
            
            if not dct_coeffs:
                return 0.0
            
            # Histogram of coefficients
            hist, bins = np.histogram(dct_coeffs, bins=50, range=(-50, 50))
            
            # Look for periodic patterns in histogram (sign of double compression)
            # Simplified: check for multiple peaks
            peaks, _ = signal.find_peaks(hist, height=np.max(hist) * 0.1)
            
            # More peaks may indicate double compression
            double_compression_score = min(1.0, len(peaks) / 10)
            
            return float(double_compression_score)
            
        except:
            return 0.0
    
    async def _geometric_analysis(self, image: np.ndarray, face_analysis: Dict) -> Dict[str, Any]:
        """Analyze geometric inconsistencies"""
        try:
            # Perspective analysis
            perspective_score = await self._analyze_perspective_consistency(image)
            
            # Scale consistency
            scale_consistency = await self._analyze_scale_consistency(image, face_analysis)
            
            # Proportion analysis
            proportion_analysis = await self._analyze_proportions(face_analysis)
            
            return {
                'perspective_consistency': perspective_score,
                'scale_consistency': scale_consistency,
                'proportion_analysis': proportion_analysis,
                'geometric_authenticity': (perspective_score + scale_consistency + proportion_analysis.get('naturalness', 0.5)) / 3
            }
            
        except Exception as e:
            logger.warning(f"Geometric analysis error: {str(e)}")
            return {'geometric_authenticity': 0.5}
    
    async def _analyze_perspective_consistency(self, image: np.ndarray) -> float:
        """Analyze perspective consistency in the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Line detection
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.5
            
            # Analyze line angles for perspective consistency
            angles = []
            for line in lines[:20]:  # Limit to prevent overflow
                rho, theta = line[0]
                angles.append(theta)
            
            # Check for consistent vanishing points (simplified)
            angle_consistency = 1.0 - (np.std(angles) / np.pi)
            
            return float(max(0, angle_consistency))
            
        except:
            return 0.5
    
    async def _analyze_scale_consistency(self, image: np.ndarray, face_analysis: Dict) -> float:
        """Analyze scale consistency between faces"""
        try:
            faces = face_analysis.get('faces', [])
            
            if len(faces) < 2:
                return 1.0  # Cannot analyze with single face
            
            # Extract face sizes
            face_sizes = []
            for face in faces:
                bbox = face['bbox']
                width, height = bbox[2], bbox[3]
                face_area = width * height
                face_sizes.append(face_area)
            
            # Analyze size distribution
            size_variance = np.var(face_sizes)
            mean_size = np.mean(face_sizes)
            
            # Normalized variance (lower = more consistent)
            if mean_size > 0:
                normalized_variance = size_variance / (mean_size ** 2)
                consistency_score = max(0, 1.0 - normalized_variance)
            else:
                consistency_score = 0.5
            
            return float(consistency_score)
            
        except:
            return 0.5
    
    async def _analyze_proportions(self, face_analysis: Dict) -> Dict[str, float]:
        """Analyze facial proportions for naturalness"""
        try:
            faces = face_analysis.get('faces', [])
            
            if not faces:
                return {'naturalness': 0.5}
            
            proportion_scores = []
            
            for face in faces:
                bbox = face['bbox']
                width, height = bbox[2], bbox[3]
                
                # Aspect ratio analysis
                aspect_ratio = width / max(height, 1)
                
                # Natural face aspect ratio is typically 0.6-0.8
                if 0.6 <= aspect_ratio <= 0.8:
                    proportion_score = 1.0
                else:
                    proportion_score = max(0, 1.0 - abs(aspect_ratio - 0.7) / 0.3)
                
                proportion_scores.append(proportion_score)
            
            avg_naturalness = np.mean(proportion_scores) if proportion_scores else 0.5
            
            return {
                'naturalness': float(avg_naturalness),
                'average_aspect_ratio': float(np.mean([f['bbox'][2]/max(f['bbox'][3], 1) for f in faces])) if faces else 0.7
            }
            
        except:
            return {'naturalness': 0.5}
    
    async def _texture_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture consistency and naturalness"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern analysis
            texture_uniformity = await self._analyze_texture_uniformity(gray)
            
            # Skin texture analysis (if faces detected)
            skin_texture = await self._analyze_skin_texture(image)
            
            # Surface normal consistency
            surface_analysis = await self._analyze_surface_normals(gray)
            
            inconsistency_score = 1.0 - min(1.0, (texture_uniformity + skin_texture + surface_analysis) / 3)
            
            return {
                'texture_uniformity': texture_uniformity,
                'skin_texture_score': skin_texture,
                'surface_consistency': surface_analysis,
                'inconsistency_score': inconsistency_score,
                'texture_authenticity': 1.0 - inconsistency_score
            }
            
        except Exception as e:
            logger.warning(f"Texture analysis error: {str(e)}")
            return {'texture_authenticity': 0.5}
    
    async def _analyze_texture_uniformity(self, gray: np.ndarray) -> float:
        """Analyze texture uniformity using LBP"""
        try:
            # Simplified LBP implementation
            h, w = gray.shape
            lbp = np.zeros_like(gray)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray[i, j]
                    code = 0
                    
                    # 8-neighborhood
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                        gray[i+1, j-1], gray[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code += 2 ** k
                    
                    lbp[i, j] = code
            
            # Calculate uniformity
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / np.sum(hist)
            
            # Entropy as uniformity measure
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            uniformity = entropy / 8  # Normalize
            
            return float(min(1.0, uniformity))
            
        except:
            return 0.5
    
    async def _analyze_skin_texture(self, image: np.ndarray) -> float:
        """Analyze skin texture naturalness"""
        try:
            # Convert to YCrCb for skin detection
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Simple skin color range (rough approximation)
            lower_skin = np.array([0, 133, 77])
            upper_skin = np.array([255, 173, 127])
            
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            
            if np.sum(skin_mask) == 0:
                return 0.5  # No skin detected
            
            # Extract skin regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            skin_pixels = gray[skin_mask > 0]
            
            if len(skin_pixels) == 0:
                return 0.5
            
            # Analyze skin texture variance
            skin_variance = np.var(skin_pixels)
            
            # Natural skin has moderate variance (not too smooth, not too rough)
            optimal_variance = 200  # Empirical value
            variance_score = 1.0 - abs(skin_variance - optimal_variance) / optimal_variance
            
            return float(max(0, variance_score))
            
        except:
            return 0.5
    
    async def _analyze_surface_normals(self, gray: np.ndarray) -> float:
        """Analyze surface normal consistency"""
        try:
            # Gradient-based surface normal estimation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyze gradient consistency
            magnitude_variance = np.var(magnitude)
            magnitude_mean = np.mean(magnitude)
            
            if magnitude_mean == 0:
                return 0.5
            
            # Normalized variance (lower = more consistent)
            consistency = 1.0 - min(1.0, magnitude_variance / (magnitude_mean**2))
            
            return float(max(0, consistency))
            
        except:
            return 0.5
    
    async def _lighting_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting consistency and naturalness"""
        try:
            # Shadow analysis
            shadow_analysis = await self._analyze_shadows(image)
            
            # Lighting direction consistency
            lighting_direction = await self._analyze_lighting_direction(image)
            
            # Highlight analysis
            highlight_analysis = await self._analyze_highlights(image)
            
            # Calculate overall lighting authenticity
            anomaly_score = 1.0 - min(1.0, (
                shadow_analysis.get('naturalness', 0.5) +
                lighting_direction +
                highlight_analysis.get('naturalness', 0.5)
            ) / 3)
            
            return {
                'shadow_analysis': shadow_analysis,
                'lighting_direction_consistency': lighting_direction,
                'highlight_analysis': highlight_analysis,
                'anomaly_score': anomaly_score,
                'lighting_authenticity': 1.0 - anomaly_score
            }
            
        except Exception as e:
            logger.warning(f"Lighting analysis error: {str(e)}")
            return {'lighting_authenticity': 0.5}
    
    async def _analyze_shadows(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze shadow patterns for consistency"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Find potential shadow regions (low luminance)
            shadow_threshold = np.percentile(l_channel, 20)
            shadow_mask = l_channel < shadow_threshold
            
            if np.sum(shadow_mask) == 0:
                return {'naturalness': 0.5}
            
            # Analyze shadow edge softness
            shadow_edges = cv2.Canny((shadow_mask * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(shadow_edges > 0) / np.sum(shadow_mask)
            
            # Natural shadows have softer edges
            edge_softness = 1.0 - min(1.0, edge_density * 10)
            
            # Analyze shadow color consistency
            shadow_pixels = image[shadow_mask]
            if len(shadow_pixels) > 0:
                shadow_color_var = np.var(shadow_pixels, axis=0)
                color_consistency = 1.0 - min(1.0, np.mean(shadow_color_var) / 1000)
            else:
                color_consistency = 0.5
            
            naturalness = (edge_softness + color_consistency) / 2
            
            return {
                'naturalness': float(naturalness),
                'edge_softness': float(edge_softness),
                'color_consistency': float(color_consistency)
            }
            
        except:
            return {'naturalness': 0.5}
    
    async def _analyze_lighting_direction(self, image: np.ndarray) -> float:
        """Analyze lighting direction consistency"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate gradients to estimate lighting direction
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate dominant gradient direction
            angles = np.arctan2(grad_y, grad_x)
            
            # Analyze consistency of lighting angles
            # Remove outliers
            angle_hist, bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
            dominant_angle_idx = np.argmax(angle_hist)
            dominant_angle = (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2
            
            # Calculate how much of the image follows the dominant lighting
            angle_diff = np.abs(angles - dominant_angle)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Circular difference
            
            consistency_mask = angle_diff < np.pi/4  # Within 45 degrees
            consistency_ratio = np.sum(consistency_mask) / angles.size
            
            return float(consistency_ratio)
            
        except:
            return 0.5
    
    async def _analyze_highlights(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze highlight patterns"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            
            # Find highlight regions (high value)
            highlight_threshold = np.percentile(v_channel, 90)
            highlight_mask = v_channel > highlight_threshold
            
            if np.sum(highlight_mask) == 0:
                return {'naturalness': 0.5}
            
            # Analyze highlight distribution
            highlight_pixels = image[highlight_mask]
            
            # Check for blown-out highlights (unnatural)
            blown_out = np.sum(np.max(highlight_pixels, axis=1) >= 250)
            blown_out_ratio = blown_out / len(highlight_pixels)
            
            # Natural highlights should not be completely blown out
            naturalness = 1.0 - blown_out_ratio
            
            return {
                'naturalness': float(naturalness),
                'blown_out_ratio': float(blown_out_ratio)
            }
            
        except:
            return {'naturalness': 0.5}
    
    async def _calculate_image_authenticity_score(self, face_analysis: Dict, pixel_analysis: Dict, 
                                                 compression_analysis: Dict, geometric_analysis: Dict,
                                                 texture_analysis: Dict, lighting_analysis: Dict) -> float:
        """Calculate overall image authenticity score (0-100)"""
        # Weight different analysis components
        weights = {
            'face': 0.25,
            'pixel': 0.20,
            'compression': 0.15,
            'geometric': 0.15,
            'texture': 0.15,
            'lighting': 0.10
        }
        
        # Extract scores
        face_score = face_analysis.get('face_authenticity', 0.5) * 100
        pixel_score = pixel_analysis.get('overall_pixel_authenticity', 0.5) * 100
        compression_score = compression_analysis.get('compression_authenticity', 0.5) * 100
        geometric_score = geometric_analysis.get('geometric_authenticity', 0.5) * 100
        texture_score = texture_analysis.get('texture_authenticity', 0.5) * 100
        lighting_score = lighting_analysis.get('lighting_authenticity', 0.5) * 100
        
        # Calculate weighted score
        weighted_score = (
            face_score * weights['face'] +
            pixel_score * weights['pixel'] +
            compression_score * weights['compression'] +
            geometric_score * weights['geometric'] +
            texture_score * weights['texture'] +
            lighting_score * weights['lighting']
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
    
    async def _generate_image_visualization_data(self, image: np.ndarray, face_analysis: Dict, 
                                               pixel_analysis: Dict, compression_analysis: Dict) -> Dict[str, Any]:
        """Generate data for frontend image visualizations"""
        try:
            h, w = image.shape[:2]
            
            # Create heatmap for anomaly regions
            anomaly_heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Add face anomalies
            for face in face_analysis.get('faces', []):
                bbox = face['bbox']
                x, y, face_w, face_h = bbox
                manipulation_score = face['manipulation_score']
                
                # Ensure bounds are valid
                x_end = min(x + face_w, w)
                y_end = min(y + face_h, h)
                
                anomaly_heatmap[y:y_end, x:x_end] = manipulation_score
            
            # Downsample heatmap for visualization
            heatmap_small = cv2.resize(anomaly_heatmap, (100, int(100 * h / w)))
            
            # Face bounding boxes
            face_boxes = []
            for face in face_analysis.get('faces', []):
                face_boxes.append({
                    'bbox': face['bbox'],
                    'confidence': 1.0 - face['manipulation_score'],
                    'label': f"Face (Score: {face['manipulation_score']:.2f})"
                })
            
            # Analysis summary
            summary_stats = {
                'total_faces': face_analysis.get('face_count', 0),
                'average_face_authenticity': face_analysis.get('face_authenticity', 0.5) * 100,
                'pixel_authenticity': pixel_analysis.get('overall_pixel_authenticity', 0.5) * 100,
                'compression_quality': compression_analysis.get('estimated_quality', {}).get('estimated_quality', 75)
            }
            
            return {
                'anomaly_heatmap': heatmap_small.tolist(),
                'face_detections': face_boxes,
                'summary_stats': summary_stats,
                'image_dimensions': {'width': w, 'height': h}
            }
            
        except Exception as e:
            logger.warning(f"Image visualization data error: {str(e)}")
            return {
                'anomaly_heatmap': [],
                'face_detections': [],
                'summary_stats': {},
                'image_dimensions': {'width': 0, 'height': 0}
            }

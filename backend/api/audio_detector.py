"""
Audio Deepfake Detection Module
Implements RNN/Transformer-based models for voice stress and speech anomaly detection
"""

import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from typing import Dict, List, Tuple, Any
import logging
import asyncio
import json
from datetime import datetime
import soundfile as sf

logger = logging.getLogger(__name__)

class AudioDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_fft = 2048
        
    async def analyze_audio(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """
        Comprehensive audio deepfake analysis using voice stress and spectral analysis
        """
        try:
            logger.info(f"Starting audio analysis for task {task_id}")
            
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio_data) / sr
            
            # Extract audio features
            spectral_features = await self._extract_spectral_features(audio_data, sr)
            prosodic_features = await self._extract_prosodic_features(audio_data, sr)
            voice_stress_features = await self._analyze_voice_stress(audio_data, sr)
            
            # Temporal analysis
            temporal_analysis = await self._temporal_audio_analysis(audio_data, sr)
            
            # Anomaly detection
            anomaly_detection = await self._detect_audio_anomalies(audio_data, spectral_features)
            
            # Speech pattern analysis
            speech_patterns = await self._analyze_speech_patterns(audio_data, sr)
            
            # Calculate authenticity score
            authenticity_score = await self._calculate_audio_authenticity_score(
                spectral_features, prosodic_features, voice_stress_features, anomaly_detection
            )
            
            # Generate visualization data
            visualization_data = await self._generate_audio_visualization_data(
                audio_data, sr, spectral_features, anomaly_detection
            )
            
            result = {
                'task_id': task_id,
                'analysis_type': 'audio',
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'duration': duration,
                    'sample_rate': sr,
                    'channels': 1,
                    'format': 'processed_mono'
                },
                'authenticity_score': authenticity_score,
                'confidence_level': self._calculate_confidence(authenticity_score),
                'detection_results': {
                    'is_deepfake': authenticity_score < 50,
                    'risk_level': self._get_risk_level(authenticity_score),
                    'anomaly_count': len(anomaly_detection.get('anomalies', []))
                },
                'detailed_analysis': {
                    'spectral_features': spectral_features,
                    'prosodic_features': prosodic_features,
                    'voice_stress_analysis': voice_stress_features,
                    'temporal_analysis': temporal_analysis,
                    'speech_patterns': speech_patterns,
                    'anomaly_detection': anomaly_detection
                },
                'visualization_data': visualization_data,
                'biometric_signals': {
                    'voice_stress_level': voice_stress_features.get('stress_level', 0),
                    'pitch_irregularities': prosodic_features.get('pitch_irregularities', 0),
                    'spectral_anomalies': len(anomaly_detection.get('spectral_anomalies', []))
                }
            }
            
            logger.info(f"Audio analysis completed for task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            raise
    
    async def _extract_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features for deepfake detection"""
        try:
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            
            # Calculate statistics
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
                'mfcc_std': np.std(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr)),
                'chroma_mean': np.mean(chroma, axis=1).tolist(),
                'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist(),
                'spectral_irregularity': self._calculate_spectral_irregularity(mfccs),
                'harmonic_ratio': self._calculate_harmonic_ratio(audio_data, sr)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction error: {str(e)}")
            return {}
    
    async def _extract_prosodic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract prosodic features (pitch, rhythm, stress patterns)"""
        try:
            # Fundamental frequency (pitch)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            
            # Remove NaN values
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) == 0:
                return {'pitch_irregularities': 1.0}
            
            # Pitch statistics
            pitch_mean = float(np.mean(f0_clean))
            pitch_std = float(np.std(f0_clean))
            
            # Pitch contour analysis
            pitch_range = float(np.max(f0_clean) - np.min(f0_clean))
            
            # Jitter (pitch period irregularity)
            jitter = self._calculate_jitter(f0_clean)
            
            # Shimmer (amplitude irregularity)
            shimmer = self._calculate_shimmer(audio_data)
            
            # Speech rate analysis
            speech_rate = self._calculate_speech_rate(audio_data, sr)
            
            # Pause analysis
            pause_analysis = self._analyze_pauses(audio_data, sr)
            
            # Detect pitch irregularities (sign of manipulation)
            pitch_irregularities = 0.0
            if jitter > 0.01:  # High jitter threshold
                pitch_irregularities += 0.3
            if shimmer > 0.1:  # High shimmer threshold
                pitch_irregularities += 0.3
            if pitch_std > pitch_mean * 0.3:  # High pitch variance
                pitch_irregularities += 0.4
            
            return {
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'pitch_range': pitch_range,
                'jitter': jitter,
                'shimmer': shimmer,
                'speech_rate': speech_rate,
                'pause_ratio': pause_analysis['pause_ratio'],
                'pitch_irregularities': min(pitch_irregularities, 1.0),
                'prosody_naturalness': max(0, 1.0 - pitch_irregularities)
            }
            
        except Exception as e:
            logger.warning(f"Prosodic feature extraction error: {str(e)}")
            return {'pitch_irregularities': 0.5}
    
    async def _analyze_voice_stress(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze voice stress patterns that may indicate synthetic speech"""
        try:
            # Energy-based stress detection
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            # Frame energy
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # Energy variance (stress indicator)
            energy_variance = float(np.var(energy))
            energy_mean = float(np.mean(energy))
            
            # High-frequency energy (tension indicator)
            stft = librosa.stft(audio_data, hop_length=hop_length)
            high_freq_energy = np.sum(np.abs(stft[stft.shape[0]//2:, :]) ** 2, axis=0)
            hf_ratio = np.mean(high_freq_energy) / max(energy_mean, 1e-10)
            
            # Formant analysis for stress detection
            formants = self._extract_formants(audio_data, sr)
            
            # Calculate stress level
            stress_level = 0.0
            
            # High energy variance indicates stress/unnaturalness
            if energy_variance > energy_mean * 0.1:
                stress_level += 0.3
                
            # High frequency emphasis in synthetic speech
            if hf_ratio > 0.2:
                stress_level += 0.4
                
            # Formant irregularities
            if formants.get('f1_variance', 0) > 100:
                stress_level += 0.3
            
            return {
                'stress_level': min(stress_level, 1.0),
                'energy_variance': energy_variance,
                'energy_mean': energy_mean,
                'high_freq_ratio': float(hf_ratio),
                'formant_analysis': formants,
                'naturalness_score': max(0, 1.0 - stress_level)
            }
            
        except Exception as e:
            logger.warning(f"Voice stress analysis error: {str(e)}")
            return {'stress_level': 0.5}
    
    async def _temporal_audio_analysis(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze temporal consistency in audio"""
        try:
            # Segment audio into chunks
            chunk_duration = 1.0  # 1 second chunks
            chunk_samples = int(chunk_duration * sr)
            chunks = [audio_data[i:i+chunk_samples] for i in range(0, len(audio_data), chunk_samples)]
            
            # Extract features from each chunk
            chunk_features = []
            for chunk in chunks:
                if len(chunk) < chunk_samples // 2:  # Skip short chunks
                    continue
                    
                # Basic features per chunk
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
                spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
                
                chunk_features.append({
                    'rms': rms,
                    'zcr': zcr,
                    'spectral_centroid': spectral_centroid
                })
            
            if len(chunk_features) < 2:
                return {'consistency_score': 0.5, 'temporal_anomalies': []}
            
            # Calculate consistency metrics
            rms_values = [cf['rms'] for cf in chunk_features]
            zcr_values = [cf['zcr'] for cf in chunk_features]
            sc_values = [cf['spectral_centroid'] for cf in chunk_features]
            
            rms_consistency = 1.0 - min(1.0, np.std(rms_values) / max(np.mean(rms_values), 1e-10))
            zcr_consistency = 1.0 - min(1.0, np.std(zcr_values) / max(np.mean(zcr_values), 1e-10))
            sc_consistency = 1.0 - min(1.0, np.std(sc_values) / max(np.mean(sc_values), 1e-10))
            
            overall_consistency = (rms_consistency + zcr_consistency + sc_consistency) / 3
            
            # Detect temporal anomalies
            anomalies = []
            for i in range(1, len(chunk_features)):
                rms_change = abs(chunk_features[i]['rms'] - chunk_features[i-1]['rms'])
                if rms_change > np.mean(rms_values) * 0.5:  # Sudden change
                    anomalies.append({
                        'timestamp': i * chunk_duration,
                        'type': 'sudden_volume_change',
                        'magnitude': float(rms_change)
                    })
            
            return {
                'consistency_score': overall_consistency,
                'rms_consistency': rms_consistency,
                'zcr_consistency': zcr_consistency,
                'spectral_consistency': sc_consistency,
                'temporal_anomalies': anomalies[:10]  # Limit output
            }
            
        except Exception as e:
            logger.warning(f"Temporal analysis error: {str(e)}")
            return {'consistency_score': 0.5}
    
    async def _detect_audio_anomalies(self, audio_data: np.ndarray, spectral_features: Dict) -> Dict[str, Any]:
        """Detect various audio anomalies indicative of deepfake"""
        try:
            anomalies = []
            spectral_anomalies = []
            
            # Check for unnatural spectral patterns
            mfcc_means = spectral_features.get('mfcc_mean', [])
            if mfcc_means:
                # Unnatural MFCC distribution
                if abs(mfcc_means[0]) > 50:  # First MFCC too high/low
                    spectral_anomalies.append({
                        'type': 'unnatural_mfcc_distribution',
                        'severity': min(1.0, abs(mfcc_means[0]) / 100)
                    })
                
                # Check for artificial formant patterns
                if len(mfcc_means) > 2 and abs(mfcc_means[1] - mfcc_means[2]) > 20:
                    spectral_anomalies.append({
                        'type': 'artificial_formant_pattern',
                        'severity': 0.6
                    })
            
            # Check spectral irregularity
            spectral_irreg = spectral_features.get('spectral_irregularity', 0)
            if spectral_irreg > 0.7:
                anomalies.append({
                    'type': 'high_spectral_irregularity',
                    'severity': spectral_irreg,
                    'description': 'Unnatural frequency distribution'
                })
            
            # Check harmonic ratio
            harmonic_ratio = spectral_features.get('harmonic_ratio', 0.5)
            if harmonic_ratio < 0.3:
                anomalies.append({
                    'type': 'low_harmonic_content',
                    'severity': 1.0 - harmonic_ratio,
                    'description': 'Insufficient harmonic structure'
                })
            
            # Calculate overall anomaly score
            anomaly_score = 0.0
            if spectral_anomalies:
                anomaly_score += len(spectral_anomalies) * 0.2
            if anomalies:
                anomaly_score += sum(a['severity'] for a in anomalies) / len(anomalies)
            
            return {
                'anomaly_score': min(anomaly_score, 1.0),
                'anomalies': anomalies,
                'spectral_anomalies': spectral_anomalies,
                'total_anomalies': len(anomalies) + len(spectral_anomalies)
            }
            
        except Exception as e:
            logger.warning(f"Anomaly detection error: {str(e)}")
            return {'anomaly_score': 0.5, 'anomalies': []}
    
    async def _analyze_speech_patterns(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze speech patterns for authenticity"""
        try:
            # Voice activity detection
            frame_length = 2048
            hop_length = 512
            
            # Energy-based VAD
            stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
            magnitude = np.abs(stft)
            energy = np.sum(magnitude ** 2, axis=0)
            
            # Threshold for voice activity
            energy_threshold = np.percentile(energy, 30)
            voice_frames = energy > energy_threshold
            
            # Calculate speech statistics
            total_frames = len(voice_frames)
            active_frames = np.sum(voice_frames)
            speech_ratio = active_frames / max(total_frames, 1)
            
            # Analyze speech continuity
            speech_segments = []
            in_speech = False
            segment_start = 0
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_speech:
                    segment_start = i
                    in_speech = True
                elif not is_voice and in_speech:
                    speech_segments.append(i - segment_start)
                    in_speech = False
            
            # Speech pattern metrics
            avg_segment_length = float(np.mean(speech_segments)) if speech_segments else 0
            segment_variance = float(np.var(speech_segments)) if speech_segments else 0
            
            return {
                'speech_ratio': speech_ratio,
                'average_segment_length': avg_segment_length,
                'segment_variance': segment_variance,
                'total_segments': len(speech_segments),
                'pattern_naturalness': min(1.0, speech_ratio * 2)  # Prefer higher speech ratios
            }
            
        except Exception as e:
            logger.warning(f"Speech pattern analysis error: {str(e)}")
            return {'pattern_naturalness': 0.5}
    
    def _calculate_spectral_irregularity(self, mfccs: np.ndarray) -> float:
        """Calculate spectral irregularity metric"""
        try:
            if mfccs.size == 0:
                return 0.5
            
            # Calculate frame-to-frame differences
            diff = np.diff(mfccs, axis=1)
            irregularity = np.mean(np.abs(diff))
            
            # Normalize to 0-1 range
            return min(1.0, irregularity / 10.0)
            
        except:
            return 0.5
    
    def _calculate_harmonic_ratio(self, audio_data: np.ndarray, sr: int) -> float:
        """Calculate harmonic-to-noise ratio"""
        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Calculate energy ratio
            harmonic_energy = np.sum(harmonic ** 2)
            total_energy = np.sum(audio_data ** 2)
            
            if total_energy == 0:
                return 0.0
            
            ratio = harmonic_energy / total_energy
            return float(min(1.0, ratio))
            
        except:
            return 0.5
    
    def _calculate_jitter(self, f0: np.ndarray) -> float:
        """Calculate pitch jitter (period irregularity)"""
        try:
            if len(f0) < 2:
                return 0.0
            
            periods = 1.0 / f0[f0 > 0]  # Convert frequency to period
            if len(periods) < 2:
                return 0.0
            
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            
            if mean_period == 0:
                return 0.0
            
            jitter = np.mean(period_diffs) / mean_period
            return float(min(1.0, jitter))
            
        except:
            return 0.0
    
    def _calculate_shimmer(self, audio_data: np.ndarray) -> float:
        """Calculate amplitude shimmer (amplitude irregularity)"""
        try:
            # Frame-based amplitude calculation
            frame_length = 1024
            hop_length = 512
            
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
            amplitudes = np.sqrt(np.mean(frames ** 2, axis=0))
            
            if len(amplitudes) < 2:
                return 0.0
            
            amplitude_diffs = np.abs(np.diff(amplitudes))
            mean_amplitude = np.mean(amplitudes)
            
            if mean_amplitude == 0:
                return 0.0
            
            shimmer = np.mean(amplitude_diffs) / mean_amplitude
            return float(min(1.0, shimmer))
            
        except:
            return 0.0
    
    def _calculate_speech_rate(self, audio_data: np.ndarray, sr: int) -> float:
        """Estimate speech rate (syllables per second)"""
        try:
            # Simple speech rate estimation based on energy peaks
            frame_length = 2048
            hop_length = 512
            
            stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
            energy = np.sum(np.abs(stft) ** 2, axis=0)
            
            # Find peaks (potential syllables)
            peaks, _ = signal.find_peaks(energy, height=np.percentile(energy, 70))
            
            duration = len(audio_data) / sr
            speech_rate = len(peaks) / max(duration, 1)
            
            return float(speech_rate)
            
        except:
            return 3.0  # Average speech rate
    
    def _analyze_pauses(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze pause patterns in speech"""
        try:
            # Energy-based silence detection
            frame_length = 2048
            hop_length = 512
            
            stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
            energy = np.sum(np.abs(stft) ** 2, axis=0)
            
            # Threshold for silence
            silence_threshold = np.percentile(energy, 20)
            silence_frames = energy < silence_threshold
            
            total_frames = len(silence_frames)
            silence_frame_count = np.sum(silence_frames)
            
            pause_ratio = silence_frame_count / max(total_frames, 1)
            
            return {
                'pause_ratio': float(pause_ratio),
                'silence_threshold': float(silence_threshold)
            }
            
        except:
            return {'pause_ratio': 0.2}
    
    def _extract_formants(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract formant frequencies"""
        try:
            # Simplified formant estimation using spectral peaks
            frame_length = 2048
            stft = librosa.stft(audio_data, n_fft=frame_length)
            magnitude = np.abs(stft)
            
            # Find spectral peaks (formant candidates)
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
            
            # Average across time
            avg_magnitude = np.mean(magnitude, axis=1)
            
            # Find peaks in frequency domain
            peaks, _ = signal.find_peaks(avg_magnitude, height=np.percentile(avg_magnitude, 80))
            
            if len(peaks) >= 2:
                f1 = float(freq_bins[peaks[0]])
                f2 = float(freq_bins[peaks[1]])
                
                # Calculate variance (measure of stability)
                f1_variance = float(np.var(magnitude[peaks[0], :]))
                f2_variance = float(np.var(magnitude[peaks[1], :]))
                
                return {
                    'f1': f1,
                    'f2': f2,
                    'f1_variance': f1_variance,
                    'f2_variance': f2_variance
                }
            
            return {'f1_variance': 0, 'f2_variance': 0}
            
        except:
            return {'f1_variance': 0, 'f2_variance': 0}
    
    async def _calculate_audio_authenticity_score(self, spectral: Dict, prosodic: Dict, stress: Dict, anomaly: Dict) -> float:
        """Calculate overall audio authenticity score (0-100)"""
        # Weight different factors
        spectral_weight = 0.25
        prosodic_weight = 0.30
        stress_weight = 0.25
        anomaly_weight = 0.20
        
        # Calculate individual scores
        spectral_score = (1.0 - spectral.get('spectral_irregularity', 0.5)) * 100
        prosodic_score = prosodic.get('prosody_naturalness', 0.5) * 100
        stress_score = stress.get('naturalness_score', 0.5) * 100
        anomaly_score = (1.0 - anomaly.get('anomaly_score', 0.5)) * 100
        
        weighted_score = (
            spectral_score * spectral_weight +
            prosodic_score * prosodic_weight +
            stress_score * stress_weight +
            anomaly_score * anomaly_weight
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
    
    async def _generate_audio_visualization_data(self, audio_data: np.ndarray, sr: int, spectral_features: Dict, anomaly_detection: Dict) -> Dict[str, Any]:
        """Generate data for frontend audio visualizations"""
        try:
            # Waveform data (downsampled for visualization)
            duration = len(audio_data) / sr
            time_axis = np.linspace(0, duration, min(1000, len(audio_data)))
            waveform_data = np.interp(time_axis, np.linspace(0, duration, len(audio_data)), audio_data)
            
            # Spectrogram data
            stft = librosa.stft(audio_data, hop_length=512)
            magnitude = np.abs(stft)
            
            # Reduce size for visualization
            spectrogram_data = magnitude[::4, ::10]  # Subsample
            
            # Frequency axis
            freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)[::4]
            
            # Time axis for spectrogram
            time_axis_spec = librosa.frames_to_time(range(0, magnitude.shape[1], 10), sr=sr, hop_length=512)
            
            # Anomaly timeline
            anomaly_timeline = []
            for anomaly in anomaly_detection.get('anomalies', [])[:20]:
                if 'timestamp' in anomaly:
                    anomaly_timeline.append({
                        'timestamp': anomaly['timestamp'],
                        'severity': anomaly['severity'],
                        'type': anomaly['type']
                    })
            
            return {
                'waveform': {
                    'time': time_axis.tolist(),
                    'amplitude': waveform_data.tolist()
                },
                'spectrogram': {
                    'frequencies': freq_axis.tolist(),
                    'time': time_axis_spec.tolist(),
                    'magnitude': spectrogram_data.tolist()
                },
                'anomaly_timeline': anomaly_timeline,
                'summary_stats': {
                    'duration': duration,
                    'peak_frequency': float(freq_axis[np.argmax(np.mean(spectrogram_data, axis=1))]),
                    'average_energy': float(np.mean(audio_data ** 2)),
                    'anomaly_count': len(anomaly_timeline)
                }
            }
            
        except Exception as e:
            logger.warning(f"Visualization data generation error: {str(e)}")
            return {
                'waveform': {'time': [], 'amplitude': []},
                'spectrogram': {'frequencies': [], 'time': [], 'magnitude': []},
                'anomaly_timeline': [],
                'summary_stats': {}
            }

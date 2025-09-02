"""
Sample Dataset Generator and Manager
Creates realistic sample datasets for deepfake detection demonstration
Includes real and AI-generated content across image, video, and audio modalities
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import base64
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleDatasetManager:
    """
    Manages sample datasets for deepfake detection demonstration
    """
    
    def __init__(self, dataset_root: str = "sample_datasets"):
        """
        Initialize dataset manager
        Args:
            dataset_root: Root directory for sample datasets
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(exist_ok=True)
        
        # Dataset structure
        self.categories = {
            'real': {
                'images': 'real/images',
                'videos': 'real/videos', 
                'audio': 'real/audio'
            },
            'ai_generated': {
                'images': 'ai_generated/images',
                'videos': 'ai_generated/videos',
                'audio': 'ai_generated/audio'
            }
        }
        
        # Create directory structure
        self._create_directory_structure()
        
        # Sample metadata
        self.metadata = {
            'created_date': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'Sample dataset for deepfake detection demonstration',
            'categories': self.categories,
            'samples': {}
        }
        
        logger.info(f"Dataset manager initialized at: {self.dataset_root}")
    
    def _create_directory_structure(self):
        """Create the directory structure for datasets"""
        for category, modalities in self.categories.items():
            for modality, path in modalities.items():
                full_path = self.dataset_root / path
                full_path.mkdir(parents=True, exist_ok=True)
    
    def create_sample_images(self) -> Dict:
        """
        Create sample image dataset entries
        Note: In production, these would be actual image files
        """
        image_samples = {
            'real': [
                {
                    'filename': 'real_portrait_001.jpg',
                    'description': 'Professional headshot - natural lighting and expression',
                    'source': 'Stock photography',
                    'characteristics': ['natural_skin_texture', 'consistent_lighting', 'realistic_shadows'],
                    'confidence_indicators': ['eye_symmetry', 'natural_blinking', 'consistent_face_geometry'],
                    'file_size': '2.4MB',
                    'resolution': '1920x1080',
                    'metadata': {
                        'camera': 'DSLR Canon EOS R5',
                        'lighting': 'Natural window light',
                        'compression': 'High quality JPEG'
                    }
                },
                {
                    'filename': 'real_candid_002.jpg',
                    'description': 'Casual photo with natural expression and environment',
                    'source': 'Authentic photography',
                    'characteristics': ['natural_motion_blur', 'environmental_consistency', 'authentic_emotions'],
                    'confidence_indicators': ['natural_micro_expressions', 'consistent_depth_of_field'],
                    'file_size': '1.8MB',
                    'resolution': '1440x1080',
                    'metadata': {
                        'setting': 'Outdoor natural lighting',
                        'subject': 'Spontaneous expression',
                        'authenticity_score': 0.98
                    }
                },
                {
                    'filename': 'real_group_003.jpg',
                    'description': 'Group photo with multiple subjects in natural setting',
                    'source': 'Event photography',
                    'characteristics': ['consistent_lighting_across_subjects', 'natural_interactions'],
                    'confidence_indicators': ['synchronized_shadows', 'natural_group_dynamics'],
                    'file_size': '3.1MB',
                    'resolution': '2560x1440'
                }
            ],
            'ai_generated': [
                {
                    'filename': 'ai_stylegan_001.jpg',
                    'description': 'StyleGAN3 generated portrait with high realism',
                    'source': 'StyleGAN3 neural network',
                    'characteristics': ['slight_asymmetry', 'perfect_skin_texture', 'artificial_sharpness'],
                    'detection_indicators': ['uncanny_valley_effect', 'too_perfect_features', 'subtle_geometric_inconsistencies'],
                    'file_size': '2.2MB',
                    'resolution': '1024x1024',
                    'generation_params': {
                        'model': 'StyleGAN3',
                        'seed': 42,
                        'truncation': 0.7,
                        'generation_time': '3.2s'
                    }
                },
                {
                    'filename': 'ai_diffusion_002.jpg',
                    'description': 'Stable Diffusion generated portrait with realistic details',
                    'source': 'Stable Diffusion v2.1',
                    'characteristics': ['hyperrealistic_details', 'subtle_artifacts', 'perfect_symmetry'],
                    'detection_indicators': ['compression_artifacts', 'unnatural_smoothness', 'pixel_level_inconsistencies'],
                    'file_size': '1.9MB',
                    'resolution': '768x768',
                    'generation_params': {
                        'prompt': 'professional headshot, studio lighting, 8k quality',
                        'steps': 50,
                        'cfg_scale': 7.5,
                        'sampler': 'DPM++ 2M Karras'
                    }
                },
                {
                    'filename': 'ai_deepfake_003.jpg',
                    'description': 'Face-swapped image using advanced deepfake technology',
                    'source': 'DeepFaceLab synthesis',
                    'characteristics': ['face_boundary_artifacts', 'color_mismatch', 'resolution_inconsistency'],
                    'detection_indicators': ['blending_artifacts', 'temporal_inconsistency', 'facial_landmark_drift'],
                    'file_size': '2.0MB',
                    'resolution': '1280x720',
                    'synthesis_details': {
                        'base_model': 'DeepFaceLab SAEHD',
                        'training_iterations': 100000,
                        'face_swap_quality': 'High'
                    }
                }
            ]
        }
        
        # Save sample files metadata
        for category, samples in image_samples.items():
            for sample in samples:
                # Create placeholder file (in production, would be actual image)
                file_path = self.dataset_root / self.categories[category]['images'] / sample['filename']
                self._create_placeholder_file(file_path, 'image', sample)
        
        return image_samples
    
    def create_sample_videos(self) -> Dict:
        """
        Create sample video dataset entries
        """
        video_samples = {
            'real': [
                {
                    'filename': 'real_interview_001.mp4',
                    'description': 'Natural interview footage with consistent facial expressions',
                    'duration': '45 seconds',
                    'fps': 30,
                    'resolution': '1920x1080',
                    'characteristics': ['natural_head_movements', 'consistent_lighting', 'authentic_speech_sync'],
                    'confidence_indicators': ['temporal_consistency', 'natural_blinking_patterns'],
                    'file_size': '12.3MB',
                    'metadata': {
                        'camera': 'Professional video camera',
                        'audio_sync': 'Perfect synchronization',
                        'compression': 'H.264 high quality'
                    }
                },
                {
                    'filename': 'real_presentation_002.mp4',
                    'description': 'Business presentation with natural gestures and expressions',
                    'duration': '60 seconds',
                    'fps': 24,
                    'resolution': '1280x720',
                    'characteristics': ['natural_gesticulation', 'consistent_background', 'authentic_emotions'],
                    'confidence_indicators': ['natural_micro_expressions', 'consistent_shadow_movement'],
                    'file_size': '8.7MB'
                }
            ],
            'ai_generated': [
                {
                    'filename': 'ai_faceswap_001.mp4',
                    'description': 'Face-swapped video with temporal inconsistencies',
                    'duration': '30 seconds',
                    'fps': 30,
                    'resolution': '1024x768',
                    'characteristics': ['facial_boundary_flicker', 'lighting_inconsistency', 'unnatural_expressions'],
                    'detection_indicators': ['temporal_artifacts', 'face_warping', 'color_bleeding'],
                    'file_size': '15.2MB',
                    'synthesis_details': {
                        'method': 'First Order Motion Model',
                        'source_video': 'Training dataset',
                        'driving_video': 'Target performance'
                    }
                },
                {
                    'filename': 'ai_talking_head_002.mp4',
                    'description': 'AI-generated talking head with synthesized speech',
                    'duration': '40 seconds',
                    'fps': 25,
                    'resolution': '512x512',
                    'characteristics': ['perfect_lip_sync', 'unnatural_stillness', 'artificial_background'],
                    'detection_indicators': ['lack_of_micro_movements', 'synthetic_voice_patterns'],
                    'file_size': '6.8MB',
                    'generation_params': {
                        'model': 'Wav2Lip + Real-Time Face Reenactment',
                        'audio_source': 'Text-to-speech synthesis',
                        'quality_setting': 'High'
                    }
                }
            ]
        }
        
        # Save sample files metadata
        for category, samples in video_samples.items():
            for sample in samples:
                file_path = self.dataset_root / self.categories[category]['videos'] / sample['filename']
                self._create_placeholder_file(file_path, 'video', sample)
        
        return video_samples
    
    def create_sample_audio(self) -> Dict:
        """
        Create sample audio dataset entries
        """
        audio_samples = {
            'real': [
                {
                    'filename': 'real_speech_001.wav',
                    'description': 'Natural human speech with authentic vocal characteristics',
                    'duration': '30 seconds',
                    'sample_rate': 44100,
                    'bit_depth': 16,
                    'characteristics': ['natural_breathing', 'vocal_fry', 'micro_hesitations'],
                    'confidence_indicators': ['formant_consistency', 'natural_prosody', 'authentic_emotions'],
                    'file_size': '2.6MB',
                    'metadata': {
                        'recording_environment': 'Studio with acoustic treatment',
                        'microphone': 'Condenser microphone',
                        'noise_floor': '-60dB'
                    }
                },
                {
                    'filename': 'real_conversation_002.wav',
                    'description': 'Natural conversation with multiple speakers',
                    'duration': '45 seconds',
                    'sample_rate': 48000,
                    'characteristics': ['natural_turn_taking', 'overlapping_speech', 'environmental_sounds'],
                    'confidence_indicators': ['speaker_separation', 'natural_interruptions'],
                    'file_size': '4.1MB'
                }
            ],
            'ai_generated': [
                {
                    'filename': 'ai_tts_001.wav',
                    'description': 'Text-to-speech generated voice with high quality',
                    'duration': '25 seconds',
                    'sample_rate': 22050,
                    'characteristics': ['perfect_pronunciation', 'lack_of_variation', 'mechanical_rhythm'],
                    'detection_indicators': ['spectral_artifacts', 'unnatural_prosody', 'missing_breath_sounds'],
                    'file_size': '1.1MB',
                    'synthesis_details': {
                        'model': 'Tacotron 2 + WaveGlow',
                        'voice_identity': 'Female professional speaker',
                        'quality_setting': 'High fidelity'
                    }
                },
                {
                    'filename': 'ai_voice_clone_002.wav',
                    'description': 'Voice cloning of specific speaker using neural synthesis',
                    'duration': '35 seconds',
                    'sample_rate': 24000,
                    'characteristics': ['cloned_vocal_identity', 'synthetic_emotional_expression'],
                    'detection_indicators': ['phase_inconsistencies', 'unnatural_formant_transitions'],
                    'file_size': '1.7MB',
                    'cloning_details': {
                        'source_speaker': 'Target voice samples',
                        'training_duration': '20 minutes of source audio',
                        'cloning_accuracy': '95%'
                    }
                }
            ]
        }
        
        # Save sample files metadata
        for category, samples in audio_samples.items():
            for sample in samples:
                file_path = self.dataset_root / self.categories[category]['audio'] / sample['filename']
                self._create_placeholder_file(file_path, 'audio', sample)
        
        return audio_samples
    
    def _create_placeholder_file(self, file_path: Path, file_type: str, metadata: Dict):
        """
        Create placeholder file with embedded metadata
        In production, these would be actual media files
        """
        # Create a JSON metadata file alongside
        metadata_path = file_path.with_suffix('.json')
        
        placeholder_content = {
            'type': file_type,
            'placeholder': True,
            'metadata': metadata,
            'created': datetime.now().isoformat(),
            'note': f'This is a placeholder for demonstration. In production, this would be an actual {file_type} file.'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(placeholder_content, f, indent=2)
        
        # Create minimal placeholder file
        with open(file_path, 'w') as f:
            f.write(f"# Placeholder {file_type} file for {metadata.get('filename', 'demo')}\n")
            f.write(f"# This represents a {metadata.get('description', 'sample file')} \n")
    
    def create_multimodal_test_cases(self) -> List[Dict]:
        """
        Create test cases that combine multiple modalities
        """
        test_cases = [
            {
                'case_id': 'TC001',
                'name': 'Authentic Multi-Modal Content',
                'description': 'Real person across all modalities - should show high consistency',
                'files': {
                    'image': 'real_portrait_001.jpg',
                    'video': 'real_interview_001.mp4',
                    'audio': 'real_speech_001.wav'
                },
                'expected_result': {
                    'prediction': 'Real',
                    'confidence_range': [85, 95],
                    'agreement_expected': 'high',
                    'risk_level': 'Low'
                },
                'test_focus': 'Baseline authentic content detection'
            },
            {
                'case_id': 'TC002',
                'name': 'Full AI-Generated Content',
                'description': 'AI-generated content across all modalities',
                'files': {
                    'image': 'ai_stylegan_001.jpg',
                    'video': 'ai_faceswap_001.mp4',
                    'audio': 'ai_tts_001.wav'
                },
                'expected_result': {
                    'prediction': 'AI-Generated',
                    'confidence_range': [80, 95],
                    'agreement_expected': 'high',
                    'risk_level': 'High'
                },
                'test_focus': 'Full synthetic content detection'
            },
            {
                'case_id': 'TC003',
                'name': 'Mixed Content - Real Image, Fake Audio',
                'description': 'Real image with synthesized voice - common attack vector',
                'files': {
                    'image': 'real_portrait_001.jpg',
                    'video': None,
                    'audio': 'ai_voice_clone_002.wav'
                },
                'expected_result': {
                    'prediction': 'AI-Generated',
                    'confidence_range': [70, 85],
                    'agreement_expected': 'medium',
                    'risk_level': 'Medium'
                },
                'test_focus': 'Cross-modal inconsistency detection'
            },
            {
                'case_id': 'TC004',
                'name': 'Sophisticated Deepfake Video',
                'description': 'High-quality face swap with original audio',
                'files': {
                    'image': None,
                    'video': 'ai_faceswap_001.mp4',
                    'audio': 'real_speech_001.wav'
                },
                'expected_result': {
                    'prediction': 'AI-Generated',
                    'confidence_range': [75, 90],
                    'agreement_expected': 'medium',
                    'risk_level': 'High'
                },
                'test_focus': 'Video manipulation with authentic audio'
            },
            {
                'case_id': 'TC005',
                'name': 'Edge Case - Low Quality Real Content',
                'description': 'Authentic but low quality content that might trigger false positives',
                'files': {
                    'image': 'real_candid_002.jpg',
                    'video': None,
                    'audio': 'real_conversation_002.wav'
                },
                'expected_result': {
                    'prediction': 'Real',
                    'confidence_range': [60, 80],
                    'agreement_expected': 'medium',
                    'risk_level': 'Low'
                },
                'test_focus': 'Robustness against quality variations'
            }
        ]
        
        # Save test cases
        test_cases_path = self.dataset_root / 'test_cases.json'
        with open(test_cases_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        return test_cases
    
    def generate_demo_scenarios(self) -> Dict:
        """
        Generate comprehensive demo scenarios for live presentation
        """
        scenarios = {
            'live_demo_sequence': [
                {
                    'step': 1,
                    'title': 'Baseline Authentication',
                    'description': 'Demonstrate detection on clearly real content',
                    'test_case': 'TC001',
                    'expected_outcome': 'High confidence Real classification',
                    'talking_points': [
                        'All modalities agree on authenticity',
                        'Natural characteristics detected',
                        'Confidence scores align across modalities'
                    ]
                },
                {
                    'step': 2,
                    'title': 'Obvious AI Detection',
                    'description': 'Show detection of clearly synthetic content',
                    'test_case': 'TC002',
                    'expected_outcome': 'High confidence AI-Generated classification',
                    'talking_points': [
                        'Multiple synthesis artifacts detected',
                        'Temporal inconsistencies in video',
                        'Spectral anomalies in audio'
                    ]
                },
                {
                    'step': 3,
                    'title': 'Sophisticated Attack Detection',
                    'description': 'Challenge the system with high-quality deepfakes',
                    'test_case': 'TC004',
                    'expected_outcome': 'Successful detection despite quality',
                    'talking_points': [
                        'Subtle artifacts still detectable',
                        'Multi-modal analysis increases accuracy',
                        'Fusion strategy adapts to evidence'
                    ]
                },
                {
                    'step': 4,
                    'title': 'Cross-Modal Inconsistency',
                    'description': 'Detect mismatched real and synthetic content',
                    'test_case': 'TC003',
                    'expected_outcome': 'Detection through modality disagreement',
                    'talking_points': [
                        'Individual modalities may disagree',
                        'Fusion system weighs evidence',
                        'Conservative approach for safety'
                    ]
                }
            ],
            'technical_demonstration': {
                'confidence_visualization': 'Show real-time confidence meters',
                'attention_heatmaps': 'Display model attention on facial regions',
                'spectral_analysis': 'Show audio frequency analysis',
                'temporal_consistency': 'Visualize frame-by-frame analysis',
                'anomaly_highlighting': 'Mark detected artifacts'
            },
            'interactive_features': {
                'upload_testing': 'Allow audience to test their own content',
                'parameter_adjustment': 'Demonstrate fusion weight tuning',
                'modality_ablation': 'Show single vs multi-modal performance',
                'real_time_analysis': 'Live webcam/microphone analysis'
            }
        }
        
        # Save demo scenarios
        scenarios_path = self.dataset_root / 'demo_scenarios.json'
        with open(scenarios_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        return scenarios
    
    def initialize_complete_dataset(self) -> Dict:
        """
        Initialize the complete sample dataset with all components
        """
        logger.info("Initializing complete sample dataset...")
        
        # Generate all sample types
        image_samples = self.create_sample_images()
        video_samples = self.create_sample_videos()
        audio_samples = self.create_sample_audio()
        test_cases = self.create_multimodal_test_cases()
        demo_scenarios = self.generate_demo_scenarios()
        
        # Update metadata
        self.metadata.update({
            'samples': {
                'images': image_samples,
                'videos': video_samples,
                'audio': audio_samples
            },
            'test_cases': test_cases,
            'demo_scenarios': demo_scenarios,
            'statistics': {
                'total_real_samples': len(image_samples['real']) + len(video_samples['real']) + len(audio_samples['real']),
                'total_ai_samples': len(image_samples['ai_generated']) + len(video_samples['ai_generated']) + len(audio_samples['ai_generated']),
                'total_test_cases': len(test_cases),
                'modalities_covered': ['image', 'video', 'audio'],
                'demo_scenarios': len(demo_scenarios['live_demo_sequence'])
            }
        })
        
        # Save master metadata
        metadata_path = self.dataset_root / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Sample dataset initialized successfully!")
        logger.info(f"Total samples: {self.metadata['statistics']['total_real_samples'] + self.metadata['statistics']['total_ai_samples']}")
        logger.info(f"Test cases: {self.metadata['statistics']['total_test_cases']}")
        logger.info(f"Demo scenarios: {self.metadata['statistics']['demo_scenarios']}")
        
        return self.metadata
    
    def get_sample_for_testing(self, test_case_id: str) -> Optional[Dict]:
        """
        Get a specific test case for demonstration
        Args:
            test_case_id: ID of the test case (e.g., 'TC001')
        Returns:
            Test case data or None if not found
        """
        test_cases_path = self.dataset_root / 'test_cases.json'
        if not test_cases_path.exists():
            return None
        
        with open(test_cases_path, 'r') as f:
            test_cases = json.load(f)
        
        for case in test_cases:
            if case['case_id'] == test_case_id:
                return case
        
        return None
    
    def list_available_samples(self) -> Dict:
        """
        List all available samples organized by category and modality
        """
        metadata_path = self.dataset_root / 'dataset_metadata.json'
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata.get('samples', {})

# Demo function
def demo_sample_dataset():
    """Demo function to initialize and showcase sample dataset"""
    manager = SampleDatasetManager()
    dataset_info = manager.initialize_complete_dataset()
    
    logger.info("=== Sample Dataset Demo ===")
    logger.info(f"Dataset location: {manager.dataset_root}")
    logger.info(f"Total real samples: {dataset_info['statistics']['total_real_samples']}")
    logger.info(f"Total AI samples: {dataset_info['statistics']['total_ai_samples']}")
    logger.info(f"Test cases available: {dataset_info['statistics']['total_test_cases']}")
    
    # Show test case example
    test_case = manager.get_sample_for_testing('TC001')
    if test_case:
        logger.info(f"\nExample test case: {test_case['name']}")
        logger.info(f"Description: {test_case['description']}")
        logger.info(f"Expected result: {test_case['expected_result']['prediction']}")
    
    return dataset_info

if __name__ == "__main__":
    demo_sample_dataset()

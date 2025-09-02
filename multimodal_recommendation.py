"""
Multimodal Recommendation System for PlayStation Store
Author: Yang Liu
Description: Advanced recommendation using video, text, and image features
Experience: Similar system deployed at SHAREit for short video recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib
from collections import defaultdict


@dataclass
class VideoFeatures:
    """Video content features for games/trailers"""
    video_id: str
    duration_seconds: float
    resolution: str  # 4K, 1080p, 720p
    fps: int
    bitrate: float
    
    # Content features
    scene_changes: int
    avg_brightness: float
    color_histogram: np.ndarray
    motion_intensity: float
    
    # Audio features
    has_speech: bool
    music_genre: str
    avg_loudness: float
    
    # Engagement metrics
    avg_watch_time: float
    completion_rate: float
    replay_rate: float
    skip_rate: float
    
    # Temporal features
    peak_engagement_points: List[float]  # Timestamps where users are most engaged
    drop_off_points: List[float]  # Where users typically stop watching


@dataclass
class TextFeatures:
    """Text features from game descriptions, reviews, etc."""
    text_id: str
    title: str
    description: str
    
    # NLP features
    embedding: np.ndarray  # BERT/Sentence-BERT embedding
    keywords: List[str]
    sentiment_score: float
    readability_score: float
    
    # Game-specific
    genres: List[str]
    tags: List[str]
    esrb_rating: str
    languages: List[str]


@dataclass
class ImageFeatures:
    """Image features from game covers, screenshots"""
    image_id: str
    
    # Visual features
    embedding: np.ndarray  # ResNet/CLIP embedding
    dominant_colors: List[Tuple[int, int, int]]
    brightness: float
    contrast: float
    sharpness: float
    
    # Content detection
    has_text: bool
    has_faces: bool
    object_types: List[str]  # Detected objects
    scene_type: str  # Indoor, outdoor, abstract, etc.
    
    # Aesthetic scores
    composition_score: float
    quality_score: float


class MultimodalFeatureExtractor:
    """Extract and process features from multiple modalities"""
    
    def __init__(self):
        self.video_features_cache = {}
        self.text_features_cache = {}
        self.image_features_cache = {}
        
    def extract_video_features(self, video_path: str, video_metadata: Dict) -> VideoFeatures:
        """
        Extract comprehensive video features
        In production, this would use CV2, FFmpeg, etc.
        """
        # Simulate feature extraction
        video_id = video_metadata.get('id', hashlib.md5(video_path.encode()).hexdigest())
        
        # Duration-based features
        duration = video_metadata.get('duration', 120)
        
        # Engagement patterns based on duration
        if duration < 30:  # Short trailer
            completion_rate = 0.85
            skip_rate = 0.10
        elif duration < 120:  # Regular trailer
            completion_rate = 0.65
            skip_rate = 0.25
        else:  # Long gameplay video
            completion_rate = 0.35
            skip_rate = 0.40
        
        # Generate realistic engagement points
        peak_points = self._generate_engagement_peaks(duration)
        drop_points = self._generate_drop_off_points(duration)
        
        features = VideoFeatures(
            video_id=video_id,
            duration_seconds=duration,
            resolution=video_metadata.get('resolution', '1080p'),
            fps=video_metadata.get('fps', 30),
            bitrate=video_metadata.get('bitrate', 5000),
            
            # Content features (simulated)
            scene_changes=int(duration / 3),  # Avg scene every 3 seconds
            avg_brightness=np.random.uniform(0.3, 0.7),
            color_histogram=np.random.rand(256),
            motion_intensity=np.random.uniform(0.2, 0.8),
            
            # Audio
            has_speech=video_metadata.get('has_speech', True),
            music_genre=video_metadata.get('music_genre', 'orchestral'),
            avg_loudness=np.random.uniform(-20, -5),
            
            # Engagement
            avg_watch_time=duration * completion_rate,
            completion_rate=completion_rate,
            replay_rate=np.random.uniform(0.05, 0.15),
            skip_rate=skip_rate,
            
            # Temporal
            peak_engagement_points=peak_points,
            drop_off_points=drop_points
        )
        
        self.video_features_cache[video_id] = features
        return features
    
    def _generate_engagement_peaks(self, duration: float) -> List[float]:
        """Generate realistic engagement peak timestamps"""
        peaks = []
        
        # Opening hook (first 5-10 seconds)
        if duration > 10:
            peaks.append(np.random.uniform(5, 10))
        
        # Action sequences (distributed throughout)
        n_peaks = int(duration / 30)  # One peak every 30 seconds
        for i in range(n_peaks):
            peak_time = (i + 1) * 30 + np.random.uniform(-5, 5)
            if peak_time < duration - 5:
                peaks.append(peak_time)
        
        # Climax (around 70-80% of video)
        if duration > 60:
            peaks.append(duration * np.random.uniform(0.7, 0.8))
        
        return sorted(peaks)
    
    def _generate_drop_off_points(self, duration: float) -> List[float]:
        """Generate typical drop-off points"""
        drops = []
        
        # Early drop (attention span)
        if duration > 15:
            drops.append(np.random.uniform(10, 15))
        
        # Mid-video drop (boring parts)
        if duration > 60:
            drops.append(duration * np.random.uniform(0.4, 0.5))
        
        # Late drop (got the idea, skip to end)
        if duration > 30:
            drops.append(duration * np.random.uniform(0.85, 0.95))
        
        return sorted(drops)
    
    def extract_text_features(self, text_data: Dict) -> TextFeatures:
        """Extract text features using NLP"""
        
        # Simulate BERT embedding
        embedding_dim = 768
        text_embedding = np.random.randn(embedding_dim)
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        # Extract keywords (simulated)
        description = text_data.get('description', '')
        keywords = self._extract_keywords(description)
        
        features = TextFeatures(
            text_id=text_data.get('id', 'text_001'),
            title=text_data.get('title', ''),
            description=description,
            embedding=text_embedding,
            keywords=keywords,
            sentiment_score=np.random.uniform(-1, 1),
            readability_score=np.random.uniform(0.3, 0.9),
            genres=text_data.get('genres', ['Action', 'Adventure']),
            tags=text_data.get('tags', ['multiplayer', 'open-world']),
            esrb_rating=text_data.get('rating', 'T'),
            languages=text_data.get('languages', ['English', 'Spanish'])
        )
        
        self.text_features_cache[features.text_id] = features
        return features
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        # In production, use TF-IDF, RAKE, or KeyBERT
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 4 and w not in common_words]
        return list(set(keywords))[:10]
    
    def extract_image_features(self, image_data: Dict) -> ImageFeatures:
        """Extract visual features from images"""
        
        # Simulate ResNet/CLIP embedding
        embedding_dim = 2048
        image_embedding = np.random.randn(embedding_dim)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        features = ImageFeatures(
            image_id=image_data.get('id', 'img_001'),
            embedding=image_embedding,
            dominant_colors=[
                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                for _ in range(3)
            ],
            brightness=np.random.uniform(0.2, 0.8),
            contrast=np.random.uniform(0.3, 0.9),
            sharpness=np.random.uniform(0.4, 1.0),
            has_text=np.random.choice([True, False], p=[0.7, 0.3]),
            has_faces=np.random.choice([True, False], p=[0.4, 0.6]),
            object_types=np.random.choice(
                ['character', 'weapon', 'vehicle', 'landscape', 'building'],
                size=np.random.randint(1, 4),
                replace=False
            ).tolist(),
            scene_type=np.random.choice(['outdoor', 'indoor', 'abstract', 'ui']),
            composition_score=np.random.uniform(0.5, 1.0),
            quality_score=np.random.uniform(0.6, 1.0)
        )
        
        self.image_features_cache[features.image_id] = features
        return features


class MultimodalFeatureAnalyzer:
    """Analyze and derive insights from multimodal features"""
    
    def __init__(self):
        self.feature_importance = {
            'video_duration': 0.15,
            'video_completion_rate': 0.25,
            'text_sentiment': 0.10,
            'image_quality': 0.08,
            'video_engagement_peaks': 0.20,
            'text_embedding_similarity': 0.12,
            'visual_embedding_similarity': 0.10
        }
    
    def analyze_video_engagement(self, features: VideoFeatures) -> Dict:
        """Analyze video engagement patterns"""
        
        analysis = {
            'engagement_score': 0.0,
            'optimal_preview_length': 0.0,
            'key_moments': [],
            'improvement_suggestions': []
        }
        
        # Calculate engagement score
        engagement_components = {
            'completion': features.completion_rate * 0.4,
            'replay': features.replay_rate * 10 * 0.2,  # Scaled
            'skip_inverse': (1 - features.skip_rate) * 0.2,
            'watch_ratio': (features.avg_watch_time / features.duration_seconds) * 0.2
        }
        
        analysis['engagement_score'] = sum(engagement_components.values())
        
        # Determine optimal preview length
        if features.peak_engagement_points:
            # Include first 2 peaks for preview
            preview_peaks = features.peak_engagement_points[:2]
            if preview_peaks:
                analysis['optimal_preview_length'] = min(
                    max(preview_peaks[-1] + 5, 15),  # At least 15 seconds
                    min(features.duration_seconds * 0.3, 60)  # Max 30% or 60 seconds
                )
        
        # Identify key moments
        analysis['key_moments'] = [
            {'timestamp': peak, 'type': 'engagement_peak'}
            for peak in features.peak_engagement_points
        ]
        
        # Suggestions based on metrics
        if features.skip_rate > 0.3:
            analysis['improvement_suggestions'].append(
                "High skip rate detected. Consider shorter, more engaging opening."
            )
        
        if features.completion_rate < 0.5:
            analysis['improvement_suggestions'].append(
                "Low completion rate. Video might be too long or loses engagement mid-way."
            )
        
        if len(features.drop_off_points) > len(features.peak_engagement_points):
            analysis['improvement_suggestions'].append(
                "More drop-offs than peaks. Add more engaging moments throughout."
            )
        
        return analysis
    
    def analyze_content_alignment(self, 
                                 video: VideoFeatures,
                                 text: TextFeatures,
                                 image: ImageFeatures) -> Dict:
        """Analyze alignment between different modalities"""
        
        alignment = {
            'overall_consistency': 0.0,
            'video_text_alignment': 0.0,
            'video_image_alignment': 0.0,
            'text_image_alignment': 0.0,
            'recommendations': []
        }
        
        # Video-Text alignment
        # Check if video duration matches description complexity
        if video.duration_seconds > 120 and text.readability_score < 0.5:
            alignment['video_text_alignment'] = 0.3
            alignment['recommendations'].append(
                "Long video with complex description might confuse users"
            )
        else:
            alignment['video_text_alignment'] = 0.8
        
        # Video-Image alignment
        # Check if video brightness matches image brightness
        brightness_diff = abs(video.avg_brightness - image.brightness)
        alignment['video_image_alignment'] = 1.0 - brightness_diff
        
        # Text-Image alignment
        # In production, would use CLIP to check semantic alignment
        alignment['text_image_alignment'] = np.random.uniform(0.6, 0.9)
        
        # Overall consistency
        alignment['overall_consistency'] = np.mean([
            alignment['video_text_alignment'],
            alignment['video_image_alignment'],
            alignment['text_image_alignment']
        ])
        
        return alignment
    
    def generate_feature_report(self,
                               video: VideoFeatures,
                               text: TextFeatures,
                               image: ImageFeatures) -> str:
        """Generate comprehensive feature analysis report"""
        
        report = []
        report.append("="*60)
        report.append("MULTIMODAL FEATURE ANALYSIS REPORT")
        report.append("="*60)
        
        # Video Analysis
        report.append("\n📹 VIDEO FEATURES")
        report.append("-"*30)
        report.append(f"Duration: {video.duration_seconds:.1f} seconds")
        report.append(f"Resolution: {video.resolution}")
        report.append(f"Completion Rate: {video.completion_rate:.1%}")
        report.append(f"Skip Rate: {video.skip_rate:.1%}")
        report.append(f"Engagement Peaks: {len(video.peak_engagement_points)}")
        
        video_analysis = self.analyze_video_engagement(video)
        report.append(f"Engagement Score: {video_analysis['engagement_score']:.2f}/1.0")
        report.append(f"Optimal Preview: {video_analysis['optimal_preview_length']:.1f}s")
        
        # Text Analysis
        report.append("\n📝 TEXT FEATURES")
        report.append("-"*30)
        report.append(f"Title: {text.title[:50]}...")
        report.append(f"Genres: {', '.join(text.genres)}")
        report.append(f"Tags: {', '.join(text.tags[:5])}")
        report.append(f"Sentiment: {text.sentiment_score:.2f}")
        report.append(f"Top Keywords: {', '.join(text.keywords[:5])}")
        
        # Image Analysis
        report.append("\n🖼️ IMAGE FEATURES")
        report.append("-"*30)
        report.append(f"Quality Score: {image.quality_score:.2f}/1.0")
        report.append(f"Scene Type: {image.scene_type}")
        report.append(f"Detected Objects: {', '.join(image.object_types)}")
        
        # Cross-modal Analysis
        report.append("\n🔄 CROSS-MODAL ALIGNMENT")
        report.append("-"*30)
        alignment = self.analyze_content_alignment(video, text, image)
        report.append(f"Overall Consistency: {alignment['overall_consistency']:.1%}")
        report.append(f"Video-Text Alignment: {alignment['video_text_alignment']:.1%}")
        report.append(f"Video-Image Alignment: {alignment['video_image_alignment']:.1%}")
        
        # Recommendations
        if video_analysis['improvement_suggestions'] or alignment['recommendations']:
            report.append("\n💡 RECOMMENDATIONS")
            report.append("-"*30)
            for suggestion in video_analysis['improvement_suggestions']:
                report.append(f"• {suggestion}")
            for rec in alignment['recommendations']:
                report.append(f"• {rec}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


class MultimodalRecommendationModel:
    """
    Fusion model combining all modalities for recommendation
    """
    
    def __init__(self):
        self.feature_extractor = MultimodalFeatureExtractor()
        self.analyzer = MultimodalFeatureAnalyzer()
        self.fusion_weights = {
            'video': 0.4,
            'text': 0.35,
            'image': 0.25
        }
    
    def create_multimodal_embedding(self,
                                   video: VideoFeatures,
                                   text: TextFeatures,
                                   image: ImageFeatures) -> np.ndarray:
        """Create unified embedding from all modalities"""
        
        # Video embedding (behavioral features)
        video_embedding = np.concatenate([
            [video.duration_seconds / 300],  # Normalized duration
            [video.completion_rate],
            [video.replay_rate * 10],  # Scaled
            [1 - video.skip_rate],
            [video.motion_intensity],
            [len(video.peak_engagement_points) / 10]  # Normalized peak count
        ])
        
        # Use pre-computed embeddings for text and image
        text_embedding = text.embedding[:128]  # Take first 128 dims
        image_embedding = image.embedding[:128]
        
        # Weighted fusion
        video_proj = video_embedding * self.fusion_weights['video']
        text_proj = text_embedding * self.fusion_weights['text']  
        image_proj = image_embedding * self.fusion_weights['image']
        
        # Concatenate and normalize
        multimodal = np.concatenate([video_proj, text_proj, image_proj])
        multimodal = multimodal / np.linalg.norm(multimodal)
        
        return multimodal
    
    def predict_engagement(self,
                          user_profile: Dict,
                          content_features: Dict) -> float:
        """
        Predict user engagement with content based on multimodal features
        """
        # Extract features
        video = content_features['video']
        text = content_features['text']
        image = content_features['image']
        
        # Create multimodal embedding
        content_embedding = self.create_multimodal_embedding(video, text, image)
        
        # User preference embedding (simulated)
        user_embedding = np.random.randn(len(content_embedding))
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        
        # Calculate similarity
        similarity = np.dot(content_embedding, user_embedding)
        
        # Adjust based on user history
        if user_profile.get('prefers_short_videos') and video.duration_seconds < 60:
            similarity *= 1.2
        
        if user_profile.get('genres'):
            genre_overlap = len(set(text.genres) & set(user_profile['genres']))
            similarity *= (1 + genre_overlap * 0.1)
        
        # Engagement prediction (0-1 scale)
        engagement = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
        
        return min(max(engagement, 0), 1)


def demo_multimodal_recommendation():
    """Demonstrate multimodal recommendation system"""
    
    print("🎮 PlayStation Multimodal Recommendation System Demo")
    print("="*60)
    
    # Initialize system
    extractor = MultimodalFeatureExtractor()
    analyzer = MultimodalFeatureAnalyzer()
    model = MultimodalRecommendationModel()
    
    # Simulate game content
    game_data = {
        'id': 'game_001',
        'title': 'Horizon Forbidden West',
        'video_metadata': {
            'duration': 180,  # 3 minute trailer
            'resolution': '4K',
            'fps': 60,
            'has_speech': True,
            'music_genre': 'epic'
        },
        'text_data': {
            'title': 'Horizon Forbidden West',
            'description': 'Explore distant lands, fight bigger machines, and encounter new tribes',
            'genres': ['Action', 'RPG', 'Adventure'],
            'tags': ['open-world', 'single-player', 'story-rich', 'sci-fi'],
            'rating': 'T'
        },
        'image_data': {
            'id': 'cover_001'
        }
    }
    
    # Extract features
    print("\n📊 Extracting Multimodal Features...")
    video_features = extractor.extract_video_features(
        'trailer.mp4',
        game_data['video_metadata']
    )
    text_features = extractor.extract_text_features(game_data['text_data'])
    image_features = extractor.extract_image_features(game_data['image_data'])
    
    # Generate analysis report
    report = analyzer.generate_feature_report(
        video_features,
        text_features,
        image_features
    )
    print(report)
    
    # Test recommendation
    print("\n🎯 Testing Recommendation Prediction...")
    
    # Simulate different user profiles
    user_profiles = [
        {'id': 'user_1', 'prefers_short_videos': True, 'genres': ['Action', 'RPG']},
        {'id': 'user_2', 'prefers_short_videos': False, 'genres': ['Puzzle', 'Strategy']},
        {'id': 'user_3', 'prefers_short_videos': False, 'genres': ['Adventure', 'RPG']}
    ]
    
    content = {
        'video': video_features,
        'text': text_features,
        'image': image_features
    }
    
    print("\nEngagement Predictions:")
    print("-"*30)
    for user in user_profiles:
        score = model.predict_engagement(user, content)
        print(f"{user['id']}: {score:.1%} (Genres: {', '.join(user['genres'])})")


if __name__ == "__main__":
    demo_multimodal_recommendation()
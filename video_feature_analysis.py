"""
Advanced Video Feature Analysis for Game Trailers
Author: Yang Liu
Description: Deep analysis of video content for recommendation
Experience: Built similar system at SHAREit for TikTok-style videos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json


@dataclass
class DetailedVideoFeatures:
    """Comprehensive video features for deep analysis"""
    
    # Basic metadata
    video_id: str
    title: str
    game_id: str
    upload_date: datetime
    
    # Duration analysis
    total_duration: float  # seconds
    intro_duration: float  # Opening sequence
    outro_duration: float  # Ending sequence
    content_duration: float  # Main content
    
    # Temporal segments
    segments: List[Dict] = field(default_factory=list)  # List of time segments with features
    
    # Pacing analysis
    avg_shot_duration: float = 0.0  # Average time between cuts
    pacing_variance: float = 0.0  # Variance in pacing
    action_density: float = 0.0  # Actions per minute
    
    # User engagement patterns
    watch_time_distribution: np.ndarray = None  # Watch time histogram
    replay_heatmap: np.ndarray = None  # Which parts get replayed
    skip_heatmap: np.ndarray = None  # Which parts get skipped
    interaction_points: List[float] = field(default_factory=list)  # Like, share, comment timestamps
    
    # Quality metrics
    resolution_changes: List[Tuple[float, str]] = field(default_factory=list)  # Adaptive streaming
    buffering_events: List[float] = field(default_factory=list)  # Buffering timestamps
    quality_score: float = 0.0
    
    # Content classification
    content_type: str = ""  # trailer, gameplay, cutscene, review
    genre_signals: Dict[str, float] = field(default_factory=dict)  # Genre confidence scores
    maturity_indicators: Dict[str, bool] = field(default_factory=dict)  # Violence, language, etc.
    
    # Emotional arc
    excitement_curve: np.ndarray = None  # Excitement level over time
    emotion_transitions: List[Tuple[float, str, str]] = field(default_factory=list)  # Time, from_emotion, to_emotion
    
    # Platform-specific
    platform_optimized: List[str] = field(default_factory=list)  # PS5, PS4, Mobile
    hdr_support: bool = False
    haptic_feedback_points: List[float] = field(default_factory=list)  # PS5 specific


class VideoEngagementAnalyzer:
    """Analyze user engagement patterns with video content"""
    
    def __init__(self):
        self.engagement_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        # Video duration categories
        self.duration_categories = {
            'micro': (0, 15),      # TikTok style
            'short': (15, 60),     # Short trailer
            'medium': (60, 180),   # Standard trailer
            'long': (180, 600),    # Gameplay demo
            'extended': (600, float('inf'))  # Full gameplay
        }
    
    def analyze_watch_patterns(self, video: DetailedVideoFeatures, 
                              user_sessions: List[Dict]) -> Dict:
        """
        Analyze how users watch the video
        
        user_sessions: List of {user_id, watch_duration, actions, timestamp}
        """
        analysis = {
            'avg_watch_duration': 0,
            'median_watch_duration': 0,
            'completion_rate': 0,
            'replay_rate': 0,
            'engagement_score': 0,
            'drop_off_curve': [],
            'key_insights': []
        }
        
        if not user_sessions:
            return analysis
        
        # Calculate basic metrics
        watch_durations = [s['watch_duration'] for s in user_sessions]
        analysis['avg_watch_duration'] = np.mean(watch_durations)
        analysis['median_watch_duration'] = np.median(watch_durations)
        
        # Completion rate
        completions = sum(1 for d in watch_durations if d >= video.total_duration * 0.95)
        analysis['completion_rate'] = completions / len(user_sessions)
        
        # Replay rate (users who watched multiple times)
        user_watch_counts = {}
        for session in user_sessions:
            user_id = session['user_id']
            user_watch_counts[user_id] = user_watch_counts.get(user_id, 0) + 1
        
        replays = sum(1 for count in user_watch_counts.values() if count > 1)
        analysis['replay_rate'] = replays / len(user_watch_counts)
        
        # Generate drop-off curve
        analysis['drop_off_curve'] = self._calculate_dropoff_curve(
            watch_durations, video.total_duration
        )
        
        # Calculate engagement score
        analysis['engagement_score'] = self._calculate_engagement_score(
            analysis['completion_rate'],
            analysis['replay_rate'],
            analysis['avg_watch_duration'] / video.total_duration
        )
        
        # Generate insights
        analysis['key_insights'] = self._generate_insights(analysis, video)
        
        return analysis
    
    def _calculate_dropoff_curve(self, watch_durations: List[float], 
                                total_duration: float) -> List[Tuple[float, float]]:
        """Calculate viewer drop-off at each time point"""
        
        # Create time buckets (every 5% of video)
        n_buckets = 20
        bucket_size = total_duration / n_buckets
        curve = []
        
        for i in range(n_buckets):
            time_point = i * bucket_size
            viewers_remaining = sum(1 for d in watch_durations if d >= time_point)
            retention_rate = viewers_remaining / len(watch_durations)
            curve.append((time_point, retention_rate))
        
        return curve
    
    def _calculate_engagement_score(self, completion_rate: float,
                                   replay_rate: float,
                                   watch_ratio: float) -> float:
        """Calculate overall engagement score (0-1)"""
        
        weights = {
            'completion': 0.4,
            'replay': 0.3,
            'watch_ratio': 0.3
        }
        
        score = (
            completion_rate * weights['completion'] +
            min(replay_rate * 2, 1.0) * weights['replay'] +  # Scale replay rate
            watch_ratio * weights['watch_ratio']
        )
        
        return min(score, 1.0)
    
    def _generate_insights(self, analysis: Dict, video: DetailedVideoFeatures) -> List[str]:
        """Generate actionable insights from analysis"""
        
        insights = []
        
        # Duration category insight
        duration_category = self._get_duration_category(video.total_duration)
        avg_watch_pct = analysis['avg_watch_duration'] / video.total_duration
        
        if duration_category in ['micro', 'short'] and avg_watch_pct < 0.8:
            insights.append(f"Short video ({duration_category}) has low watch rate ({avg_watch_pct:.1%}). Content may not be engaging enough.")
        elif duration_category in ['long', 'extended'] and analysis['completion_rate'] > 0.3:
            insights.append(f"Long video has surprisingly high completion ({analysis['completion_rate']:.1%}). Content is highly engaging.")
        
        # Drop-off analysis
        if analysis['drop_off_curve']:
            major_drops = self._find_major_dropoffs(analysis['drop_off_curve'])
            for time, drop_pct in major_drops:
                insights.append(f"Major drop-off at {time:.1f}s ({drop_pct:.1%} viewers lost)")
        
        # Replay behavior
        if analysis['replay_rate'] > 0.2:
            insights.append(f"High replay rate ({analysis['replay_rate']:.1%}) indicates memorable content")
        
        # Engagement level
        if analysis['engagement_score'] > self.engagement_thresholds['high']:
            insights.append("Exceptional engagement - consider featuring this content")
        elif analysis['engagement_score'] < self.engagement_thresholds['low']:
            insights.append("Low engagement - review content strategy")
        
        return insights
    
    def _get_duration_category(self, duration: float) -> str:
        """Categorize video by duration"""
        for category, (min_dur, max_dur) in self.duration_categories.items():
            if min_dur <= duration < max_dur:
                return category
        return 'extended'
    
    def _find_major_dropoffs(self, curve: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Find significant drop-off points"""
        major_drops = []
        
        for i in range(1, len(curve)):
            prev_retention = curve[i-1][1]
            curr_retention = curve[i][1]
            drop = prev_retention - curr_retention
            
            # Major drop is >10% absolute or >20% relative
            if drop > 0.1 or (prev_retention > 0 and drop / prev_retention > 0.2):
                major_drops.append((curve[i][0], drop))
        
        return major_drops


class VideoContentClassifier:
    """Classify and tag video content"""
    
    def __init__(self):
        self.content_patterns = {
            'trailer': {
                'duration_range': (30, 180),
                'features': ['quick_cuts', 'music_driven', 'title_cards'],
                'avg_shot_duration': (0.5, 3.0)
            },
            'gameplay': {
                'duration_range': (60, 1800),
                'features': ['hud_elements', 'continuous_action', 'player_control'],
                'avg_shot_duration': (5.0, 30.0)
            },
            'cutscene': {
                'duration_range': (30, 600),
                'features': ['cinematic', 'dialogue', 'story_driven'],
                'avg_shot_duration': (3.0, 10.0)
            },
            'review': {
                'duration_range': (300, 1800),
                'features': ['commentary', 'mixed_footage', 'analysis'],
                'avg_shot_duration': (5.0, 15.0)
            }
        }
        
        self.genre_keywords = {
            'action': ['combat', 'explosion', 'fight', 'battle', 'weapon'],
            'rpg': ['character', 'level', 'quest', 'inventory', 'skill'],
            'sports': ['team', 'score', 'match', 'tournament', 'player'],
            'racing': ['car', 'track', 'speed', 'lap', 'vehicle'],
            'puzzle': ['solve', 'logic', 'challenge', 'brain', 'think'],
            'horror': ['dark', 'scary', 'monster', 'survival', 'fear']
        }
    
    def classify_content(self, video: DetailedVideoFeatures) -> Dict:
        """Classify video content type and attributes"""
        
        classification = {
            'content_type': '',
            'confidence': 0.0,
            'detected_genres': {},
            'content_flags': [],
            'audience_fit': ''
        }
        
        # Determine content type
        type_scores = {}
        for content_type, patterns in self.content_patterns.items():
            score = self._calculate_type_score(video, patterns)
            type_scores[content_type] = score
        
        # Get best match
        best_type = max(type_scores, key=type_scores.get)
        classification['content_type'] = best_type
        classification['confidence'] = type_scores[best_type]
        
        # Detect genres (would use actual video analysis in production)
        classification['detected_genres'] = self._detect_genres(video)
        
        # Content flags
        classification['content_flags'] = self._detect_content_flags(video)
        
        # Audience fit
        classification['audience_fit'] = self._determine_audience_fit(
            classification['content_flags']
        )
        
        return classification
    
    def _calculate_type_score(self, video: DetailedVideoFeatures, 
                             patterns: Dict) -> float:
        """Calculate how well video matches content type patterns"""
        
        score = 0.0
        
        # Check duration
        dur_min, dur_max = patterns['duration_range']
        if dur_min <= video.total_duration <= dur_max:
            score += 0.3
        
        # Check shot duration
        if 'avg_shot_duration' in patterns:
            shot_min, shot_max = patterns['avg_shot_duration']
            if shot_min <= video.avg_shot_duration <= shot_max:
                score += 0.3
        
        # Check features (simulated)
        feature_matches = np.random.randint(0, len(patterns['features']) + 1)
        score += (feature_matches / len(patterns['features'])) * 0.4
        
        return min(score, 1.0)
    
    def _detect_genres(self, video: DetailedVideoFeatures) -> Dict[str, float]:
        """Detect game genres from video content"""
        
        # In production, would use computer vision and audio analysis
        # Here we simulate with random scores
        genres = {}
        
        for genre in self.genre_keywords.keys():
            # Simulate detection confidence
            confidence = np.random.uniform(0, 1)
            if confidence > 0.3:  # Threshold
                genres[genre] = confidence
        
        # Normalize to top 3
        if len(genres) > 3:
            genres = dict(sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3])
        
        return genres
    
    def _detect_content_flags(self, video: DetailedVideoFeatures) -> List[str]:
        """Detect content that might need flagging"""
        
        flags = []
        
        # Simulate content detection
        if np.random.random() > 0.7:
            flags.append('violence_mild')
        if np.random.random() > 0.8:
            flags.append('language_mild')
        if video.total_duration > 600:
            flags.append('long_form')
        if video.avg_shot_duration < 1.0:
            flags.append('rapid_cuts')
        
        return flags
    
    def _determine_audience_fit(self, content_flags: List[str]) -> str:
        """Determine appropriate audience based on content"""
        
        if any('violence' in flag or 'language' in flag for flag in content_flags):
            return 'teen_and_up'
        elif 'rapid_cuts' in content_flags:
            return 'experienced_gamers'
        else:
            return 'all_audiences'


class VideoRecommendationOptimizer:
    """Optimize video recommendations based on analysis"""
    
    def __init__(self):
        self.engagement_analyzer = VideoEngagementAnalyzer()
        self.content_classifier = VideoContentClassifier()
    
    def optimize_recommendations(self, 
                                video_catalog: List[DetailedVideoFeatures],
                                user_profile: Dict) -> List[Tuple[str, float]]:
        """
        Generate optimized video recommendations
        
        Returns: List of (video_id, score) tuples
        """
        recommendations = []
        
        for video in video_catalog:
            score = self._calculate_recommendation_score(video, user_profile)
            recommendations.append((video.video_id, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def _calculate_recommendation_score(self, 
                                      video: DetailedVideoFeatures,
                                      user_profile: Dict) -> float:
        """Calculate recommendation score for a video"""
        
        score = 0.0
        
        # Duration preference
        user_pref_duration = user_profile.get('preferred_duration', 'medium')
        video_duration_cat = self._get_duration_category(video.total_duration)
        
        if user_pref_duration == video_duration_cat:
            score += 0.3
        
        # Genre match
        user_genres = set(user_profile.get('genres', []))
        video_genres = set(video.genre_signals.keys())
        
        if user_genres and video_genres:
            genre_overlap = len(user_genres & video_genres) / len(user_genres)
            score += genre_overlap * 0.4
        
        # Quality preference
        if user_profile.get('quality_conscious', False):
            score += video.quality_score * 0.2
        
        # Platform match
        user_platform = user_profile.get('platform', 'PS5')
        if user_platform in video.platform_optimized:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_duration_category(self, duration: float) -> str:
        """Get duration category for video"""
        if duration < 60:
            return 'short'
        elif duration < 180:
            return 'medium'
        elif duration < 600:
            return 'long'
        else:
            return 'extended'
    
    def generate_optimization_report(self, video: DetailedVideoFeatures,
                                    engagement_data: Dict) -> str:
        """Generate optimization recommendations for content creators"""
        
        report = []
        report.append(f"VIDEO OPTIMIZATION REPORT: {video.title}")
        report.append("="*50)
        
        # Engagement analysis
        report.append("\nENGAGEMENT METRICS:")
        report.append(f"- Engagement Score: {engagement_data.get('engagement_score', 0):.2f}/1.0")
        report.append(f"- Completion Rate: {engagement_data.get('completion_rate', 0):.1%}")
        report.append(f"- Replay Rate: {engagement_data.get('replay_rate', 0):.1%}")
        
        # Content classification
        classification = self.content_classifier.classify_content(video)
        report.append(f"\nCONTENT TYPE: {classification['content_type']} ({classification['confidence']:.1%} confidence)")
        
        # Optimization suggestions
        report.append("\nOPTIMIZATION SUGGESTIONS:")
        
        if engagement_data.get('completion_rate', 0) < 0.5:
            if video.total_duration > 180:
                report.append("- Consider creating a shorter version (< 3 minutes)")
            report.append("- Add engagement hooks in the first 10 seconds")
        
        if engagement_data.get('replay_rate', 0) < 0.1:
            report.append("- Add memorable moments or surprises")
            report.append("- Consider adding easter eggs for fans")
        
        insights = engagement_data.get('key_insights', [])
        if insights:
            report.append("\nKEY INSIGHTS:")
            for insight in insights:
                report.append(f"- {insight}")
        
        return "\n".join(report)


def demo_video_analysis():
    """Demonstrate video feature analysis"""
    
    print("🎬 Advanced Video Feature Analysis Demo")
    print("="*60)
    
    # Create sample video
    video = DetailedVideoFeatures(
        video_id="vid_001",
        title="God of War Ragnarok - Launch Trailer",
        game_id="game_gowr",
        upload_date=datetime.now() - timedelta(days=30),
        total_duration=165.0,  # 2:45
        intro_duration=5.0,
        outro_duration=10.0,
        content_duration=150.0,
        avg_shot_duration=2.3,
        pacing_variance=0.8,
        action_density=12.5,
        quality_score=0.95,
        content_type="trailer",
        platform_optimized=["PS5", "PS4"],
        hdr_support=True
    )
    
    # Add genre signals
    video.genre_signals = {
        'action': 0.9,
        'adventure': 0.8,
        'rpg': 0.6
    }
    
    # Simulate user sessions
    user_sessions = []
    for i in range(1000):
        # Simulate realistic watch patterns
        if np.random.random() < 0.3:  # 30% complete
            watch_duration = video.total_duration
        elif np.random.random() < 0.5:  # 35% watch most
            watch_duration = video.total_duration * np.random.uniform(0.7, 0.95)
        else:  # 35% drop off early
            watch_duration = video.total_duration * np.random.uniform(0.1, 0.7)
        
        user_sessions.append({
            'user_id': f'user_{i}',
            'watch_duration': watch_duration,
            'actions': [],
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 720))
        })
    
    # Analyze engagement
    analyzer = VideoEngagementAnalyzer()
    engagement = analyzer.analyze_watch_patterns(video, user_sessions)
    
    print("\n📊 Engagement Analysis:")
    print(f"Average Watch Duration: {engagement['avg_watch_duration']:.1f}s")
    print(f"Completion Rate: {engagement['completion_rate']:.1%}")
    print(f"Replay Rate: {engagement['replay_rate']:.1%}")
    print(f"Engagement Score: {engagement['engagement_score']:.2f}/1.0")
    
    print("\n💡 Key Insights:")
    for insight in engagement['key_insights']:
        print(f"- {insight}")
    
    # Content classification
    classifier = VideoContentClassifier()
    classification = classifier.classify_content(video)
    
    print(f"\n🏷️ Content Classification:")
    print(f"Type: {classification['content_type']} ({classification['confidence']:.1%} confidence)")
    print(f"Detected Genres: {', '.join(f'{g}({s:.1%})' for g, s in classification['detected_genres'].items())}")
    print(f"Audience: {classification['audience_fit']}")
    
    # Generate optimization report
    optimizer = VideoRecommendationOptimizer()
    report = optimizer.generate_optimization_report(video, engagement)
    
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    demo_video_analysis()
"""
User Comment and Sentiment Analysis Metrics for Recommendation Systems
Comprehensive analysis of user feedback, comments, and sentiment indicators
Author: Yang Liu
"""

import numpy as np
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class SentimentType(Enum):
    VERY_POSITIVE = "very_positive"    # 5 stars, love it, excellent
    POSITIVE = "positive"              # 4 stars, like it, good
    NEUTRAL = "neutral"                # 3 stars, ok, average
    NEGATIVE = "negative"              # 2 stars, dislike, poor
    VERY_NEGATIVE = "very_negative"    # 1 star, hate it, terrible

class FeedbackType(Enum):
    EXPLICIT = "explicit"              # Direct rating, like/dislike button
    IMPLICIT = "implicit"              # Behavior-based (time spent, purchase)
    COMMENT = "comment"                # Text comment/review
    SOCIAL = "social"                  # Share, recommend to friend

@dataclass
class UserFeedbackEvent:
    """Comprehensive user feedback event"""
    user_id: str
    item_id: str
    recommendation_id: str
    session_id: str
    feedback_type: FeedbackType
    sentiment: SentimentType
    timestamp: float
    
    # Explicit feedback
    rating_score: Optional[float] = None  # 1-5 star rating
    like_dislike: Optional[bool] = None   # True=like, False=dislike, None=no action
    
    # Comment analysis
    comment_text: Optional[str] = None
    comment_length: int = 0
    comment_sentiment_score: float = 0.0  # -1 to 1
    comment_keywords: List[str] = field(default_factory=list)
    
    # Behavioral signals
    dwell_time_seconds: float = 0.0
    interaction_depth: float = 0.0  # How much they engaged (0-1)
    follow_up_actions: List[str] = field(default_factory=list)  # What they did after
    
    # Context
    recommendation_position: int = -1
    recommendation_source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


class UserSentimentAnalyzer:
    """Advanced user sentiment and feedback analysis for recommendations"""
    
    def __init__(self, window_hours: int = 24):
        self.window_seconds = window_hours * 3600
        self.feedback_events: List[UserFeedbackEvent] = []
        
        # Sentiment keywords for basic text analysis
        self.positive_keywords = {
            'love', 'excellent', 'amazing', 'perfect', 'great', 'awesome', 
            'fantastic', 'wonderful', 'brilliant', 'outstanding', 'superb',
            'recommend', 'satisfied', 'happy', 'pleased', 'impressed'
        }
        
        self.negative_keywords = {
            'hate', 'terrible', 'awful', 'horrible', 'disappointing', 'bad',
            'poor', 'waste', 'regret', 'useless', 'frustrated', 'annoyed',
            'disappointed', 'unsatisfied', 'unhappy', 'misleading'
        }
        
        # Sentiment scoring weights
        self.sentiment_weights = {
            SentimentType.VERY_POSITIVE: 1.0,
            SentimentType.POSITIVE: 0.5,
            SentimentType.NEUTRAL: 0.0,
            SentimentType.NEGATIVE: -0.5,
            SentimentType.VERY_NEGATIVE: -1.0
        }
        
    async def record_feedback_event(self, event: UserFeedbackEvent):
        """Record user feedback event with automatic sentiment analysis"""
        
        # Auto-analyze comment if provided
        if event.comment_text and not event.comment_keywords:
            event.comment_keywords = self._extract_keywords(event.comment_text)
            event.comment_sentiment_score = self._analyze_text_sentiment(event.comment_text)
            event.comment_length = len(event.comment_text)
        
        # Infer sentiment from rating if not provided
        if event.sentiment == SentimentType.NEUTRAL and event.rating_score:
            event.sentiment = self._rating_to_sentiment(event.rating_score)
        
        self.feedback_events.append(event)
        await self._cleanup_expired_events()
    
    async def _cleanup_expired_events(self):
        """Remove expired feedback events"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        self.feedback_events = [
            event for event in self.feedback_events 
            if event.timestamp >= cutoff_time
        ]
    
    async def calculate_sentiment_metrics(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """
        Calculate comprehensive sentiment metrics
        
        Key Metrics:
        - Overall sentiment distribution
        - Sentiment by recommendation source
        - Comment sentiment analysis
        - Behavioral sentiment indicators
        - Sentiment trend over time
        """
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_events = [e for e in self.feedback_events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {'error': 'no_recent_feedback'}
        
        # Basic sentiment distribution
        sentiment_counts = Counter(event.sentiment for event in recent_events)
        total_feedback = len(recent_events)
        
        sentiment_distribution = {
            sentiment.value: {
                'count': sentiment_counts.get(sentiment, 0),
                'percentage': sentiment_counts.get(sentiment, 0) / total_feedback * 100
            }
            for sentiment in SentimentType
        }
        
        # Calculate overall sentiment score (-1 to 1)
        overall_sentiment_score = sum(
            sentiment_counts.get(sentiment, 0) * self.sentiment_weights[sentiment]
            for sentiment in SentimentType
        ) / total_feedback
        
        # Explicit feedback analysis (ratings, likes/dislikes)
        explicit_feedback = [e for e in recent_events if e.feedback_type == FeedbackType.EXPLICIT]
        explicit_analysis = await self._analyze_explicit_feedback(explicit_feedback)
        
        # Comment analysis
        comment_feedback = [e for e in recent_events if e.comment_text]
        comment_analysis = await self._analyze_comment_feedback(comment_feedback)
        
        # Behavioral sentiment analysis
        behavioral_analysis = await self._analyze_behavioral_sentiment(recent_events)
        
        # Sentiment by recommendation source
        source_sentiment = await self._analyze_sentiment_by_source(recent_events)
        
        # Sentiment quality assessment
        quality_assessment = self._assess_sentiment_quality(
            overall_sentiment_score, 
            len(explicit_feedback), 
            len(comment_feedback)
        )
        
        return {
            'overall_metrics': {
                'total_feedback_events': total_feedback,
                'overall_sentiment_score': overall_sentiment_score,
                'sentiment_distribution': sentiment_distribution,
                'positive_ratio': (sentiment_counts.get(SentimentType.VERY_POSITIVE, 0) + 
                                 sentiment_counts.get(SentimentType.POSITIVE, 0)) / total_feedback,
                'negative_ratio': (sentiment_counts.get(SentimentType.VERY_NEGATIVE, 0) + 
                                 sentiment_counts.get(SentimentType.NEGATIVE, 0)) / total_feedback
            },
            'explicit_feedback_analysis': explicit_analysis,
            'comment_analysis': comment_analysis,
            'behavioral_sentiment': behavioral_analysis,
            'source_sentiment_breakdown': source_sentiment,
            'quality_assessment': quality_assessment,
            'actionable_insights': self._generate_sentiment_insights(
                overall_sentiment_score, sentiment_distribution, comment_analysis
            )
        }
    
    async def calculate_comment_quality_metrics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze comment quality and engagement metrics
        
        Metrics:
        - Comment volume and frequency
        - Comment length distribution
        - Keyword analysis and topics
        - Sentiment progression in comments
        - Comment helpfulness indicators
        """
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        comment_events = [
            e for e in self.feedback_events 
            if e.timestamp >= cutoff_time and e.comment_text
        ]
        
        if not comment_events:
            return {'error': 'no_comments_in_timerange'}
        
        # Basic comment statistics
        comment_lengths = [e.comment_length for e in comment_events]
        comment_sentiments = [e.comment_sentiment_score for e in comment_events]
        
        # Keyword frequency analysis
        all_keywords = []
        for event in comment_events:
            all_keywords.extend(event.comment_keywords)
        
        keyword_frequency = Counter(all_keywords)
        top_keywords = keyword_frequency.most_common(20)
        
        # Sentiment distribution in comments
        comment_sentiment_categories = {
            'very_positive': len([s for s in comment_sentiments if s > 0.6]),
            'positive': len([s for s in comment_sentiments if 0.2 < s <= 0.6]),
            'neutral': len([s for s in comment_sentiments if -0.2 <= s <= 0.2]),
            'negative': len([s for s in comment_sentiments if -0.6 <= s < -0.2]),
            'very_negative': len([s for s in comment_sentiments if s < -0.6])
        }
        
        # Comment quality indicators
        quality_indicators = {
            'avg_comment_length': np.mean(comment_lengths),
            'median_comment_length': np.median(comment_lengths),
            'detailed_comments_ratio': len([l for l in comment_lengths if l > 50]) / len(comment_lengths),
            'avg_sentiment_score': np.mean(comment_sentiments),
            'sentiment_variance': np.var(comment_sentiments),
            'engagement_score': self._calculate_comment_engagement_score(comment_events)
        }
        
        # Topic analysis (simplified)
        topic_analysis = await self._analyze_comment_topics(comment_events)
        
        # Comment trends over time
        time_trends = await self._analyze_comment_trends(comment_events)
        
        return {
            'comment_volume': {
                'total_comments': len(comment_events),
                'unique_users': len(set(e.user_id for e in comment_events)),
                'avg_comments_per_user': len(comment_events) / len(set(e.user_id for e in comment_events)),
                'comments_per_hour': len(comment_events) / time_range_hours
            },
            'quality_indicators': quality_indicators,
            'sentiment_analysis': {
                'sentiment_categories': comment_sentiment_categories,
                'avg_sentiment': np.mean(comment_sentiments),
                'sentiment_std': np.std(comment_sentiments)
            },
            'keyword_analysis': {
                'top_keywords': top_keywords,
                'unique_keywords': len(set(all_keywords)),
                'keyword_diversity': len(set(all_keywords)) / max(len(all_keywords), 1)
            },
            'topic_analysis': topic_analysis,
            'time_trends': time_trends,
            'quality_assessment': self._assess_comment_quality(quality_indicators),
            'improvement_suggestions': self._generate_comment_improvement_suggestions(quality_indicators, topic_analysis)
        }
    
    async def calculate_recommendation_satisfaction_score(self, time_range_hours: int = 6) -> Dict[str, Any]:
        """
        Calculate overall recommendation satisfaction score
        Combines explicit feedback, behavioral signals, and sentiment analysis
        """
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_events = [e for e in self.feedback_events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {'error': 'insufficient_feedback_data'}
        
        # Group events by item for item-level analysis
        item_feedback = defaultdict(list)
        for event in recent_events:
            item_feedback[event.item_id].append(event)
        
        item_satisfaction_scores = {}
        
        for item_id, events in item_feedback.items():
            # Explicit satisfaction signals
            explicit_score = self._calculate_explicit_satisfaction(events)
            
            # Behavioral satisfaction signals  
            behavioral_score = self._calculate_behavioral_satisfaction(events)
            
            # Comment-based satisfaction
            comment_score = self._calculate_comment_satisfaction(events)
            
            # Combined satisfaction score (0-10 scale)
            combined_score = (
                explicit_score * 0.4 +      # 40% weight on explicit feedback
                behavioral_score * 0.35 +   # 35% weight on behavior
                comment_score * 0.25        # 25% weight on comments
            )
            
            item_satisfaction_scores[item_id] = {
                'overall_satisfaction': combined_score,
                'explicit_score': explicit_score,
                'behavioral_score': behavioral_score,
                'comment_score': comment_score,
                'feedback_count': len(events),
                'user_count': len(set(e.user_id for e in events))
            }
        
        # System-level satisfaction metrics
        all_scores = [scores['overall_satisfaction'] for scores in item_satisfaction_scores.values()]
        system_satisfaction = {
            'avg_satisfaction_score': np.mean(all_scores),
            'median_satisfaction_score': np.median(all_scores),
            'satisfaction_distribution': self._categorize_satisfaction_scores(all_scores),
            'high_satisfaction_items': len([s for s in all_scores if s >= 8.0]),
            'low_satisfaction_items': len([s for s in all_scores if s <= 4.0]),
            'total_items_with_feedback': len(item_satisfaction_scores)
        }
        
        # Satisfaction trends and patterns
        satisfaction_insights = self._analyze_satisfaction_patterns(item_satisfaction_scores, recent_events)
        
        return {
            'system_satisfaction': system_satisfaction,
            'item_satisfaction_scores': dict(list(item_satisfaction_scores.items())[:10]),  # Top 10 for display
            'satisfaction_insights': satisfaction_insights,
            'recommendation_quality_grade': self._grade_recommendation_quality(system_satisfaction['avg_satisfaction_score']),
            'improvement_priorities': self._identify_satisfaction_improvement_priorities(item_satisfaction_scores)
        }
    
    # Helper methods for sentiment analysis
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from comment text"""
        if not text:
            return []
        
        # Simple keyword extraction (in production, use NLP libraries)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter meaningful words (remove stop words, keep sentiment-bearing words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return list(set(keywords))  # Return unique keywords
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (simplified version)"""
        if not text:
            return 0.0
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0
        
        # Normalize to -1 to 1 scale
        sentiment_score = (positive_count - negative_count) / max(total_sentiment_words, 1)
        return max(-1.0, min(1.0, sentiment_score))
    
    def _rating_to_sentiment(self, rating: float) -> SentimentType:
        """Convert numerical rating to sentiment type"""
        if rating >= 4.5:
            return SentimentType.VERY_POSITIVE
        elif rating >= 3.5:
            return SentimentType.POSITIVE
        elif rating >= 2.5:
            return SentimentType.NEUTRAL
        elif rating >= 1.5:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.VERY_NEGATIVE
    
    async def _analyze_explicit_feedback(self, explicit_events: List[UserFeedbackEvent]) -> Dict[str, Any]:
        """Analyze explicit feedback (ratings, likes/dislikes)"""
        if not explicit_events:
            return {}
        
        # Rating analysis
        ratings = [e.rating_score for e in explicit_events if e.rating_score is not None]
        
        # Like/dislike analysis
        likes = len([e for e in explicit_events if e.like_dislike is True])
        dislikes = len([e for e in explicit_events if e.like_dislike is False])
        no_reaction = len([e for e in explicit_events if e.like_dislike is None])
        
        return {
            'rating_analysis': {
                'total_ratings': len(ratings),
                'avg_rating': np.mean(ratings) if ratings else 0,
                'rating_distribution': Counter(ratings) if ratings else {},
                'rating_std': np.std(ratings) if ratings else 0
            },
            'like_dislike_analysis': {
                'total_likes': likes,
                'total_dislikes': dislikes,
                'no_reaction': no_reaction,
                'like_ratio': likes / max(likes + dislikes, 1),
                'engagement_ratio': (likes + dislikes) / len(explicit_events)
            }
        }
    
    async def _analyze_comment_feedback(self, comment_events: List[UserFeedbackEvent]) -> Dict[str, Any]:
        """Analyze comment-based feedback"""
        if not comment_events:
            return {}
        
        sentiment_scores = [e.comment_sentiment_score for e in comment_events]
        comment_lengths = [e.comment_length for e in comment_events]
        
        # Analyze sentiment keywords
        positive_comments = len([s for s in sentiment_scores if s > 0.2])
        negative_comments = len([s for s in sentiment_scores if s < -0.2])
        neutral_comments = len([s for s in sentiment_scores if -0.2 <= s <= 0.2])
        
        return {
            'comment_sentiment': {
                'avg_sentiment_score': np.mean(sentiment_scores),
                'positive_comments': positive_comments,
                'negative_comments': negative_comments,
                'neutral_comments': neutral_comments,
                'sentiment_clarity': 1 - (neutral_comments / len(comment_events))
            },
            'comment_engagement': {
                'avg_comment_length': np.mean(comment_lengths),
                'detailed_comments': len([l for l in comment_lengths if l > 100]),
                'brief_comments': len([l for l in comment_lengths if l <= 20])
            }
        }
    
    async def _analyze_behavioral_sentiment(self, events: List[UserFeedbackEvent]) -> Dict[str, Any]:
        """Analyze behavioral indicators of sentiment"""
        
        behavioral_events = [e for e in events if e.dwell_time_seconds > 0 or e.interaction_depth > 0]
        
        if not behavioral_events:
            return {}
        
        dwell_times = [e.dwell_time_seconds for e in behavioral_events]
        interaction_depths = [e.interaction_depth for e in behavioral_events]
        
        # Behavioral sentiment indicators
        engaged_interactions = len([e for e in behavioral_events if e.dwell_time_seconds > 30])
        deep_interactions = len([e for e in behavioral_events if e.interaction_depth > 0.5])
        
        return {
            'engagement_indicators': {
                'avg_dwell_time': np.mean(dwell_times),
                'avg_interaction_depth': np.mean(interaction_depths),
                'engaged_interaction_ratio': engaged_interactions / len(behavioral_events),
                'deep_interaction_ratio': deep_interactions / len(behavioral_events)
            },
            'behavioral_satisfaction_score': (
                min(np.mean(dwell_times) / 60, 1) * 0.4 +  # Normalized dwell time
                np.mean(interaction_depths) * 0.6          # Interaction depth
            ) * 10  # Scale to 0-10
        }
    
    async def _analyze_sentiment_by_source(self, events: List[UserFeedbackEvent]) -> Dict[str, Dict]:
        """Analyze sentiment breakdown by recommendation source"""
        
        source_events = defaultdict(list)
        for event in events:
            source_events[event.recommendation_source or 'unknown'].append(event)
        
        source_analysis = {}
        for source, source_event_list in source_events.items():
            sentiment_scores = [self.sentiment_weights[e.sentiment] for e in source_event_list]
            
            source_analysis[source] = {
                'event_count': len(source_event_list),
                'avg_sentiment': np.mean(sentiment_scores),
                'positive_ratio': len([s for s in sentiment_scores if s > 0]) / len(sentiment_scores),
                'negative_ratio': len([s for s in sentiment_scores if s < 0]) / len(sentiment_scores)
            }
        
        return source_analysis
    
    def _assess_sentiment_quality(self, overall_score: float, explicit_count: int, comment_count: int) -> Dict[str, str]:
        """Assess overall sentiment quality"""
        
        # Sentiment score assessment
        if overall_score > 0.4:
            score_assessment = 'HIGHLY_POSITIVE'
        elif overall_score > 0.1:
            score_assessment = 'MODERATELY_POSITIVE'
        elif overall_score > -0.1:
            score_assessment = 'NEUTRAL'
        elif overall_score > -0.4:
            score_assessment = 'MODERATELY_NEGATIVE'
        else:
            score_assessment = 'HIGHLY_NEGATIVE'
        
        # Feedback volume assessment
        total_feedback = explicit_count + comment_count
        if total_feedback > 100:
            volume_assessment = 'HIGH_VOLUME'
        elif total_feedback > 20:
            volume_assessment = 'MODERATE_VOLUME'
        else:
            volume_assessment = 'LOW_VOLUME'
        
        return {
            'sentiment_assessment': score_assessment,
            'feedback_volume': volume_assessment,
            'overall_quality': f'{score_assessment}_{volume_assessment}'
        }
    
    def _generate_sentiment_insights(self, overall_score: float, distribution: Dict, comment_analysis: Dict) -> List[str]:
        """Generate actionable insights from sentiment analysis"""
        insights = []
        
        if overall_score < -0.2:
            insights.append('NEGATIVE_SENTIMENT_ALERT: Overall sentiment is negative, investigate recommendation quality')
        
        # Check for polarized opinions
        positive_pct = distribution.get(SentimentType.VERY_POSITIVE.value, {}).get('percentage', 0)
        negative_pct = distribution.get(SentimentType.VERY_NEGATIVE.value, {}).get('percentage', 0)
        
        if positive_pct > 30 and negative_pct > 30:
            insights.append('POLARIZED_FEEDBACK: High both positive and negative feedback, segment users for personalization')
        
        if comment_analysis and comment_analysis.get('comment_sentiment', {}).get('sentiment_clarity', 0) < 0.6:
            insights.append('UNCLEAR_SENTIMENT: Many neutral comments, consider improving recommendation explanation')
        
        return insights
    
    async def _analyze_comment_topics(self, comment_events: List[UserFeedbackEvent]) -> Dict[str, Any]:
        """Analyze topics mentioned in comments (simplified version)"""
        
        # Topic categories (simplified)
        topic_keywords = {
            'quality': ['quality', 'material', 'build', 'durable', 'cheap', 'flimsy'],
            'price': ['price', 'expensive', 'cheap', 'value', 'cost', 'affordable'],
            'delivery': ['shipping', 'delivery', 'fast', 'slow', 'arrived', 'package'],
            'design': ['design', 'look', 'appearance', 'color', 'style', 'beautiful', 'ugly'],
            'usability': ['easy', 'difficult', 'user', 'interface', 'complicated', 'simple']
        }
        
        topic_mentions = defaultdict(int)
        topic_sentiments = defaultdict(list)
        
        for event in comment_events:
            if event.comment_keywords:
                for keyword in event.comment_keywords:
                    for topic, topic_words in topic_keywords.items():
                        if keyword in topic_words:
                            topic_mentions[topic] += 1
                            topic_sentiments[topic].append(event.comment_sentiment_score)
        
        topic_analysis = {}
        for topic, mentions in topic_mentions.items():
            topic_analysis[topic] = {
                'mention_count': mentions,
                'avg_sentiment': np.mean(topic_sentiments[topic]) if topic_sentiments[topic] else 0,
                'sentiment_std': np.std(topic_sentiments[topic]) if topic_sentiments[topic] else 0
            }
        
        return topic_analysis
    
    async def _analyze_comment_trends(self, comment_events: List[UserFeedbackEvent]) -> Dict[str, Any]:
        """Analyze comment trends over time"""
        
        if len(comment_events) < 10:
            return {'error': 'insufficient_data_for_trends'}
        
        # Sort by timestamp
        sorted_events = sorted(comment_events, key=lambda x: x.timestamp)
        
        # Split into time buckets (hourly)
        time_buckets = defaultdict(list)
        for event in sorted_events:
            hour_bucket = int(event.timestamp // 3600)
            time_buckets[hour_bucket].append(event.comment_sentiment_score)
        
        # Calculate trend
        bucket_sentiments = []
        bucket_times = []
        
        for bucket_time in sorted(time_buckets.keys()):
            bucket_sentiments.append(np.mean(time_buckets[bucket_time]))
            bucket_times.append(bucket_time)
        
        # Simple trend analysis
        if len(bucket_sentiments) >= 3:
            trend_slope = np.polyfit(range(len(bucket_sentiments)), bucket_sentiments, 1)[0]
            trend_direction = 'IMPROVING' if trend_slope > 0.01 else 'DECLINING' if trend_slope < -0.01 else 'STABLE'
        else:
            trend_slope = 0
            trend_direction = 'INSUFFICIENT_DATA'
        
        return {
            'sentiment_trend': {
                'direction': trend_direction,
                'slope': trend_slope,
                'recent_sentiment': bucket_sentiments[-1] if bucket_sentiments else 0,
                'early_sentiment': bucket_sentiments[0] if bucket_sentiments else 0
            },
            'volume_trend': {
                'total_hours_with_comments': len(time_buckets),
                'avg_comments_per_hour': len(comment_events) / max(len(time_buckets), 1)
            }
        }
    
    def _calculate_comment_engagement_score(self, comment_events: List[UserFeedbackEvent]) -> float:
        """Calculate overall comment engagement score"""
        if not comment_events:
            return 0.0
        
        # Factors for engagement score
        avg_length = np.mean([e.comment_length for e in comment_events])
        sentiment_clarity = 1 - len([e for e in comment_events if abs(e.comment_sentiment_score) < 0.1]) / len(comment_events)
        keyword_richness = np.mean([len(e.comment_keywords) for e in comment_events])
        
        # Engagement score (0-10 scale)
        engagement_score = (
            min(avg_length / 100, 1) * 4 +        # Length factor (max 4 points)
            sentiment_clarity * 3 +                # Clarity factor (max 3 points)
            min(keyword_richness / 5, 1) * 3      # Richness factor (max 3 points)
        )
        
        return engagement_score
    
    def _calculate_explicit_satisfaction(self, events: List[UserFeedbackEvent]) -> float:
        """Calculate satisfaction score from explicit feedback"""
        explicit_events = [e for e in events if e.feedback_type == FeedbackType.EXPLICIT]
        
        if not explicit_events:
            return 5.0  # Neutral score
        
        scores = []
        for event in explicit_events:
            if event.rating_score:
                scores.append(event.rating_score * 2)  # Convert 1-5 to 0-10 scale
            elif event.like_dislike is not None:
                scores.append(8.0 if event.like_dislike else 2.0)
        
        return np.mean(scores) if scores else 5.0
    
    def _calculate_behavioral_satisfaction(self, events: List[UserFeedbackEvent]) -> float:
        """Calculate satisfaction score from behavioral signals"""
        behavioral_scores = []
        
        for event in events:
            score = 5.0  # Start with neutral
            
            # Dwell time factor
            if event.dwell_time_seconds > 60:
                score += 2.0
            elif event.dwell_time_seconds > 30:
                score += 1.0
            elif event.dwell_time_seconds < 5:
                score -= 2.0
            
            # Interaction depth factor
            if event.interaction_depth > 0.7:
                score += 1.5
            elif event.interaction_depth > 0.3:
                score += 0.5
            
            # Follow-up actions
            if 'purchase' in event.follow_up_actions:
                score += 2.0
            elif 'add_cart' in event.follow_up_actions:
                score += 1.0
            elif 'share' in event.follow_up_actions:
                score += 1.5
            
            behavioral_scores.append(max(0, min(10, score)))
        
        return np.mean(behavioral_scores) if behavioral_scores else 5.0
    
    def _calculate_comment_satisfaction(self, events: List[UserFeedbackEvent]) -> float:
        """Calculate satisfaction score from comments"""
        comment_events = [e for e in events if e.comment_text]
        
        if not comment_events:
            return 5.0  # Neutral if no comments
        
        sentiment_scores = [e.comment_sentiment_score for e in comment_events]
        
        # Convert sentiment (-1 to 1) to satisfaction (0 to 10)
        satisfaction_scores = [(score + 1) * 5 for score in sentiment_scores]
        
        return np.mean(satisfaction_scores)
    
    def _categorize_satisfaction_scores(self, scores: List[float]) -> Dict[str, int]:
        """Categorize satisfaction scores into buckets"""
        return {
            'excellent': len([s for s in scores if s >= 8.0]),      # 8-10
            'good': len([s for s in scores if 6.0 <= s < 8.0]),     # 6-8
            'average': len([s for s in scores if 4.0 <= s < 6.0]),  # 4-6
            'poor': len([s for s in scores if 2.0 <= s < 4.0]),     # 2-4
            'terrible': len([s for s in scores if s < 2.0])         # 0-2
        }
    
    def _analyze_satisfaction_patterns(self, item_scores: Dict[str, Dict], events: List[UserFeedbackEvent]) -> Dict[str, Any]:
        """Analyze patterns in satisfaction scores"""
        
        # Find best and worst performing items
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1]['overall_satisfaction'], reverse=True)
        
        best_items = sorted_items[:5]
        worst_items = sorted_items[-5:]
        
        # Analyze satisfaction by recommendation source
        source_satisfaction = defaultdict(list)
        for event in events:
            item_score = item_scores.get(event.item_id, {}).get('overall_satisfaction', 5.0)
            source_satisfaction[event.recommendation_source or 'unknown'].append(item_score)
        
        source_analysis = {
            source: {
                'avg_satisfaction': np.mean(scores),
                'item_count': len(scores)
            }
            for source, scores in source_satisfaction.items()
        }
        
        return {
            'best_performing_items': [{'item_id': item[0], 'score': item[1]['overall_satisfaction']} for item in best_items],
            'worst_performing_items': [{'item_id': item[0], 'score': item[1]['overall_satisfaction']} for item in worst_items],
            'source_performance': source_analysis
        }
    
    def _grade_recommendation_quality(self, avg_satisfaction: float) -> str:
        """Grade overall recommendation quality"""
        if avg_satisfaction >= 8.0:
            return 'A_EXCELLENT'
        elif avg_satisfaction >= 7.0:
            return 'B_GOOD'
        elif avg_satisfaction >= 6.0:
            return 'C_AVERAGE'
        elif avg_satisfaction >= 5.0:
            return 'D_BELOW_AVERAGE'
        else:
            return 'F_POOR'
    
    def _identify_satisfaction_improvement_priorities(self, item_scores: Dict[str, Dict]) -> List[str]:
        """Identify priorities for improving satisfaction"""
        priorities = []
        
        low_satisfaction_items = [
            item_id for item_id, scores in item_scores.items()
            if scores['overall_satisfaction'] < 5.0 and scores['feedback_count'] >= 5
        ]
        
        if len(low_satisfaction_items) > len(item_scores) * 0.2:
            priorities.append('HIGH_PRIORITY: Over 20% of items have low satisfaction scores')
        
        # Check for items with conflicting signals
        conflicting_items = [
            item_id for item_id, scores in item_scores.items()
            if abs(scores['explicit_score'] - scores['behavioral_score']) > 3.0
        ]
        
        if conflicting_items:
            priorities.append(f'MEDIUM_PRIORITY: {len(conflicting_items)} items show conflicting satisfaction signals')
        
        return priorities
    
    def _assess_comment_quality(self, quality_indicators: Dict[str, float]) -> str:
        """Assess overall comment quality"""
        avg_length = quality_indicators.get('avg_comment_length', 0)
        engagement_score = quality_indicators.get('engagement_score', 0)
        detailed_ratio = quality_indicators.get('detailed_comments_ratio', 0)
        
        if avg_length > 80 and engagement_score > 7 and detailed_ratio > 0.4:
            return 'HIGH_QUALITY_COMMENTS'
        elif avg_length > 40 and engagement_score > 5:
            return 'MODERATE_QUALITY_COMMENTS'
        else:
            return 'LOW_QUALITY_COMMENTS'
    
    def _generate_comment_improvement_suggestions(self, quality_indicators: Dict, topic_analysis: Dict) -> List[str]:
        """Generate suggestions for improving comment quality"""
        suggestions = []
        
        if quality_indicators.get('avg_comment_length', 0) < 30:
            suggestions.append('ENCOURAGE_DETAILED_FEEDBACK: Add prompts to encourage more detailed comments')
        
        if quality_indicators.get('engagement_score', 0) < 5:
            suggestions.append('IMPROVE_ENGAGEMENT: Make comment interface more engaging and user-friendly')
        
        # Topic-specific suggestions
        if 'price' in topic_analysis and topic_analysis['price'].get('avg_sentiment', 0) < -0.3:
            suggestions.append('PRICE_CONCERN: Address pricing concerns in recommendations')
        
        if 'quality' in topic_analysis and topic_analysis['quality'].get('avg_sentiment', 0) < -0.2:
            suggestions.append('QUALITY_CONCERN: Focus on higher quality item recommendations')
        
        return suggestions


# Demo function
async def demonstrate_sentiment_analysis():
    """Demonstrate comprehensive sentiment and feedback analysis"""
    
    print("=" * 80)
    print("User Sentiment and Feedback Analysis Demo")
    print("=" * 80)
    
    analyzer = UserSentimentAnalyzer()
    
    # Simulate diverse feedback events
    print("\nSimulating user feedback events...")
    
    current_time = time.time()
    sample_feedback = []
    
    # Create realistic feedback scenarios
    feedback_scenarios = [
        # Positive feedback
        UserFeedbackEvent(
            user_id="user_001", item_id="item_005", recommendation_id="rec_001",
            session_id="sess_001", feedback_type=FeedbackType.EXPLICIT,
            sentiment=SentimentType.VERY_POSITIVE, timestamp=current_time - 1800,
            rating_score=5.0, like_dislike=True, comment_text="Love this product! Excellent quality and fast delivery.",
            dwell_time_seconds=180, interaction_depth=0.8, follow_up_actions=['purchase', 'share']
        ),
        
        # Negative feedback with detailed comment
        UserFeedbackEvent(
            user_id="user_002", item_id="item_010", recommendation_id="rec_002", 
            session_id="sess_002", feedback_type=FeedbackType.COMMENT,
            sentiment=SentimentType.NEGATIVE, timestamp=current_time - 3600,
            comment_text="Disappointed with the quality. Price was too high for what I received. Would not recommend.",
            dwell_time_seconds=45, interaction_depth=0.3
        ),
        
        # Mixed behavioral signals
        UserFeedbackEvent(
            user_id="user_003", item_id="item_015", recommendation_id="rec_003",
            session_id="sess_003", feedback_type=FeedbackType.IMPLICIT,
            sentiment=SentimentType.POSITIVE, timestamp=current_time - 900,
            dwell_time_seconds=240, interaction_depth=0.9, follow_up_actions=['add_cart', 'favorite']
        ),
        
        # Neutral feedback
        UserFeedbackEvent(
            user_id="user_004", item_id="item_020", recommendation_id="rec_004",
            session_id="sess_004", feedback_type=FeedbackType.EXPLICIT,
            sentiment=SentimentType.NEUTRAL, timestamp=current_time - 2700,
            rating_score=3.0, comment_text="Average product. Nothing special but does the job.",
            dwell_time_seconds=60, interaction_depth=0.4
        )
    ]
    
    # Add more events for statistical significance
    for i in range(50):
        event = UserFeedbackEvent(
            user_id=f"user_{i % 10:03d}",
            item_id=f"item_{i % 20:03d}",
            recommendation_id=f"rec_{i:03d}",
            session_id=f"sess_{i:03d}",
            feedback_type=np.random.choice(list(FeedbackType)),
            sentiment=np.random.choice(list(SentimentType)),
            timestamp=current_time - np.random.exponential(7200),  # Last 2 hours
            rating_score=np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
            like_dislike=np.random.choice([True, False, None]),
            dwell_time_seconds=max(5, np.random.exponential(60)),
            interaction_depth=np.random.beta(2, 3)
        )
        sample_feedback.append(event)
    
    # Record all feedback events
    for event in sample_feedback:
        await analyzer.record_feedback_event(event)
    
    print(f"Recorded {len(sample_feedback)} feedback events")
    
    # 1. Overall sentiment analysis
    print("\n1. SENTIMENT METRICS ANALYSIS")
    print("-" * 50)
    sentiment_results = await analyzer.calculate_sentiment_metrics(time_range_hours=2)
    
    overall = sentiment_results['overall_metrics']
    print(f"Overall Sentiment Score: {overall['overall_sentiment_score']:.3f} (-1 to 1 scale)")
    print(f"Positive Ratio: {overall['positive_ratio']:.3f}")
    print(f"Negative Ratio: {overall['negative_ratio']:.3f}")
    print(f"Total Feedback Events: {overall['total_feedback_events']}")
    
    quality = sentiment_results['quality_assessment']
    print(f"Sentiment Assessment: {quality['sentiment_assessment']}")
    print(f"Feedback Volume: {quality['feedback_volume']}")
    
    # 2. Comment quality analysis
    print("\n2. COMMENT QUALITY ANALYSIS")
    print("-" * 50)
    comment_results = await analyzer.calculate_comment_quality_metrics(time_range_hours=24)
    
    if 'error' not in comment_results:
        comment_volume = comment_results['comment_volume']
        quality_indicators = comment_results['quality_indicators']
        
        print(f"Total Comments: {comment_volume['total_comments']}")
        print(f"Average Comment Length: {quality_indicators['avg_comment_length']:.1f} characters")
        print(f"Comment Engagement Score: {quality_indicators['engagement_score']:.2f}/10")
        print(f"Average Sentiment: {quality_indicators['avg_sentiment_score']:.3f}")
        
        if comment_results['keyword_analysis']['top_keywords']:
            print(f"Top Keywords: {', '.join([kw[0] for kw in comment_results['keyword_analysis']['top_keywords'][:5]])}")
    
    # 3. Recommendation satisfaction
    print("\n3. RECOMMENDATION SATISFACTION ANALYSIS")
    print("-" * 50)
    satisfaction_results = await analyzer.calculate_recommendation_satisfaction_score(time_range_hours=6)
    
    if 'error' not in satisfaction_results:
        system_satisfaction = satisfaction_results['system_satisfaction']
        print(f"Average Satisfaction Score: {system_satisfaction['avg_satisfaction_score']:.2f}/10")
        print(f"Recommendation Quality Grade: {satisfaction_results['recommendation_quality_grade']}")
        print(f"High Satisfaction Items: {system_satisfaction['high_satisfaction_items']}")
        print(f"Low Satisfaction Items: {system_satisfaction['low_satisfaction_items']}")
        
        satisfaction_dist = system_satisfaction['satisfaction_distribution']
        print(f"Satisfaction Distribution: Excellent({satisfaction_dist['excellent']}), Good({satisfaction_dist['good']}), Average({satisfaction_dist['average']})")
    
    # 4. Actionable insights
    print("\n4. ACTIONABLE INSIGHTS")
    print("-" * 50)
    
    if sentiment_results.get('actionable_insights'):
        print("Sentiment Insights:")
        for insight in sentiment_results['actionable_insights']:
            print(f"  - {insight}")
    
    if satisfaction_results.get('improvement_priorities'):
        print("Satisfaction Improvement Priorities:")
        for priority in satisfaction_results['improvement_priorities']:
            print(f"  - {priority}")
    
    print("\n" + "=" * 80)
    print("Sentiment and Feedback Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_sentiment_analysis())
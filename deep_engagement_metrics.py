"""
Deep User Engagement Metrics for Recommendation Systems
Comprehensive analysis of user behavior patterns including likes, favorites, cart additions, and usage time
Author: Yang Liu
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

@dataclass 
class UserEngagementEvent:
    """User engagement event with detailed tracking"""
    user_id: str
    item_id: str
    session_id: str
    action_type: str  # impression, click, view, like, favorite, add_cart, purchase, share, comment
    timestamp: float
    dwell_time_seconds: float = 0.0
    scroll_depth: float = 0.0  # 0-1, how much content was scrolled
    interaction_quality: str = "normal"  # normal, high_intent, exploratory
    recommendation_position: int = -1
    context: Dict[str, Any] = field(default_factory=dict)


class DeepEngagementAnalyzer:
    """Comprehensive user engagement analysis including deep interaction patterns"""
    
    def __init__(self, window_hours: int = 24):
        self.window_seconds = window_hours * 3600
        self.events: deque = deque()
        
        # Weighted action values for engagement scoring
        self.action_weights = {
            'impression': 0.1,
            'click': 0.3,
            'view': 0.5,
            'like': 0.7,
            'favorite': 0.8,
            'add_cart': 0.9,
            'share': 0.85,
            'comment': 0.75,
            'purchase': 1.0
        }
        
        # Engagement thresholds
        self.engagement_thresholds = {
            'dwell_time_engaged': 30,  # seconds
            'dwell_time_deep': 120,    # seconds
            'scroll_engaged': 0.5,     # 50% scroll
            'scroll_deep': 0.8,        # 80% scroll
        }
    
    async def record_engagement_event(self, event: UserEngagementEvent):
        """Record user engagement event and update metrics"""
        self.events.append(event)
        await self._cleanup_expired_events()
    
    async def _cleanup_expired_events(self):
        """Remove expired events from sliding window"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()
    
    async def calculate_deep_engagement_ratios(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """
        Calculate deep engagement ratios including likes, favorites, cart additions
        
        Key Ratios:
        - Like Rate: likes / views
        - Favorite Rate: favorites / views  
        - Add to Cart Rate: add_carts / views
        - Share Rate: shares / views
        - Comment Rate: comments / views
        - Engagement Progression: impression → click → view → engage → convert
        """
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {'error': 'no_recent_events'}
        
        # Count events by action type
        action_counts = defaultdict(int)
        for event in recent_events:
            action_counts[event.action_type] += 1
        
        # Base metrics
        impressions = action_counts.get('impression', 0)
        clicks = action_counts.get('click', 0)
        views = action_counts.get('view', 0)
        likes = action_counts.get('like', 0)
        favorites = action_counts.get('favorite', 0)
        add_carts = action_counts.get('add_cart', 0)
        shares = action_counts.get('share', 0)
        comments = action_counts.get('comment', 0)
        purchases = action_counts.get('purchase', 0)
        
        # Safe division helper
        def safe_divide(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator > 0 else 0.0
        
        # Calculate engagement ratios
        engagement_ratios = {
            # Basic funnel
            'ctr': safe_divide(clicks, impressions),
            'view_rate': safe_divide(views, clicks),
            
            # Deep engagement ratios (based on views as denominator)
            'like_rate': safe_divide(likes, views),
            'favorite_rate': safe_divide(favorites, views), 
            'add_cart_rate': safe_divide(add_carts, views),
            'share_rate': safe_divide(shares, views),
            'comment_rate': safe_divide(comments, views),
            'purchase_rate': safe_divide(purchases, views),
            
            # Alternative ratios (based on impressions)
            'impression_to_like_rate': safe_divide(likes, impressions),
            'impression_to_favorite_rate': safe_divide(favorites, impressions),
            'impression_to_cart_rate': safe_divide(add_carts, impressions),
            
            # Combined engagement indicators
            'total_engagement_actions': likes + favorites + add_carts + shares + comments,
            'engagement_rate': safe_divide(likes + favorites + add_carts + shares + comments, views),
            'positive_signal_rate': safe_divide(likes + favorites + shares, views),
            'intent_signal_rate': safe_divide(add_carts + purchases, views)
        }
        
        # Calculate engagement progression analysis
        progression_analysis = {
            'impression_to_click_dropoff': 1 - engagement_ratios['ctr'],
            'click_to_view_dropoff': 1 - engagement_ratios['view_rate'],
            'view_to_engagement_dropoff': 1 - engagement_ratios['engagement_rate'],
            'view_to_purchase_dropoff': 1 - engagement_ratios['purchase_rate']
        }
        
        # Identify engagement bottlenecks
        max_dropoff = max(progression_analysis.items(), key=lambda x: x[1])
        
        # Calculate user satisfaction indicators
        satisfaction_score = (
            engagement_ratios['like_rate'] * 0.3 +
            engagement_ratios['favorite_rate'] * 0.25 +
            engagement_ratios['share_rate'] * 0.2 +
            engagement_ratios['add_cart_rate'] * 0.15 +
            engagement_ratios['purchase_rate'] * 0.1
        )
        
        return {
            'action_counts': dict(action_counts),
            'engagement_ratios': engagement_ratios,
            'progression_analysis': progression_analysis,
            'biggest_engagement_bottleneck': {
                'stage': max_dropoff[0],
                'dropoff_rate': max_dropoff[1],
                'severity': 'HIGH' if max_dropoff[1] > 0.8 else 'MEDIUM' if max_dropoff[1] > 0.6 else 'LOW'
            },
            'user_satisfaction_score': satisfaction_score,
            'engagement_quality_assessment': self._assess_engagement_quality(engagement_ratios),
            'optimization_recommendations': self._get_engagement_optimization_tips(engagement_ratios, progression_analysis)
        }
    
    async def calculate_usage_time_metrics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Calculate comprehensive usage time and session metrics
        
        Metrics:
        - Session duration distribution
        - Dwell time per item analysis
        - Time-based engagement patterns
        - Usage intensity indicators
        """
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {'error': 'no_recent_events'}
        
        # Group events by session
        session_events = defaultdict(list)
        for event in recent_events:
            session_events[event.session_id].append(event)
        
        # Calculate session-level metrics
        session_metrics = []
        total_dwell_times = []
        
        for session_id, events in session_events.items():
            if len(events) < 2:
                continue
                
            # Sort events by timestamp
            events_sorted = sorted(events, key=lambda x: x.timestamp)
            
            # Session duration (from first to last event)
            session_start = events_sorted[0].timestamp
            session_end = events_sorted[-1].timestamp
            session_duration = session_end - session_start
            
            # Count different action types in session
            session_actions = defaultdict(int)
            session_dwell_total = 0
            session_scroll_depths = []
            
            for event in events_sorted:
                session_actions[event.action_type] += 1
                if event.dwell_time_seconds > 0:
                    session_dwell_total += event.dwell_time_seconds
                    total_dwell_times.append(event.dwell_time_seconds)
                if event.scroll_depth > 0:
                    session_scroll_depths.append(event.scroll_depth)
            
            # Calculate session engagement score
            session_engagement_score = sum(
                count * self.action_weights.get(action, 0) 
                for action, count in session_actions.items()
            )
            
            session_metrics.append({
                'session_id': session_id,
                'duration_seconds': session_duration,
                'total_dwell_seconds': session_dwell_total,
                'num_events': len(events_sorted),
                'unique_items': len(set(e.item_id for e in events_sorted)),
                'engagement_score': session_engagement_score,
                'avg_scroll_depth': np.mean(session_scroll_depths) if session_scroll_depths else 0,
                'actions_per_minute': len(events_sorted) / max(session_duration / 60, 0.1),
                'converted': any(e.action_type == 'purchase' for e in events_sorted)
            })
        
        if not session_metrics:
            return {'error': 'no_valid_sessions'}
        
        # Aggregate session statistics
        session_durations = [s['duration_seconds'] for s in session_metrics]
        total_dwell_seconds = [s['total_dwell_seconds'] for s in session_metrics]
        engagement_scores = [s['engagement_score'] for s in session_metrics]
        
        # Dwell time analysis
        dwell_time_analysis = {}
        if total_dwell_times:
            dwell_time_analysis = {
                'mean_dwell_seconds': np.mean(total_dwell_times),
                'median_dwell_seconds': np.median(total_dwell_times),
                'p75_dwell_seconds': np.percentile(total_dwell_times, 75),
                'p95_dwell_seconds': np.percentile(total_dwell_times, 95),
                'dwell_time_distribution': self._categorize_dwell_times(total_dwell_times),
                'quality_engagement_rate': len([t for t in total_dwell_times if t >= self.engagement_thresholds['dwell_time_engaged']]) / len(total_dwell_times),
                'deep_engagement_rate': len([t for t in total_dwell_times if t >= self.engagement_thresholds['dwell_time_deep']]) / len(total_dwell_times)
            }
        
        # Session analysis
        session_analysis = {
            'total_sessions': len(session_metrics),
            'avg_session_duration_seconds': np.mean(session_durations),
            'median_session_duration_seconds': np.median(session_durations),
            'avg_events_per_session': np.mean([s['num_events'] for s in session_metrics]),
            'avg_items_per_session': np.mean([s['unique_items'] for s in session_metrics]),
            'conversion_rate_per_session': len([s for s in session_metrics if s['converted']]) / len(session_metrics),
            'high_engagement_sessions': len([s for s in session_metrics if s['engagement_score'] > 5]) / len(session_metrics),
            'session_duration_distribution': self._categorize_session_durations(session_durations)
        }
        
        # Time-based usage patterns
        usage_patterns = self._analyze_usage_patterns(recent_events)
        
        # Overall usage intensity score
        usage_intensity = self._calculate_usage_intensity_score(session_metrics, total_dwell_times)
        
        return {
            'session_analysis': session_analysis,
            'dwell_time_analysis': dwell_time_analysis,
            'usage_patterns': usage_patterns,
            'usage_intensity_score': usage_intensity,
            'detailed_session_metrics': session_metrics[:10],  # Top 10 sessions for debugging
            'time_range_hours': time_range_hours,
            'total_events_analyzed': len(recent_events),
            'usage_quality_assessment': self._assess_usage_quality(session_analysis, dwell_time_analysis),
            'optimization_insights': self._get_usage_optimization_insights(session_analysis, dwell_time_analysis)
        }
    
    async def calculate_user_behavior_cohorts(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Segment users based on engagement behavior patterns
        
        Cohorts:
        - Power Users: High engagement across multiple dimensions
        - Casual Browsers: High view time, low interaction
        - Quick Deciders: Short time, high conversion
        - Explorers: High diversity, moderate engagement
        - Passive Users: Low engagement overall
        """
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Group events by user
        user_events = defaultdict(list)
        for event in recent_events:
            user_events[event.user_id].append(event)
        
        user_cohorts = {
            'power_users': [],
            'casual_browsers': [],
            'quick_deciders': [],
            'explorers': [],
            'passive_users': []
        }
        
        for user_id, events in user_events.items():
            if len(events) < 2:
                continue
                
            # Calculate user behavior metrics
            action_counts = defaultdict(int)
            total_dwell = sum(e.dwell_time_seconds for e in events)
            unique_items = len(set(e.item_id for e in events))
            
            for event in events:
                action_counts[event.action_type] += 1
            
            # User behavior indicators
            engagement_actions = action_counts.get('like', 0) + action_counts.get('favorite', 0) + action_counts.get('share', 0)
            conversion_actions = action_counts.get('add_cart', 0) + action_counts.get('purchase', 0)
            view_actions = action_counts.get('view', 0)
            
            avg_dwell_per_item = total_dwell / max(unique_items, 1)
            engagement_rate = engagement_actions / max(len(events), 1)
            conversion_rate = conversion_actions / max(view_actions, 1)
            item_diversity = unique_items / max(len(events), 1)
            
            # Classify user into cohort
            user_profile = {
                'user_id': user_id,
                'total_events': len(events),
                'engagement_rate': engagement_rate,
                'conversion_rate': conversion_rate,
                'avg_dwell_per_item': avg_dwell_per_item,
                'item_diversity': item_diversity,
                'total_dwell_time': total_dwell
            }
            
            # Cohort classification logic
            if engagement_rate > 0.3 and conversion_rate > 0.2 and total_dwell > 300:
                user_cohorts['power_users'].append(user_profile)
            elif avg_dwell_per_item > 60 and engagement_rate < 0.1:
                user_cohorts['casual_browsers'].append(user_profile)
            elif conversion_rate > 0.4 and avg_dwell_per_item < 30:
                user_cohorts['quick_deciders'].append(user_profile)
            elif item_diversity > 0.7 and engagement_rate > 0.1:
                user_cohorts['explorers'].append(user_profile)
            else:
                user_cohorts['passive_users'].append(user_profile)
        
        # Calculate cohort statistics
        cohort_stats = {}
        total_users = sum(len(cohort) for cohort in user_cohorts.values())
        
        for cohort_name, cohort_users in user_cohorts.items():
            if cohort_users:
                cohort_stats[cohort_name] = {
                    'user_count': len(cohort_users),
                    'percentage': len(cohort_users) / max(total_users, 1) * 100,
                    'avg_engagement_rate': np.mean([u['engagement_rate'] for u in cohort_users]),
                    'avg_conversion_rate': np.mean([u['conversion_rate'] for u in cohort_users]),
                    'avg_dwell_time': np.mean([u['total_dwell_time'] for u in cohort_users]),
                    'avg_item_diversity': np.mean([u['item_diversity'] for u in cohort_users])
                }
            else:
                cohort_stats[cohort_name] = {'user_count': 0, 'percentage': 0}
        
        return {
            'cohort_distribution': cohort_stats,
            'total_users_analyzed': total_users,
            'cohort_insights': self._generate_cohort_insights(cohort_stats),
            'personalization_strategies': self._get_cohort_personalization_strategies(cohort_stats),
            'detailed_cohorts': {k: v[:5] for k, v in user_cohorts.items()}  # Top 5 users per cohort for debugging
        }
    
    # Helper methods for analysis
    
    def _assess_engagement_quality(self, ratios: Dict[str, float]) -> str:
        """Assess overall engagement quality based on ratios"""
        engagement_score = (
            ratios.get('like_rate', 0) * 0.3 +
            ratios.get('favorite_rate', 0) * 0.25 +
            ratios.get('add_cart_rate', 0) * 0.2 +
            ratios.get('share_rate', 0) * 0.15 +
            ratios.get('purchase_rate', 0) * 0.1
        )
        
        if engagement_score > 0.15:
            return 'EXCELLENT_ENGAGEMENT'
        elif engagement_score > 0.08:
            return 'GOOD_ENGAGEMENT'
        elif engagement_score > 0.04:
            return 'MODERATE_ENGAGEMENT'
        else:
            return 'LOW_ENGAGEMENT'
    
    def _get_engagement_optimization_tips(self, ratios: Dict[str, float], progression: Dict[str, float]) -> List[str]:
        """Generate engagement optimization recommendations"""
        tips = []
        
        if ratios.get('like_rate', 0) < 0.05:
            tips.append('LOW_LIKE_RATE: Consider improving content quality and personalization')
        
        if ratios.get('favorite_rate', 0) < 0.03:
            tips.append('LOW_FAVORITE_RATE: Make favorite/save functionality more prominent')
        
        if ratios.get('add_cart_rate', 0) < 0.08:
            tips.append('LOW_CART_RATE: Optimize product presentation and pricing visibility')
        
        if ratios.get('share_rate', 0) < 0.02:
            tips.append('LOW_SHARE_RATE: Add social sharing incentives and improve viral features')
        
        # Find biggest dropoff point
        max_dropoff = max(progression.items(), key=lambda x: x[1])
        if max_dropoff[1] > 0.7:
            tips.append(f'MAJOR_BOTTLENECK: Focus on improving {max_dropoff[0].replace("_dropoff", "")} stage')
        
        return tips
    
    def _categorize_dwell_times(self, dwell_times: List[float]) -> Dict[str, int]:
        """Categorize dwell times into engagement buckets"""
        return {
            'bounce': len([t for t in dwell_times if t < 5]),
            'brief': len([t for t in dwell_times if 5 <= t < 30]),
            'engaged': len([t for t in dwell_times if 30 <= t < 120]),
            'deep': len([t for t in dwell_times if t >= 120])
        }
    
    def _categorize_session_durations(self, durations: List[float]) -> Dict[str, int]:
        """Categorize session durations"""
        return {
            'very_short': len([d for d in durations if d < 60]),      # < 1 minute
            'short': len([d for d in durations if 60 <= d < 300]),    # 1-5 minutes  
            'medium': len([d for d in durations if 300 <= d < 900]),  # 5-15 minutes
            'long': len([d for d in durations if d >= 900])           # > 15 minutes
        }
    
    def _analyze_usage_patterns(self, events: List[UserEngagementEvent]) -> Dict[str, Any]:
        """Analyze temporal usage patterns"""
        if not events:
            return {}
        
        # Group by hour of day
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for event in events:
            dt = datetime.fromtimestamp(event.timestamp)
            hourly_activity[dt.hour] += 1
            daily_activity[dt.weekday()] += 1
        
        # Find peak hours and days
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1])
        peak_day = max(daily_activity.items(), key=lambda x: x[1])
        
        return {
            'hourly_distribution': dict(hourly_activity),
            'daily_distribution': dict(daily_activity),
            'peak_hour': {'hour': peak_hour[0], 'activity_count': peak_hour[1]},
            'peak_day': {'day': peak_day[0], 'activity_count': peak_day[1]},
            'total_active_hours': len(hourly_activity),
            'total_active_days': len(daily_activity)
        }
    
    def _calculate_usage_intensity_score(self, session_metrics: List[Dict], dwell_times: List[float]) -> float:
        """Calculate overall usage intensity score (0-10)"""
        if not session_metrics or not dwell_times:
            return 0.0
        
        # Session intensity factors
        avg_session_duration = np.mean([s['duration_seconds'] for s in session_metrics])
        avg_events_per_session = np.mean([s['num_events'] for s in session_metrics])
        conversion_rate = len([s for s in session_metrics if s['converted']]) / len(session_metrics)
        
        # Dwell time factors
        avg_dwell_time = np.mean(dwell_times)
        engaged_dwell_rate = len([t for t in dwell_times if t >= 30]) / len(dwell_times)
        
        # Combined intensity score (0-10 scale)
        intensity_score = (
            min(avg_session_duration / 300, 1) * 2 +    # Session duration factor (max 2 points)
            min(avg_events_per_session / 10, 1) * 2 +   # Events per session factor (max 2 points) 
            conversion_rate * 3 +                        # Conversion factor (max 3 points)
            min(avg_dwell_time / 60, 1) * 2 +          # Dwell time factor (max 2 points)
            engaged_dwell_rate * 1                      # Engagement rate factor (max 1 point)
        )
        
        return min(intensity_score, 10.0)
    
    def _assess_usage_quality(self, session_analysis: Dict, dwell_analysis: Dict) -> str:
        """Assess overall usage quality"""
        avg_session_duration = session_analysis.get('avg_session_duration_seconds', 0)
        conversion_rate = session_analysis.get('conversion_rate_per_session', 0)
        quality_engagement_rate = dwell_analysis.get('quality_engagement_rate', 0)
        
        if avg_session_duration > 600 and conversion_rate > 0.3 and quality_engagement_rate > 0.6:
            return 'PREMIUM_USAGE_QUALITY'
        elif avg_session_duration > 300 and conversion_rate > 0.15 and quality_engagement_rate > 0.4:
            return 'GOOD_USAGE_QUALITY'
        elif avg_session_duration > 120 and quality_engagement_rate > 0.2:
            return 'MODERATE_USAGE_QUALITY'
        else:
            return 'LOW_USAGE_QUALITY'
    
    def _get_usage_optimization_insights(self, session_analysis: Dict, dwell_analysis: Dict) -> List[str]:
        """Generate usage optimization insights"""
        insights = []
        
        avg_session_duration = session_analysis.get('avg_session_duration_seconds', 0)
        if avg_session_duration < 120:
            insights.append('SHORT_SESSIONS: Focus on increasing session stickiness and content depth')
        
        conversion_rate = session_analysis.get('conversion_rate_per_session', 0)
        if conversion_rate < 0.1:
            insights.append('LOW_CONVERSION: Improve purchase funnel and reduce friction')
        
        quality_engagement_rate = dwell_analysis.get('quality_engagement_rate', 0)
        if quality_engagement_rate < 0.3:
            insights.append('LOW_ENGAGEMENT: Enhance content relevance and interaction design')
        
        return insights
    
    def _generate_cohort_insights(self, cohort_stats: Dict) -> List[str]:
        """Generate insights about user cohorts"""
        insights = []
        
        total_users = sum(stats.get('user_count', 0) for stats in cohort_stats.values())
        
        for cohort_name, stats in cohort_stats.items():
            percentage = stats.get('percentage', 0)
            
            if cohort_name == 'power_users' and percentage > 15:
                insights.append(f'HIGH_POWER_USER_RATIO: {percentage:.1f}% power users indicate strong product-market fit')
            elif cohort_name == 'passive_users' and percentage > 50:
                insights.append(f'HIGH_PASSIVE_USER_RATIO: {percentage:.1f}% passive users suggest engagement issues')
            elif cohort_name == 'quick_deciders' and percentage > 20:
                insights.append(f'HIGH_QUICK_DECIDER_RATIO: {percentage:.1f}% quick deciders indicate efficient purchase flow')
        
        return insights
    
    def _get_cohort_personalization_strategies(self, cohort_stats: Dict) -> Dict[str, str]:
        """Get personalization strategies for each cohort"""
        strategies = {}
        
        for cohort_name, stats in cohort_stats.items():
            if stats.get('user_count', 0) > 0:
                if cohort_name == 'power_users':
                    strategies[cohort_name] = 'Provide exclusive content, early access, and advanced features'
                elif cohort_name == 'casual_browsers':
                    strategies[cohort_name] = 'Focus on visual appeal and easy discovery mechanisms'
                elif cohort_name == 'quick_deciders':
                    strategies[cohort_name] = 'Streamline purchase flow and highlight key product benefits'
                elif cohort_name == 'explorers':
                    strategies[cohort_name] = 'Emphasize diversity and discovery features'
                elif cohort_name == 'passive_users':
                    strategies[cohort_name] = 'Use gentle engagement tactics and clear value propositions'
        
        return strategies


# Demo function
async def demonstrate_deep_engagement_analysis():
    """Demonstrate deep engagement analysis capabilities"""
    
    print("=" * 80)
    print("Deep User Engagement Metrics Analysis Demo")
    print("=" * 80)
    
    analyzer = DeepEngagementAnalyzer()
    
    # Simulate engagement events
    print("\nSimulating user engagement events...")
    
    sample_events = []
    current_time = time.time()
    
    # Create diverse user engagement scenarios
    for i in range(1000):
        event = UserEngagementEvent(
            user_id=f"user_{i % 50}",
            item_id=f"item_{i % 100}", 
            session_id=f"session_{i % 20}",
            action_type=np.random.choice([
                'impression', 'click', 'view', 'like', 'favorite', 
                'add_cart', 'share', 'comment', 'purchase'
            ], p=[0.4, 0.2, 0.15, 0.08, 0.05, 0.06, 0.02, 0.02, 0.02]),
            timestamp=current_time - np.random.exponential(3600),
            dwell_time_seconds=max(0, np.random.exponential(45)),
            scroll_depth=np.random.beta(2, 5),
            recommendation_position=i % 10
        )
        sample_events.append(event)
    
    # Record events
    for event in sample_events:
        await analyzer.record_engagement_event(event)
    
    print(f"Recorded {len(sample_events)} engagement events")
    
    # 1. Deep engagement ratios analysis
    print("\n1. DEEP ENGAGEMENT RATIOS ANALYSIS")
    print("-" * 50)
    engagement_results = await analyzer.calculate_deep_engagement_ratios(time_range_hours=1)
    
    ratios = engagement_results['engagement_ratios']
    print(f"Like Rate: {ratios['like_rate']:.4f}")
    print(f"Favorite Rate: {ratios['favorite_rate']:.4f}")
    print(f"Add to Cart Rate: {ratios['add_cart_rate']:.4f}")
    print(f"Share Rate: {ratios['share_rate']:.4f}")
    print(f"Overall Engagement Rate: {ratios['engagement_rate']:.4f}")
    print(f"User Satisfaction Score: {engagement_results['user_satisfaction_score']:.4f}")
    print(f"Quality Assessment: {engagement_results['engagement_quality_assessment']}")
    
    # 2. Usage time analysis
    print("\n2. USAGE TIME & SESSION ANALYSIS")
    print("-" * 50)
    usage_results = await analyzer.calculate_usage_time_metrics(time_range_hours=24)
    
    session_analysis = usage_results['session_analysis']
    dwell_analysis = usage_results['dwell_time_analysis']
    
    print(f"Total Sessions: {session_analysis['total_sessions']}")
    print(f"Average Session Duration: {session_analysis['avg_session_duration_seconds']:.1f} seconds")
    print(f"Average Dwell Time: {dwell_analysis['mean_dwell_seconds']:.1f} seconds")
    print(f"Quality Engagement Rate: {dwell_analysis['quality_engagement_rate']:.4f}")
    print(f"Deep Engagement Rate: {dwell_analysis['deep_engagement_rate']:.4f}")
    print(f"Usage Intensity Score: {usage_results['usage_intensity_score']:.2f}/10")
    print(f"Usage Quality: {usage_results['usage_quality_assessment']}")
    
    # 3. User behavior cohorts
    print("\n3. USER BEHAVIOR COHORTS")
    print("-" * 50)
    cohort_results = await analyzer.calculate_user_behavior_cohorts(time_range_hours=24)
    
    print(f"Total Users Analyzed: {cohort_results['total_users_analyzed']}")
    print("\nCohort Distribution:")
    
    for cohort, stats in cohort_results['cohort_distribution'].items():
        print(f"  {cohort.replace('_', ' ').title()}: {stats['user_count']} users ({stats['percentage']:.1f}%)")
    
    print("\nKey Insights:")
    for insight in cohort_results['cohort_insights']:
        print(f"  - {insight}")
    
    print("\n" + "=" * 80)
    print("Deep Engagement Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_deep_engagement_analysis())
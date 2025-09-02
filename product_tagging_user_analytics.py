#!/usr/bin/env python3
"""
Product Tagging System and User Tag Interaction Analytics
Multi-window user behavior statistics for Electronic E-commerce
Author: Yang Liu
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, Counter
import redis
from datetime import datetime, timedelta


class InteractionType(Enum):
    VIEW = "view"
    CLICK = "click" 
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    ADD_CART = "add_cart"
    PURCHASE = "purchase"
    FAVORITE = "favorite"


class TimeWindow(Enum):
    HOUR_1 = "1h"
    HOUR_6 = "6h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    DAY_7 = "7d"
    DAY_30 = "30d"


@dataclass
class ProductTag:
    tag_id: str
    tag_name: str
    tag_category: str
    confidence: float = 1.0
    source: str = "manual"  # manual, ai_generated, user_generated
    created_time: float = 0.0


@dataclass
class TaggedProduct:
    product_id: str
    title: str
    description: str
    category: str
    price: float
    tags: List[ProductTag] = field(default_factory=list)
    auto_tags: Dict[str, float] = field(default_factory=dict)  # AI generated tags with confidence


@dataclass
class UserTagInteraction:
    user_id: str
    tag_id: str
    product_id: str
    interaction_type: InteractionType
    timestamp: float
    session_id: str = ""
    context_info: Dict = field(default_factory=dict)


@dataclass
class UserTagStats:
    tag_id: str
    # Multi-window statistics
    stats_1h: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stats_6h: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stats_1d: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stats_3d: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stats_7d: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stats_30d: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Ratios and preferences
    interaction_ratios: Dict[str, float] = field(default_factory=dict)
    preference_score: float = 0.0
    last_updated: float = 0.0


class ProductTaggingEngine:
    """Product tagging engine for Electronic products"""
    
    def __init__(self):
        self.electronic_tag_templates = self._init_electronic_tag_templates()
        
    def _init_electronic_tag_templates(self) -> Dict[str, List[str]]:
        """Initialize Electronic-specific tag templates"""
        return {
            # Camera Category
            "camera": [
                "mirrorless", "dslr", "full_frame", "aps_c", "micro_four_thirds",
                "4k_video", "8k_video", "image_stabilization", "weather_sealed",
                "professional", "enthusiast", "beginner_friendly", "compact",
                "high_resolution", "low_light", "portrait", "landscape", "macro",
                "sports_photography", "wildlife", "street_photography"
            ],
            
            # Audio Category
            "audio": [
                "wireless", "wired", "noise_canceling", "open_back", "closed_back",
                "over_ear", "on_ear", "in_ear", "true_wireless", "bluetooth",
                "hi_res_audio", "ldac", "aptx", "bass_heavy", "balanced_sound",
                "studio_monitor", "casual_listening", "gaming", "sports",
                "travel_friendly", "foldable", "long_battery", "quick_charge"
            ],
            
            # Gaming Category  
            "gaming": [
                "console", "handheld", "vr_ready", "4k_gaming", "8k_gaming",
                "ray_tracing", "high_refresh", "backward_compatible", 
                "exclusive_games", "multiplayer", "single_player", "indie_games",
                "aaa_games", "family_friendly", "mature_content", "racing",
                "action", "adventure", "rpg", "fps", "strategy", "simulation"
            ],
            
            # TV Category
            "tv": [
                "oled", "led", "qled", "4k", "8k", "hdr10", "dolby_vision",
                "smart_tv", "android_tv", "gaming_mode", "low_input_lag",
                "large_screen", "medium_screen", "small_screen", "curved",
                "flat", "wall_mountable", "premium", "budget", "energy_efficient"
            ],
            
            # Common attributes
            "attributes": [
                "premium", "budget", "mid_range", "flagship", "entry_level",
                "innovative", "reliable", "durable", "lightweight", "portable",
                "stylish", "minimalist", "retro", "futuristic", "professional_grade",
                "consumer", "prosumer", "enterprise", "limited_edition", "new_release"
            ]
        }
    
    def extract_tags_from_text(self, text: str, category: str) -> List[ProductTag]:
        """Extract tags from product text using keyword matching"""
        text_lower = text.lower()
        extracted_tags = []
        
        # Get category-specific tags
        category_tags = self.electronic_tag_templates.get(category, [])
        common_tags = self.electronic_tag_templates.get("attributes", [])
        all_tags = category_tags + common_tags
        
        for tag_keyword in all_tags:
            # Simple keyword matching (can be enhanced with NLP)
            if tag_keyword.replace("_", " ") in text_lower or tag_keyword in text_lower:
                confidence = 0.8 if tag_keyword in category_tags else 0.6
                
                tag = ProductTag(
                    tag_id=tag_keyword,
                    tag_name=tag_keyword.replace("_", " ").title(),
                    tag_category=category,
                    confidence=confidence,
                    source="ai_generated",
                    created_time=time.time()
                )
                extracted_tags.append(tag)
        
        return extracted_tags
    
    def tag_electronic_product(self, product_id: str, title: str, description: str, 
                        category: str, price: float, model_number: str = "") -> TaggedProduct:
        """Comprehensive tagging for Electronic product"""
        
        # Combine text for analysis
        full_text = f"{title} {description} {model_number}".lower()
        
        # Extract tags from text
        auto_tags = self.extract_tags_from_text(full_text, category)
        
        # Price-based tags
        if price > 1500:
            auto_tags.append(ProductTag("premium", "Premium", category, 0.9, "rule_based"))
            auto_tags.append(ProductTag("high_end", "High-end", category, 0.9, "rule_based"))
        elif price > 500:
            auto_tags.append(ProductTag("mid_range", "Mid-range", category, 0.9, "rule_based"))
        else:
            auto_tags.append(ProductTag("budget", "Budget", category, 0.9, "rule_based"))
        
        # Model-specific tags for Electronic
        if "1000x" in full_text:
            auto_tags.append(ProductTag("flagship_audio", "Flagship Audio", category, 0.95, "rule_based"))
        
        if "alpha" in full_text or "fx" in full_text:
            auto_tags.append(ProductTag("professional_camera", "Professional Camera", category, 0.95, "rule_based"))
        
        if "playstation" in full_text or "ps5" in full_text:
            auto_tags.append(ProductTag("gaming_console", "Gaming Console", category, 0.95, "rule_based"))
        
        # Create auto_tags dict for confidence scores
        auto_tags_dict = {tag.tag_id: tag.confidence for tag in auto_tags}
        
        return TaggedProduct(
            product_id=product_id,
            title=title,
            description=description,
            category=category,
            price=price,
            tags=auto_tags,
            auto_tags=auto_tags_dict
        )


class UserTagAnalytics:
    """User tag interaction analytics with multi-window statistics"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.time_windows = {
            TimeWindow.HOUR_1: 3600,
            TimeWindow.HOUR_6: 3600 * 6,
            TimeWindow.DAY_1: 86400,
            TimeWindow.DAY_3: 86400 * 3,
            TimeWindow.DAY_7: 86400 * 7,
            TimeWindow.DAY_30: 86400 * 30
        }
        
        # Interaction weights for preference calculation
        self.interaction_weights = {
            InteractionType.VIEW: 1.0,
            InteractionType.CLICK: 2.0,
            InteractionType.LIKE: 3.0,
            InteractionType.SHARE: 4.0,
            InteractionType.COMMENT: 5.0,
            InteractionType.ADD_CART: 6.0,
            InteractionType.FAVORITE: 7.0,
            InteractionType.PURCHASE: 10.0
        }
    
    def log_interaction(self, user_id: str, tag_id: str, product_id: str,
                       interaction_type: InteractionType, session_id: str = "",
                       context_info: Dict = None) -> None:
        """Log user interaction with a tag"""
        
        interaction = UserTagInteraction(
            user_id=user_id,
            tag_id=tag_id,
            product_id=product_id,
            interaction_type=interaction_type,
            timestamp=time.time(),
            session_id=session_id,
            context_info=context_info or {}
        )
        
        # Store interaction in Redis (for real-time processing)
        interaction_key = f"tag_interaction:{user_id}:{tag_id}:{int(interaction.timestamp)}"
        self.redis_client.hmset(interaction_key, {
            'product_id': product_id,
            'interaction_type': interaction_type.value,
            'timestamp': interaction.timestamp,
            'session_id': session_id,
            'context': json.dumps(context_info or {})
        })
        
        # Set expiration (keep for 31 days)
        self.redis_client.expire(interaction_key, 86400 * 31)
        
        # Update user tag statistics
        self.update_user_tag_stats(user_id, tag_id, interaction_type, interaction.timestamp)
    
    def update_user_tag_stats(self, user_id: str, tag_id: str, 
                             interaction_type: InteractionType, timestamp: float) -> None:
        """Update user tag statistics across all time windows"""
        
        stats_key = f"user_tag_stats:{user_id}:{tag_id}"
        
        # Get existing stats or create new
        existing_stats = self.redis_client.hgetall(stats_key)
        
        if existing_stats:
            stats = json.loads(existing_stats.get('stats', '{}'))
        else:
            stats = {
                '1h': defaultdict(int),
                '6h': defaultdict(int),
                '1d': defaultdict(int),
                '3d': defaultdict(int),
                '7d': defaultdict(int),
                '30d': defaultdict(int)
            }
        
        # Update all time windows
        interaction_name = interaction_type.value
        for window_name in stats.keys():
            stats[window_name][interaction_name] = stats[window_name].get(interaction_name, 0) + 1
        
        # Calculate interaction ratios and preference score
        ratios = self.calculate_interaction_ratios(stats['30d'])  # Use 30-day for ratios
        preference_score = self.calculate_preference_score(stats['30d'])
        
        # Save updated stats
        updated_data = {
            'stats': json.dumps(stats),
            'ratios': json.dumps(ratios),
            'preference_score': preference_score,
            'last_updated': timestamp
        }
        
        self.redis_client.hmset(stats_key, updated_data)
        self.redis_client.expire(stats_key, 86400 * 31)  # 31 days TTL
    
    def calculate_interaction_ratios(self, interaction_counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate interaction type ratios"""
        total_interactions = sum(interaction_counts.values())
        if total_interactions == 0:
            return {}
        
        ratios = {}
        for interaction_type, count in interaction_counts.items():
            ratios[f"{interaction_type}_ratio"] = count / total_interactions
        
        # Calculate conversion ratios
        if interaction_counts.get('view', 0) > 0:
            ratios['click_through_rate'] = interaction_counts.get('click', 0) / interaction_counts['view']
            ratios['like_rate'] = interaction_counts.get('like', 0) / interaction_counts['view']
            ratios['share_rate'] = interaction_counts.get('share', 0) / interaction_counts['view']
            ratios['comment_rate'] = interaction_counts.get('comment', 0) / interaction_counts['view']
        
        if interaction_counts.get('click', 0) > 0:
            ratios['add_cart_rate'] = interaction_counts.get('add_cart', 0) / interaction_counts['click']
            ratios['purchase_conversion'] = interaction_counts.get('purchase', 0) / interaction_counts['click']
        
        return ratios
    
    def calculate_preference_score(self, interaction_counts: Dict[str, int]) -> float:
        """Calculate overall preference score for a tag"""
        total_score = 0
        
        for interaction_type_str, count in interaction_counts.items():
            try:
                interaction_type = InteractionType(interaction_type_str)
                weight = self.interaction_weights.get(interaction_type, 1.0)
                total_score += count * weight
            except ValueError:
                continue
        
        # Normalize by total interactions to get average score per interaction
        total_interactions = sum(interaction_counts.values())
        return total_score / max(total_interactions, 1)
    
    def get_user_tag_stats(self, user_id: str, tag_id: str) -> Optional[UserTagStats]:
        """Get user tag statistics"""
        stats_key = f"user_tag_stats:{user_id}:{tag_id}"
        stats_data = self.redis_client.hgetall(stats_key)
        
        if not stats_data:
            return None
        
        stats_dict = json.loads(stats_data.get('stats', '{}'))
        ratios = json.loads(stats_data.get('ratios', '{}'))
        
        return UserTagStats(
            tag_id=tag_id,
            stats_1h=stats_dict.get('1h', {}),
            stats_6h=stats_dict.get('6h', {}),
            stats_1d=stats_dict.get('1d', {}),
            stats_3d=stats_dict.get('3d', {}),
            stats_7d=stats_dict.get('7d', {}),
            stats_30d=stats_dict.get('30d', {}),
            interaction_ratios=ratios,
            preference_score=float(stats_data.get('preference_score', 0)),
            last_updated=float(stats_data.get('last_updated', 0))
        )
    
    def get_user_top_tags(self, user_id: str, time_window: TimeWindow = TimeWindow.DAY_7,
                         top_k: int = 20) -> List[Tuple[str, float, Dict[str, int]]]:
        """Get user's top tags by preference score in time window"""
        
        # Get all tag stats for user
        pattern = f"user_tag_stats:{user_id}:*"
        tag_keys = self.redis_client.keys(pattern)
        
        tag_scores = []
        window_key = time_window.value
        
        for key in tag_keys:
            tag_id = key.split(':')[-1]
            stats_data = self.redis_client.hgetall(key)
            
            if stats_data:
                stats_dict = json.loads(stats_data.get('stats', '{}'))
                window_stats = stats_dict.get(window_key, {})
                
                if window_stats:
                    preference_score = self.calculate_preference_score(window_stats)
                    tag_scores.append((tag_id, preference_score, window_stats))
        
        # Sort by preference score
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        return tag_scores[:top_k]
    
    def get_tag_interaction_timeline(self, user_id: str, tag_id: str,
                                   days_back: int = 30) -> List[Dict]:
        """Get interaction timeline for a specific tag"""
        
        start_time = time.time() - (days_back * 86400)
        pattern = f"tag_interaction:{user_id}:{tag_id}:*"
        
        interaction_keys = self.redis_client.keys(pattern)
        timeline = []
        
        for key in interaction_keys:
            timestamp = float(key.split(':')[-1])
            if timestamp >= start_time:
                interaction_data = self.redis_client.hgetall(key)
                if interaction_data:
                    timeline.append({
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                        'product_id': interaction_data.get('product_id'),
                        'interaction_type': interaction_data.get('interaction_type'),
                        'session_id': interaction_data.get('session_id'),
                        'context': json.loads(interaction_data.get('context', '{}'))
                    })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline
    
    def analyze_user_tag_patterns(self, user_id: str) -> Dict:
        """Comprehensive analysis of user tag interaction patterns"""
        
        # Get top tags across different time windows
        patterns = {
            'top_tags_1d': self.get_user_top_tags(user_id, TimeWindow.DAY_1, 10),
            'top_tags_7d': self.get_user_top_tags(user_id, TimeWindow.DAY_7, 10),
            'top_tags_30d': self.get_user_top_tags(user_id, TimeWindow.DAY_30, 10),
        }
        
        # Calculate trend changes
        tags_1d = {tag: score for tag, score, _ in patterns['top_tags_1d']}
        tags_7d = {tag: score for tag, score, _ in patterns['top_tags_7d']}
        tags_30d = {tag: score for tag, score, _ in patterns['top_tags_30d']}
        
        # Find trending up/down tags
        trending_up = []
        trending_down = []
        
        for tag in tags_7d:
            score_7d = tags_7d[tag]
            score_30d = tags_30d.get(tag, 0)
            
            if score_7d > score_30d * 1.2:  # 20% increase
                trending_up.append((tag, score_7d, score_30d))
            elif score_7d < score_30d * 0.8:  # 20% decrease  
                trending_down.append((tag, score_7d, score_30d))
        
        patterns['trending_up'] = trending_up
        patterns['trending_down'] = trending_down
        
        return patterns


def demonstrate_product_tagging_analytics():
    """Demonstrate product tagging and user analytics system"""
    print("=" * 80)
    print("Electronic Product Tagging and User Tag Analytics System")
    print("=" * 80)
    
    # Initialize systems
    tagging_engine = ProductTaggingEngine()
    analytics = UserTagAnalytics()
    
    print("\n1. Product Auto-Tagging Examples:")
    print("-" * 50)
    
    # Sample Electronic products
    sample_products = [
        {
            'id': 'electronic_a7r5', 
            'title': 'Electronic Alpha 7R V Mirrorless Camera',
            'description': '61MP full-frame sensor with advanced AI processing and 8K video recording',
            'category': 'camera',
            'price': 3899.99
        },
        {
            'id': 'electronic_wh1000xm5',
            'title': 'Electronic WH-1000XM5 Wireless Headphones', 
            'description': 'Industry leading noise canceling with 30 hour battery life and premium sound quality',
            'category': 'audio',
            'price': 399.99
        }
    ]
    
    tagged_products = []
    for prod in sample_products:
        tagged_prod = tagging_engine.tag_electronic_product(
            prod['id'], prod['title'], prod['description'], 
            prod['category'], prod['price']
        )
        tagged_products.append(tagged_prod)
        
        print(f"\nProduct: {tagged_prod.title}")
        print(f"  Auto Tags ({len(tagged_prod.tags)}):")
        for tag in tagged_prod.tags[:10]:  # Show first 10 tags
            print(f"    - {tag.tag_name} ({tag.tag_id}) [{tag.confidence:.2f}]")
    
    print("\n2. User Tag Interaction Simulation:")
    print("-" * 50)
    
    user_id = "electronic_user_analytics_demo"
    
    # Simulate user interactions over different time periods
    interactions = [
        ("mirrorless", "electronic_a7r5", InteractionType.VIEW, {}),
        ("mirrorless", "electronic_a7r5", InteractionType.CLICK, {}),
        ("professional", "electronic_a7r5", InteractionType.LIKE, {"engagement": "high"}),
        ("high_end", "electronic_a7r5", InteractionType.ADD_CART, {}),
        ("wireless", "electronic_wh1000xm5", InteractionType.VIEW, {}),
        ("wireless", "electronic_wh1000xm5", InteractionType.CLICK, {}),
        ("noise_canceling", "electronic_wh1000xm5", InteractionType.LIKE, {}),
        ("premium", "electronic_wh1000xm5", InteractionType.PURCHASE, {"conversion": True}),
        ("flagship_audio", "electronic_wh1000xm5", InteractionType.FAVORITE, {}),
        ("mid_range", "electronic_wh1000xm5", InteractionType.COMMENT, {"rating": 5})
    ]
    
    for tag_id, product_id, interaction_type, context in interactions:
        analytics.log_interaction(
            user_id, tag_id, product_id, interaction_type, 
            session_id="demo_session", context_info=context
        )
        print(f"  Logged: {interaction_type.value} on tag '{tag_id}' for product '{product_id}'")
    
    print("\n3. User Tag Statistics Analysis:")
    print("-" * 50)
    
    # Get top tags for user
    top_tags = analytics.get_user_top_tags(user_id, TimeWindow.DAY_7, 10)
    
    print(f"\nTop Tags for {user_id} (7-day window):")
    print("  Rank | Tag ID           | Preference Score | Interactions")
    print("  -----|------------------|------------------|-------------")
    
    for i, (tag_id, score, interactions_dict) in enumerate(top_tags[:5], 1):
        total_interactions = sum(interactions_dict.values())
        print(f"  {i:4} | {tag_id:16} | {score:14.2f} | {total_interactions}")
    
    # Show detailed stats for top tag
    if top_tags:
        top_tag_id = top_tags[0][0]
        detailed_stats = analytics.get_user_tag_stats(user_id, top_tag_id)
        
        print(f"\nDetailed Stats for Tag '{top_tag_id}':")
        print("  Time Window | Views | Clicks | Likes | Purchases | Total")
        print("  ------------|-------|--------|-------|-----------|------")
        
        windows = [('1d', detailed_stats.stats_1d), ('7d', detailed_stats.stats_7d), ('30d', detailed_stats.stats_30d)]
        for window_name, stats in windows:
            views = stats.get('view', 0)
            clicks = stats.get('click', 0) 
            likes = stats.get('like', 0)
            purchases = stats.get('purchase', 0)
            total = sum(stats.values())
            
            print(f"  {window_name:11} | {views:5} | {clicks:6} | {likes:5} | {purchases:9} | {total:5}")
        
        print(f"\nInteraction Ratios:")
        for ratio_name, ratio_value in detailed_stats.interaction_ratios.items():
            if ratio_value > 0:
                print(f"  {ratio_name:20}: {ratio_value:.3f}")
    
    print("\n4. User Pattern Analysis:")
    print("-" * 50)
    
    patterns = analytics.analyze_user_tag_patterns(user_id)
    
    if patterns.get('trending_up'):
        print("\nTrending Up Tags:")
        for tag, score_7d, score_30d in patterns['trending_up'][:3]:
            change_pct = ((score_7d - score_30d) / max(score_30d, 0.01)) * 100
            print(f"  {tag:15}: {change_pct:+6.1f}% change (7d: {score_7d:.2f}, 30d: {score_30d:.2f})")
    
    print("\n" + "=" * 80)
    print("Product Tagging & User Analytics System Ready!")
    print("Features: Auto-tagging, Multi-window stats, Interaction ratios, Trend analysis")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_product_tagging_analytics()
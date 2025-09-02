#!/usr/bin/env python3
"""
User Profile Tag System with Interest Decay for Electronic E-commerce
Comprehensive user profiling with hierarchical tags and decay models
Author: Yang Liu
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import math
from collections import defaultdict
import redis


class TagLevel(Enum):
    L1_CATEGORY = "l1_category"      # Electronics, Gaming, Audio
    L2_SUBCATEGORY = "l2_subcategory"  # Camera, Headphones, PlayStation
    L3_PRODUCT = "l3_product"        # Specific models
    L4_ATTRIBUTE = "l4_attribute"    # Color, Size, Price Range
    BEHAVIORAL = "behavioral"        # Purchase, Browse, Search
    CONTEXTUAL = "contextual"        # Time, Device, Location


class DecayFunction(Enum):
    EXPONENTIAL = "exponential"      # e^(-λt)
    LINEAR = "linear"               # max(0, 1-λt)
    LOGARITHMIC = "logarithmic"     # 1/log(1+λt)
    POWER = "power"                 # t^(-λ)


@dataclass
class TagConfig:
    tag_id: str
    tag_name: str
    tag_level: TagLevel
    parent_tag: Optional[str] = None
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    decay_rate: float = 0.1
    max_lifetime_days: int = 90
    weight_boost: float = 1.0


@dataclass 
class UserInterest:
    tag_id: str
    score: float
    last_updated: float
    interaction_count: int = 0
    first_seen: float = 0.0
    context_info: Dict = field(default_factory=dict)


class ElectronicTagHierarchy:
    """Electronic product tag hierarchy system"""
    
    def __init__(self):
        self.tags = {}
        self.tag_tree = defaultdict(list)
        self._init_electronic_tags()
    
    def _init_electronic_tags(self):
        """Initialize Electronic-specific tag hierarchy"""
        
        # L1 - Top Categories
        l1_tags = [
            TagConfig("camera", "Camera", TagLevel.L1_CATEGORY, decay_rate=0.05),
            TagConfig("audio", "Audio", TagLevel.L1_CATEGORY, decay_rate=0.08),
            TagConfig("gaming", "Gaming", TagLevel.L1_CATEGORY, decay_rate=0.12),
            TagConfig("tv", "TV & Display", TagLevel.L1_CATEGORY, decay_rate=0.06),
            TagConfig("mobile", "Mobile", TagLevel.L1_CATEGORY, decay_rate=0.15),
            TagConfig("professional", "Professional", TagLevel.L1_CATEGORY, decay_rate=0.03)
        ]
        
        # L2 - Subcategories  
        l2_tags = [
            # Camera
            TagConfig("mirrorless", "Mirrorless Camera", TagLevel.L2_SUBCATEGORY, "camera"),
            TagConfig("lens", "Camera Lens", TagLevel.L2_SUBCATEGORY, "camera"),
            TagConfig("camcorder", "Camcorder", TagLevel.L2_SUBCATEGORY, "camera"),
            
            # Audio
            TagConfig("headphones", "Headphones", TagLevel.L2_SUBCATEGORY, "audio"),
            TagConfig("speakers", "Speakers", TagLevel.L2_SUBCATEGORY, "audio"),
            TagConfig("soundbar", "Soundbar", TagLevel.L2_SUBCATEGORY, "audio"),
            TagConfig("walkman", "Walkman", TagLevel.L2_SUBCATEGORY, "audio"),
            
            # Gaming
            TagConfig("playstation", "PlayStation", TagLevel.L2_SUBCATEGORY, "gaming"),
            TagConfig("games", "Games", TagLevel.L2_SUBCATEGORY, "gaming"),
            TagConfig("accessories", "Gaming Accessories", TagLevel.L2_SUBCATEGORY, "gaming"),
            
            # TV
            TagConfig("bravia", "BRAVIA TV", TagLevel.L2_SUBCATEGORY, "tv"),
            TagConfig("projector", "Projector", TagLevel.L2_SUBCATEGORY, "tv"),
            
            # Professional
            TagConfig("broadcast", "Broadcast Equipment", TagLevel.L2_SUBCATEGORY, "professional"),
            TagConfig("cinema", "Cinema Camera", TagLevel.L2_SUBCATEGORY, "professional")
        ]
        
        # L3 - Specific Products
        l3_tags = [
            TagConfig("a7r5", "Alpha 7R V", TagLevel.L3_PRODUCT, "mirrorless"),
            TagConfig("fx3", "FX3", TagLevel.L3_PRODUCT, "cinema"),
            TagConfig("wh1000xm5", "WH-1000XM5", TagLevel.L3_PRODUCT, "headphones"),
            TagConfig("wf1000xm4", "WF-1000XM4", TagLevel.L3_PRODUCT, "headphones"),
            TagConfig("ps5", "PlayStation 5", TagLevel.L3_PRODUCT, "playstation"),
            TagConfig("ps5_pro", "PlayStation 5 Pro", TagLevel.L3_PRODUCT, "playstation")
        ]
        
        # L4 - Attributes
        l4_tags = [
            TagConfig("price_high", "High-end (>$1000)", TagLevel.L4_ATTRIBUTE),
            TagConfig("price_mid", "Mid-range ($300-1000)", TagLevel.L4_ATTRIBUTE), 
            TagConfig("price_low", "Budget (<$300)", TagLevel.L4_ATTRIBUTE),
            TagConfig("color_black", "Black", TagLevel.L4_ATTRIBUTE),
            TagConfig("color_white", "White", TagLevel.L4_ATTRIBUTE),
            TagConfig("wireless", "Wireless", TagLevel.L4_ATTRIBUTE),
            TagConfig("noise_canceling", "Noise Canceling", TagLevel.L4_ATTRIBUTE),
            TagConfig("4k", "4K Support", TagLevel.L4_ATTRIBUTE),
            TagConfig("professional_grade", "Professional Grade", TagLevel.L4_ATTRIBUTE)
        ]
        
        # Behavioral Tags
        behavioral_tags = [
            TagConfig("frequent_buyer", "Frequent Buyer", TagLevel.BEHAVIORAL, decay_rate=0.02),
            TagConfig("price_sensitive", "Price Sensitive", TagLevel.BEHAVIORAL, decay_rate=0.05),
            TagConfig("early_adopter", "Early Adopter", TagLevel.BEHAVIORAL, decay_rate=0.03),
            TagConfig("comparison_shopper", "Comparison Shopper", TagLevel.BEHAVIORAL),
            TagConfig("impulse_buyer", "Impulse Buyer", TagLevel.BEHAVIORAL, decay_rate=0.2),
            TagConfig("brand_loyal", "Electronic Brand Loyal", TagLevel.BEHAVIORAL, decay_rate=0.01)
        ]
        
        # Contextual Tags
        contextual_tags = [
            TagConfig("mobile_user", "Mobile User", TagLevel.CONTEXTUAL, decay_rate=0.3),
            TagConfig("weekend_shopper", "Weekend Shopper", TagLevel.CONTEXTUAL),
            TagConfig("holiday_buyer", "Holiday Season Buyer", TagLevel.CONTEXTUAL, decay_rate=0.5),
            TagConfig("work_hours", "Work Hours Browser", TagLevel.CONTEXTUAL)
        ]
        
        # Register all tags
        all_tags = l1_tags + l2_tags + l3_tags + l4_tags + behavioral_tags + contextual_tags
        
        for tag_config in all_tags:
            self.tags[tag_config.tag_id] = tag_config
            if tag_config.parent_tag:
                self.tag_tree[tag_config.parent_tag].append(tag_config.tag_id)
    
    def get_tag_path(self, tag_id: str) -> List[str]:
        """Get full path from root to tag"""
        path = [tag_id]
        current = self.tags.get(tag_id)
        
        while current and current.parent_tag:
            path.append(current.parent_tag)
            current = self.tags.get(current.parent_tag)
        
        return list(reversed(path))
    
    def get_related_tags(self, tag_id: str) -> List[str]:
        """Get related tags (siblings + children)"""
        tag = self.tags.get(tag_id)
        if not tag:
            return []
        
        related = []
        
        # Add siblings
        if tag.parent_tag:
            siblings = self.tag_tree.get(tag.parent_tag, [])
            related.extend([t for t in siblings if t != tag_id])
        
        # Add children
        children = self.tag_tree.get(tag_id, [])
        related.extend(children)
        
        return related


class InterestDecayCalculator:
    """Calculate interest decay using different functions"""
    
    @staticmethod
    def calculate_decay(days_passed: float, decay_function: DecayFunction, 
                       decay_rate: float) -> float:
        """Calculate decay factor based on time and function"""
        
        if days_passed <= 0:
            return 1.0
        
        if decay_function == DecayFunction.EXPONENTIAL:
            return math.exp(-decay_rate * days_passed)
        
        elif decay_function == DecayFunction.LINEAR:
            return max(0.0, 1.0 - decay_rate * days_passed)
        
        elif decay_function == DecayFunction.LOGARITHMIC:
            return 1.0 / math.log(1 + decay_rate * days_passed)
        
        elif decay_function == DecayFunction.POWER:
            return math.pow(days_passed, -decay_rate) if days_passed > 0 else 1.0
        
        return 0.0
    
    @staticmethod
    def apply_context_boost(base_score: float, context_info: Dict) -> float:
        """Apply context-based score boosts"""
        boost = 1.0
        
        # Recent interaction boost
        if context_info.get('recent_interaction', False):
            boost *= 1.5
        
        # High engagement boost
        if context_info.get('engagement_level', 0) > 0.8:
            boost *= 1.3
        
        # Cross-category interest boost
        if context_info.get('cross_category_interest', False):
            boost *= 1.2
        
        # Purchase history boost
        purchase_count = context_info.get('purchase_count', 0)
        if purchase_count > 0:
            boost *= (1.0 + min(purchase_count * 0.1, 0.5))
        
        return base_score * boost


class UserProfileManager:
    """Comprehensive user profile management with tag system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.tag_hierarchy = ElectronicTagHierarchy()
        self.decay_calculator = InterestDecayCalculator()
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Configuration
        self.min_score_threshold = 0.01
        self.max_tags_per_user = 100
        self.profile_update_interval = 3600  # 1 hour
    
    def update_user_interest(self, user_id: str, tag_id: str, 
                           interaction_strength: float = 1.0,
                           context_info: Dict = None) -> None:
        """Update user interest for a specific tag"""
        current_time = time.time()
        context_info = context_info or {}
        
        # Get current profile
        profile = self.get_user_profile(user_id)
        
        if tag_id in profile:
            interest = profile[tag_id]
            
            # Apply decay to existing score
            days_passed = (current_time - interest.last_updated) / 86400
            tag_config = self.tag_hierarchy.tags[tag_id]
            
            decay_factor = self.decay_calculator.calculate_decay(
                days_passed, tag_config.decay_function, tag_config.decay_rate
            )
            
            # Update with new interaction
            decayed_score = interest.score * decay_factor
            new_score = decayed_score + interaction_strength * tag_config.weight_boost
            
            interest.score = min(new_score, 10.0)  # Cap at 10.0
            interest.last_updated = current_time
            interest.interaction_count += 1
            interest.context_info.update(context_info)
            
        else:
            # Create new interest
            tag_config = self.tag_hierarchy.tags.get(tag_id)
            if not tag_config:
                return
            
            interest = UserInterest(
                tag_id=tag_id,
                score=interaction_strength * tag_config.weight_boost,
                last_updated=current_time,
                first_seen=current_time,
                interaction_count=1,
                context_info=context_info
            )
            profile[tag_id] = interest
        
        # Propagate to parent tags
        self._propagate_to_parents(profile, tag_id, interaction_strength * 0.5, current_time)
        
        # Update related tags
        self._update_related_tags(profile, tag_id, interaction_strength * 0.2, current_time)
        
        # Save profile
        self.save_user_profile(user_id, profile)
    
    def _propagate_to_parents(self, profile: Dict[str, UserInterest], 
                            tag_id: str, strength: float, timestamp: float) -> None:
        """Propagate interest to parent tags"""
        tag_config = self.tag_hierarchy.tags.get(tag_id)
        if not tag_config or not tag_config.parent_tag:
            return
        
        parent_id = tag_config.parent_tag
        
        if parent_id in profile:
            interest = profile[parent_id]
            days_passed = (timestamp - interest.last_updated) / 86400
            parent_config = self.tag_hierarchy.tags[parent_id]
            
            decay_factor = self.decay_calculator.calculate_decay(
                days_passed, parent_config.decay_function, parent_config.decay_rate
            )
            
            interest.score = interest.score * decay_factor + strength
            interest.last_updated = timestamp
            interest.interaction_count += 1
        else:
            profile[parent_id] = UserInterest(
                tag_id=parent_id,
                score=strength,
                last_updated=timestamp,
                first_seen=timestamp,
                interaction_count=1
            )
        
        # Continue propagation
        self._propagate_to_parents(profile, parent_id, strength * 0.5, timestamp)
    
    def _update_related_tags(self, profile: Dict[str, UserInterest],
                           tag_id: str, strength: float, timestamp: float) -> None:
        """Update related tags with smaller boost"""
        related_tags = self.tag_hierarchy.get_related_tags(tag_id)
        
        for related_id in related_tags:
            if related_id in profile:
                interest = profile[related_id]
                interest.score += strength * 0.1
                interest.last_updated = timestamp
    
    def get_user_profile(self, user_id: str) -> Dict[str, UserInterest]:
        """Get user profile from cache"""
        try:
            profile_data = self.redis_client.hgetall(f"user_profile:{user_id}")
            
            profile = {}
            for tag_id, interest_json in profile_data.items():
                interest_dict = json.loads(interest_json)
                profile[tag_id] = UserInterest(**interest_dict)
            
            return profile
            
        except Exception as e:
            print(f"Error loading profile for {user_id}: {e}")
            return {}
    
    def save_user_profile(self, user_id: str, profile: Dict[str, UserInterest]) -> None:
        """Save user profile to cache"""
        try:
            # Clean up expired/low-score interests
            cleaned_profile = self._cleanup_profile(profile)
            
            # Convert to JSON for Redis storage
            profile_data = {}
            for tag_id, interest in cleaned_profile.items():
                profile_data[tag_id] = json.dumps({
                    'tag_id': interest.tag_id,
                    'score': interest.score,
                    'last_updated': interest.last_updated,
                    'interaction_count': interest.interaction_count,
                    'first_seen': interest.first_seen,
                    'context_info': interest.context_info
                })
            
            # Save to Redis
            if profile_data:
                self.redis_client.hmset(f"user_profile:{user_id}", profile_data)
                self.redis_client.expire(f"user_profile:{user_id}", 86400 * 180)  # 6 months TTL
            
        except Exception as e:
            print(f"Error saving profile for {user_id}: {e}")
    
    def _cleanup_profile(self, profile: Dict[str, UserInterest]) -> Dict[str, UserInterest]:
        """Clean up expired and low-score interests"""
        current_time = time.time()
        cleaned = {}
        
        for tag_id, interest in profile.items():
            tag_config = self.tag_hierarchy.tags.get(tag_id)
            if not tag_config:
                continue
            
            # Check expiration
            days_since_update = (current_time - interest.last_updated) / 86400
            if days_since_update > tag_config.max_lifetime_days:
                continue
            
            # Apply current decay
            decay_factor = self.decay_calculator.calculate_decay(
                days_since_update, tag_config.decay_function, tag_config.decay_rate
            )
            current_score = interest.score * decay_factor
            
            # Keep if above threshold
            if current_score >= self.min_score_threshold:
                interest.score = current_score
                cleaned[tag_id] = interest
        
        # Limit number of tags
        if len(cleaned) > self.max_tags_per_user:
            sorted_interests = sorted(cleaned.items(), key=lambda x: x[1].score, reverse=True)
            cleaned = dict(sorted_interests[:self.max_tags_per_user])
        
        return cleaned
    
    def get_top_interests(self, user_id: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get user's top interests with current scores"""
        profile = self.get_user_profile(user_id)
        current_time = time.time()
        
        scored_interests = []
        for tag_id, interest in profile.items():
            tag_config = self.tag_hierarchy.tags.get(tag_id)
            if not tag_config:
                continue
            
            # Apply current decay
            days_passed = (current_time - interest.last_updated) / 86400
            decay_factor = self.decay_calculator.calculate_decay(
                days_passed, tag_config.decay_function, tag_config.decay_rate
            )
            
            # Apply context boost
            current_score = interest.score * decay_factor
            boosted_score = self.decay_calculator.apply_context_boost(
                current_score, interest.context_info
            )
            
            if boosted_score >= self.min_score_threshold:
                scored_interests.append((tag_id, boosted_score))
        
        # Sort by score and return top-k
        scored_interests.sort(key=lambda x: x[1], reverse=True)
        return scored_interests[:top_k]
    
    def get_category_preferences(self, user_id: str) -> Dict[str, float]:
        """Get user preferences by category"""
        top_interests = self.get_top_interests(user_id, top_k=50)
        category_scores = defaultdict(float)
        
        for tag_id, score in top_interests:
            tag_config = self.tag_hierarchy.tags.get(tag_id)
            if not tag_config:
                continue
            
            # Find L1 category for this tag
            tag_path = self.tag_hierarchy.get_tag_path(tag_id)
            if tag_path:
                l1_category = tag_path[0]
                category_scores[l1_category] += score
        
        return dict(category_scores)
    
    def recommend_tags_for_user(self, user_id: str) -> List[str]:
        """Recommend new tags user might be interested in"""
        profile = self.get_user_profile(user_id)
        current_interests = set(profile.keys())
        
        # Get related tags from current interests
        candidate_tags = set()
        for tag_id in current_interests:
            related = self.tag_hierarchy.get_related_tags(tag_id)
            candidate_tags.update(related)
        
        # Remove already interested tags
        new_candidates = candidate_tags - current_interests
        
        # Score candidates based on related interest strength
        scored_candidates = []
        for candidate in new_candidates:
            related_score = 0
            for tag_id, interest in profile.items():
                if candidate in self.tag_hierarchy.get_related_tags(tag_id):
                    related_score += interest.score * 0.1
            
            if related_score > 0:
                scored_candidates.append((candidate, related_score))
        
        # Return top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [tag_id for tag_id, _ in scored_candidates[:10]]


def demonstrate_user_profile_system():
    """Demonstrate the user profile tag system"""
    print("=" * 80)
    print("Electronic User Profile Tag System with Interest Decay")
    print("=" * 80)
    
    # Initialize system
    profile_manager = UserProfileManager()
    
    print("\n1. Tag Hierarchy Overview:")
    print("-" * 40)
    for level in TagLevel:
        count = sum(1 for tag in profile_manager.tag_hierarchy.tags.values() 
                   if tag.tag_level == level)
        print(f"   {level.value}: {count} tags")
    
    print("\n2. Sample User Interest Updates:")
    print("-" * 40)
    
    user_id = "electronic_user_12345"
    
    # Simulate user interactions
    interactions = [
        ("a7r5", 2.0, {"engagement_level": 0.9, "recent_interaction": True}),
        ("mirrorless", 1.5, {"engagement_level": 0.7}),
        ("camera", 1.0, {"engagement_level": 0.6}),
        ("price_high", 0.8, {"purchase_count": 1}),
        ("professional_grade", 1.2, {"engagement_level": 0.8}),
        ("wh1000xm5", 1.8, {"recent_interaction": True}),
        ("headphones", 1.3, {}),
        ("audio", 0.9, {}),
        ("noise_canceling", 1.0, {"engagement_level": 0.7})
    ]
    
    for tag_id, strength, context in interactions:
        profile_manager.update_user_interest(user_id, tag_id, strength, context)
        print(f"   Updated interest: {tag_id} (strength: {strength})")
    
    print("\n3. User's Top Interests:")
    print("-" * 40)
    top_interests = profile_manager.get_top_interests(user_id, top_k=10)
    for i, (tag_id, score) in enumerate(top_interests, 1):
        tag_name = profile_manager.tag_hierarchy.tags[tag_id].tag_name
        print(f"   {i:2}. {tag_name:20} ({tag_id:15}) Score: {score:.3f}")
    
    print("\n4. Category Preferences:")
    print("-" * 40)
    category_prefs = profile_manager.get_category_preferences(user_id)
    for category, score in sorted(category_prefs.items(), key=lambda x: x[1], reverse=True):
        category_name = profile_manager.tag_hierarchy.tags[category].tag_name
        print(f"   {category_name:15}: {score:.3f}")
    
    print("\n5. Recommended New Interests:")
    print("-" * 40)
    recommended = profile_manager.recommend_tags_for_user(user_id)
    for tag_id in recommended[:5]:
        if tag_id in profile_manager.tag_hierarchy.tags:
            tag_name = profile_manager.tag_hierarchy.tags[tag_id].tag_name
            print(f"   {tag_name} ({tag_id})")
    
    print("\n6. Interest Decay Examples:")
    print("-" * 40)
    
    # Show how different decay functions work over time
    days_range = [1, 7, 30, 90]
    decay_functions = [DecayFunction.EXPONENTIAL, DecayFunction.LINEAR, DecayFunction.LOGARITHMIC]
    
    print("   Days | Exponential | Linear    | Logarithmic")
    print("   -----|-------------|-----------|------------")
    
    for days in days_range:
        exp_decay = InterestDecayCalculator.calculate_decay(days, DecayFunction.EXPONENTIAL, 0.1)
        lin_decay = InterestDecayCalculator.calculate_decay(days, DecayFunction.LINEAR, 0.01)
        log_decay = InterestDecayCalculator.calculate_decay(days, DecayFunction.LOGARITHMIC, 0.1)
        
        print(f"   {days:4} | {exp_decay:10.3f} | {lin_decay:8.3f} | {log_decay:10.3f}")
    
    print("\n" + "=" * 80)
    print("User Profile Tag System Ready!")
    print("Features: Hierarchical tags, Multiple decay models, Context awareness")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_user_profile_system()